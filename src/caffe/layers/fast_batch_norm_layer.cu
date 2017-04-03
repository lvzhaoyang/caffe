#include <algorithm>
#include <vector>

#include "caffe/layers/frcnn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void multicast_add_gpu_kernel(const int n, const int CS, const int S, 
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int batch_offset = index % CS;
    const int channel_index = batch_offset / S;
    out[index] += in[channel_index];
  }
}

template <typename Dtype>
__global__ void multicast_mul_gpu_kernel(const int n, const int CS, const int S, 
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int batch_offset = index % CS;
    const int channel_index = batch_offset / S;
    out[index] *= in[channel_index];
  }
}

//  multicast x[c] into x[.,c,...] and add to y[.,c,...]
template <typename Dtype>
void FastBatchNormLayer<Dtype>::multicast_add_gpu(const int N, const int C, const int S,
      const Dtype *x, Dtype *y) {
  const int count = N * batch_offset_;
  const int CS = batch_offset_;
  multicast_add_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, CS, S, x, y);
  CUDA_POST_KERNEL_CHECK;
}

//  multicast x[c] into x[.,c,...] and multiply with y[.,c,...]
template <typename Dtype>
void FastBatchNormLayer<Dtype>::multicast_mul_gpu(const int N, const int C, const int S,
      const Dtype *x, Dtype *y) {
  const int count = N * batch_offset_;
  const int CS = batch_offset_;
  multicast_mul_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, CS, S, x, y);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void FastBatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  int num = bottom[0]->shape(0);
  int spatial_dim = channel_offset_;

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
 
  // mean = -EX
  caffe_gpu_scale(mean_.count(), Dtype(-1.0),
        this->blobs_[2]->gpu_data(), mean_.mutable_gpu_data());
  caffe_copy(variance_.count(), this->blobs_[3]->gpu_data(), 
        variance_.mutable_gpu_data());
  
  // inv_var = ( eps + variance )^(-0.5)
  caffe_gpu_add_scalar(channels_, eps_, variance_.mutable_gpu_data());
  caffe_gpu_powx(channels_, variance_.gpu_data(), Dtype(-0.5), 
      inv_variance_.mutable_gpu_data());

  // X - EX
  multicast_add_gpu(num, channels_, spatial_dim, 
      mean_.gpu_data(), top_data);

  // X_norm = (X-EX) * inv_var
  multicast_mul_gpu(num, channels_, spatial_dim, 
      inv_variance_.gpu_data(), top_data);

  // Y = X_norm * scale
  multicast_mul_gpu(num, channels_, spatial_dim, 
      this->blobs_[0]->gpu_data(), top_data);

  // Y = Y + bias
  multicast_add_gpu(num, channels_, spatial_dim, 
      this->blobs_[1]->gpu_data(), top_data);
}

template <typename Dtype>
void FastBatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  int num = bottom[0]->shape(0);
  int spatial_dim = channel_offset_;

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  }

  // dE/d(X_norm) = dE/dY * scale[c]
  multicast_mul_gpu(num, channels_, spatial_dim, 
      this->blobs_[0]->gpu_data(), bottom_diff);

  // d(E)/d(X) = dE/d(X_norm) * inv_var
  multicast_mul_gpu(num, channels_, spatial_dim, 
      inv_variance_.gpu_data(), bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(FastBatchNormLayer);

// explicte instantiate gpu helper functions
template void FastBatchNormLayer<float>::multicast_add_gpu(const int N, const int C,
    const int S, const float* x, float* y);
template void FastBatchNormLayer<double>::multicast_add_gpu(const int N, const int C,
    const int S, const double* x, double* y);

template void FastBatchNormLayer<float>::multicast_mul_gpu(const int N, const int C,
    const int S, const float* x, float* y);
template void FastBatchNormLayer<double>::multicast_mul_gpu(const int N, const int C,
    const int S, const double* x, double* y);

}  // namespace caffe
