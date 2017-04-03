#include <algorithm>
#include <vector>

#include "caffe/layers/frcnn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FastBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  FastBatchNormParameter param = this->layer_param_.fast_batch_norm_param();
  
  if (bottom[0]->num_axes() == 1) {
    channels_ = 1;
    batch_offset_ = 1;
    channel_offset_ = 1;
  }
  else {
    channels_ = bottom[0]->shape(1);
    batch_offset_ = 1;
    channel_offset_ = 1;
  }
  eps_ = param.eps();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(4);
    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob<Dtype>(sz)); // scale 
    this->blobs_[1].reset(new Blob<Dtype>(sz)); // bias
    this->blobs_[2].reset(new Blob<Dtype>(sz)); // mean
    this->blobs_[3].reset(new Blob<Dtype>(sz)); // variance

    // set scale, bias, mean and var to 0
    for (int i = 0; i < 4; ++i) {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
                this->blobs_[i]->mutable_cpu_data());
    }
  }

}

template <typename Dtype>
void FastBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() >= 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);

  vector<int> sz;
  sz.push_back(channels_);
  mean_.Reshape(sz);
  variance_.Reshape(sz);
  inv_variance_.Reshape(sz);
  
  // updating offest with regard to bottom[0]
  int N = bottom[0]->shape(0);
  int NC = N* channels_;

  batch_offset_ = bottom[0]->count() / N;
  channel_offset_ = bottom[0]->count() / NC;
}

//  multicast x[c] into x[.,c,...] and add to y[.,c,...]
template <typename Dtype>
void FastBatchNormLayer<Dtype>::multicast_add_cpu(const int N, const int C, const int S,
      const Dtype *x, Dtype *y ) {
  for (int i=0; i<N; i++) 
    for (int j=0; j<C; j++) 
      for (int k=0; k<S; k++) 
        y[i*batch_offset_ + j*channel_offset_ + k] += x[j];
}

//  multicast x[c] into x[.,c,...] and multiply with y[.,c,...]
template <typename Dtype>
void FastBatchNormLayer<Dtype>::multicast_mul_cpu(const int N, const int C, const int S,
      const Dtype *x, Dtype *y ) {
  for (int i=0; i<N; i++) 
    for (int j=0; j<C; j++) 
      for (int k=0; k<S; k++) 
        y[i*batch_offset_ + j*channel_offset_ + k] *= x[j];
}

template <typename Dtype>
void FastBatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
 
  // mean = -EX
  caffe_cpu_scale(mean_.count(), Dtype(-1.0),
        this->blobs_[2]->cpu_data(), mean_.mutable_cpu_data());
  caffe_copy(variance_.count(), this->blobs_[3]->cpu_data(), 
        variance_.mutable_cpu_data());
  
  // inv_var = ( eps + variance )^(-0.5)
  caffe_add_scalar(channels_, eps_, variance_.mutable_cpu_data());
  caffe_powx(channels_, variance_.cpu_data(), Dtype(-0.5), 
        inv_variance_.mutable_cpu_data());

  // X - EX
  multicast_add_cpu(num, channels_, spatial_dim, 
        mean_.cpu_data(), top_data);

  // X_norm = (X-EX) * inv_var
  multicast_mul_cpu(num, channels_, spatial_dim, 
        inv_variance_.cpu_data(), top_data);

  // Y = X_norm * scale
  multicast_mul_cpu(num, channels_, spatial_dim, 
        this->blobs_[0]->cpu_data(), top_data);

  // Y = Y + bias
  multicast_add_cpu(num, channels_, spatial_dim, 
        this->blobs_[1]->cpu_data(), top_data);

}

template <typename Dtype>
void FastBatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  }

  // dE/d(X_norm) = dE/dY * scale[c]
  multicast_mul_cpu(num, channels_, spatial_dim, 
      this->blobs_[0]->cpu_data(), bottom_diff);

  // d(E)/d(X) = dE/d(X_norm) * inv_var
  multicast_mul_cpu(num, channels_, spatial_dim, 
      inv_variance_.cpu_data(), bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(FastBatchNormLayer);
#endif

INSTANTIATE_CLASS(FastBatchNormLayer);
REGISTER_LAYER_CLASS(FastBatchNorm);
}  // namespace caffe
