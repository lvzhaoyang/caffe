#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"
#include "caffe/layers/expectation_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_channel_weighted_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, const Dtype* coefficients, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
        double w = coefficients[c];
        sum += w * data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
void ExpectationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* coefficients= bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    int count = bottom[0]->count();
    int channels = top[0]->shape(expectation_axis_);

    // sum operation
    kernel_channel_weighted_sum<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
        CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, bottom_data,
                                  coefficients, top_data);
}

template <typename Dtype>
void ExpectationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to coefficients inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();

  }


}

INSTANTIATE_LAYER_GPU_FUNCS(ExpectationLayer);

} // namespace caffe
