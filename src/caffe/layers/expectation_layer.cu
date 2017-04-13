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
__global__ void kernel_channel_div(const int count, const int outer_num,
    const int inner_num, const Dtype* input, const Dtype* coefficients, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    int c = (index % outer_num) % inner_num;
    out[index] = input[index] * coefficients[c];
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
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* weight = bottom[1]->gpu_data();
    int count = bottom[0]->count();
    kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, inner_num_,
                                  top_diff, weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ExpectationLayer);

} // namespace caffe
