#include <algorithm>
#include <vector>

#include "caffe/layers/expectation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ExpectationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  expectation_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.expectation_param().axis());
  outer_num_ = bottom[0]->count(0, expectation_axis_);
  inner_num_ = bottom[0]->count(expectation_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[expectation_axis_] = 1;
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void ExpectationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* coefficients= bottom[1]->cpu_data();

  Dtype* scale_data=scale_.mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int channels = bottom[0]->shape(expectation_axis_);
  int dim = bottom[0]->count() / outer_num_;
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }

    // compute expectation
    caffe_cpu_gemv<Dtype>(CblasNoTrans, 1, channels, 1. , coefficients, scale_data, 0., top_data);
  }
}

template <typename Dtype>
void ExpectationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to coefficients inputs.";
  }
  if (propagate_down[0]) {

  }
}

#ifdef CPU_ONLY
STUB_GPU(ExpectationLayer);
#endif

INSTANTIATE_CLASS(ExpectationLayer);

} // namespace caffe
