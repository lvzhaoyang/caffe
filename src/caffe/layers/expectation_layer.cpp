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
  top[0]->Reshape(scale_dims);
}

template <typename Dtype>
void ExpectationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* prob = bottom[0]->cpu_data();
  const Dtype* coefficients= bottom[1]->cpu_data();

  Dtype* expectation = top[0]->mutable_cpu_data();
  int channels = bottom[0]->shape(expectation_axis_);
  int dim = bottom[0]->count() / outer_num_;

  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      Dtype value = 0;
      for (int c = 0; c < channels; ++c) {
        value += coefficients[c] * prob[i * dim + c*inner_num_ + j];
      }
      expectation[i*inner_num_ + j] = value;
    }
  }
}

template <typename Dtype>
void ExpectationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* prob_diff = bottom[0]->mutable_cpu_diff();
  Dtype* weight_diff = bottom[1]->mutable_cpu_diff();
  const Dtype* coefficients = bottom[1]->cpu_data();

  int dim = bottom[0]->count() / outer_num_;
  int channels = bottom[0]->shape(expectation_axis_);

  // propagate to weights

  // propagate to probabilities
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      for (int c = 0; c < channels; ++c) {
        Dtype weight = coefficients[c];
        prob_diff[i * dim + c*inner_num_ + j] =
            top_diff[i*dim + c*inner_num_ + j] * weight;
        weight_diff[c] = 1;
      }
    }
  }

}

#ifdef CPU_ONLY
STUB_GPU(ExpectationLayer);
#endif

INSTANTIATE_CLASS(ExpectationLayer);

} // namespace caffe
