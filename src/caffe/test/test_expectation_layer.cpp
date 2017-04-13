#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/expectation_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ExpectationLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
protected:
  ExpectationLayerTest()
    : blob_bottom_(new Blob<Dtype>(2, 10, 2, 3)),
      blob_weight_(new Blob<Dtype>(1, 10, 1, 1)),
      blob_top_(new Blob<Dtype>()),
      blob_top_gt_(new Blob<Dtype>(2, 1, 2, 3)) {
    // fill the values of blobs
    FillerParameter filler_param;
    PositiveUnitballFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    FillerParameter weight_filler_param;
    UniformFiller<Dtype> weight_filler(weight_filler_param);
    weight_filler_param.set_min(-10);
    weight_filler_param.set_max(10);
    weight_filler.Fill(this->blob_weight_);


    const Dtype* bottom = blob_bottom_->cpu_data();
    const Dtype* weight = blob_weight_->cpu_data();
    Dtype* gt = blob_top_gt_->mutable_cpu_data();

    // calculate the ground truth value blob_top
    int inner_num = 2*3;
    int outer_num = 10*2*3;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < inner_num; j++) {
        Dtype value = 0;
        for (int c = 0; c < 10; ++c) {
          value += weight[c] * bottom[i * outer_num + c*inner_num + j];
        }
        gt[i*2*3 + j] = value;
      }
    }

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_weight_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ExpectationLayerTest() {
    delete blob_bottom_;
    delete blob_weight_;
    delete blob_top_;
    delete blob_top_gt_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_weight_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* blob_top_gt_;         ///< ground truth top value
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ExpectationLayerTest, TestDtypesAndDevices);

TYPED_TEST(ExpectationLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ExpectationLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test expectation sum
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        // expectation value
        Dtype expectation = this->blob_top_->data_at(i, 0, h, w);

        // expectation should be within the range [-10, 10]
        EXPECT_GE(expectation, -10);
        EXPECT_LE(expectation, 10);
        // Test exact values
        EXPECT_EQ(this->blob_top_gt_->offset(i, 0, h, w),
                  this->blob_top_->offset(i, 0, h, w));
      }
    }
  }
}

TYPED_TEST(ExpectationLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ExpectationLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}
