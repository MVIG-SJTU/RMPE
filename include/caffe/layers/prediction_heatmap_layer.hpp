#ifndef CAFFE_PREDICTION_HEATMAP_HPP_
#define CAFFE_PREDICTION_HEATMAP_HPP_

#include "caffe/layer.hpp"
#include <vector>
#include <boost/timer/timer.hpp>
#include <opencv2/core/core.hpp>

#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{


template <typename Dtype>
class PredictionHeatmapLayer : public Layer<Dtype> {
public:
  explicit PredictionHeatmapLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DetectionHeatmap"; }


protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void PrepVis(cv::Mat img, cv::Size size);

  virtual void Visualise(cv::Mat bottom_img, cv::Size size);

  virtual void VisualiseBottom(const vector<Blob<Dtype>*>& bottom, int idx_img, std::vector<cv::Mat>& heatmaps, cv::Size size);


  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    NOT_IMPLEMENTED;
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    NOT_IMPLEMENTED;
  }
};


}

#endif /* CAFFE_PREDICTION_HEATMAP_HPP_ */