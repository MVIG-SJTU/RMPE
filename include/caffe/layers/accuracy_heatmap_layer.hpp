#ifndef CAFFE_ACCURACY_HEATMAP_LAYER_HPP_
#define CAFFE_ACCURACY_HEATMAP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the PCK accuracy for a point prediction
 *        task. By FangHaoshu
 */
template <typename Dtype>
class AccuracyHeatmapLayer : public Layer<Dtype> {
 public:
  explicit AccuracyHeatmapLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AccuracyHeatmap"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

 protected:
  /**
  *   bottom[0] is the output of PredictionHeatmapLayer
  *   bottom[1] is the label of DataHeatmapLayer
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }
};
  
  float thres;
}  // namespace caffe

#endif  // CAFFE_ACCURACY_HEATMAP_LAYER_HPP_
