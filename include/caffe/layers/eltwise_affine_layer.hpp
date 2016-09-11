#ifndef CAFFE_HEATMAP_HPP_
#define CAFFE_HEATMAP_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include <vector>
#include <boost/timer/timer.hpp>

#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{


template <typename Dtype>
class EltwiseAffineLayer : public NeuronLayer<Dtype> {
     public:
      /**
       * @param param provides EltwiseAffineParameter EltwiseAffine_param,
       *     with EltwiseAffineLayer options:
       *   - slope_filler (\b optional, FillerParameter,
        *     default {'type': constant 'value':1.0001}).
      *   - bias_filler (\b optional, FillerParameter,
        *     default {'type': constant 'value':0.0001}).
      *   - channel_shared (\b optional, default false).
      *     slopes and biases are shared across channels.
      */
     explicit EltwiseAffineLayer(const LayerParameter& param)
         : NeuronLayer<Dtype>(param) {}
   
     virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
   
     virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
   
     virtual inline const char* type() const { return "EltwiseAffine"; }
   
    protected:
     /**
      * @param bottom input Blob vector (length 1)
      *   -# @f$ (N \times C \times ...) @f$
      *      the inputs @f$ x @f$
      * @param top output Blob vector (length 1)
      *   -# @f$ (N \times C \times ...) @f$
      *      the computed outputs for each channel @f$i@f$ @f$
      *        y_i = a_i x_i + b_i
      *      @f$.
      */
     virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
     virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
   
     /**
      * @brief Computes the error gradient w.r.t. the EltwiseAffine inputs.
      *
      * @param top output Blob vector (length 1), providing the error gradient with
      *      respect to the outputs
      *   -# @f$ (N \times C \times ...) @f$
      *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
      *      with respect to computed outputs @f$ y @f$
      * @param propagate_down see Layer::Backward.
      * @param bottom input Blob vector (length 1)
      *   -# @f$ (N \times C \times ...) @f$
      *      the inputs @f$ x @f$; For each channel @f$i@f$, backward fills their
      *      diff with gradients @f$
      *        \frac{\partial E}{\partial x_i} = \left\{
       *        \begin{array}{lr}
          *            a_i \frac{\partial E}{\partial y_i}
          *        \end{array} \right.
          *      @f$.
          *      If param_propagate_down_[0] is true, it fills the diff with gradients
          *      @f$
          *        \frac{\partial E}{\partial a_i} = \left\{
             *        \begin{array}{lr}
             *            \sum_{x_i} x_i \frac{\partial E}{\partial y_i}
              *        \end{array} \right.
              *      @f$.
              *      If param_propagate_down_[1] is true, it fills the diff with gradients
              *      @f$
              *        \frac{\partial E}{\partial b_i} = \left\{
                  *        \begin{array}{lr}
                  *             frac{\partial E}{\partial y_i}
                  *        \end{array} \right.
                  *      @f$.
                  */
                 virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
                 virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
               
                 bool channel_shared_;
                 Blob<Dtype> multiplier_;  // dot multiplier for backward computation of params
                 Blob<Dtype> bias_multiplier_;
                 Blob<Dtype> backward_buff_;  // temporary buffer for backward computation
                 Blob<Dtype> bottom_memory_;  // memory for in-place computation
               };
  

}// namespace caffe

#endif /* CAFFE_HEATMAP_HPP_ */
