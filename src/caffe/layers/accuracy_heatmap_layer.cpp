#include <functional>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/accuracy_heatmap_layer.hpp"
#include "caffe/util/math_functions.hpp"

//Calculate PCK accuracy
//By FangHaoshu

namespace caffe {

template <typename Dtype>
void AccuracyHeatmapLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(),bottom[1]->num()*bottom[1]->channels()*3) << "Prediction size must equal to label's num x channel x 3";
  thres = this->layer_param_.accuracy_heatmap_param().threshold();
}

template <typename Dtype>
void AccuracyHeatmapLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void AccuracyHeatmapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    Dtype* accuracy = top[0]->mutable_cpu_data();
    cv::Mat heatmap;

    const Dtype* pred_data = bottom[0]->cpu_data(); // predictions
    const Dtype* label = bottom[1]->cpu_data();     // labels
    const int num_images = bottom[1]->num();
    const int num_channels = bottom[1]->channels();
    const int height = bottom[1]->height();
    const int width = bottom[1]->width();

    DLOG(INFO) << "label size: " << bottom[0]->height() << " " << bottom[0]->width() << " " << bottom[0]->channels();

    const int channel_size = height * width;
    const int img_size = channel_size * num_channels;
    const int output_size = height;

    int point_count = 0;
    int correct_point = 0;
    // Loop over images
    for (int idx_img = 0; idx_img < num_images; idx_img++)
    {
        // Each channel represents a point
        for (int idx_ch = 0; idx_ch < num_channels; idx_ch++)
        {
            heatmap = cv::Mat::zeros(height, width, CV_32FC1);
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    // Store visualisation for all channels                  
                    int image_idx = idx_img * img_size + idx_ch * channel_size + i * width + j;
                    heatmap.at<float>((int)i, (int)j) = (float) label[image_idx];
                }           
            }

            // Store the point and score of each heatmap
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(heatmap, &minVal, &maxVal, &minLoc, &maxLoc);

            cv::Point pred_point;
            pred_point.x = pred_data[3*num_channels*idx_img + 3*idx_ch + 0];
            pred_point.y = pred_data[3*num_channels*idx_img + 3*idx_ch + 1];
            if(maxLoc.x > 1 and maxLoc.y > 1)
            {
              point_count++;
              correct_point += (cv::norm(maxLoc - pred_point)/(output_size/10))<thres ? 1 : 0;
            }
        }
    }

    accuracy[0] = correct_point / (double)point_count;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AccuracyHeatmapLayer);
REGISTER_LAYER_CLASS(AccuracyHeatmap);

}  // namespace caffe
