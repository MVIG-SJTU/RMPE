#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/prediction_heatmap_layer.hpp"

// Enables visualisation of inputs and predictions.


namespace caffe {

template <typename Dtype>
void PredictionHeatmapLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    vector<int> top_shape(2, 1);
    const int num_person = bottom[0]->num();
    const int num_joints = bottom[0]->channels();
    top_shape.push_back(num_person);
    top_shape.push_back(3*num_joints);  //  (x,y) pair + score for each body joint, for mpii dataset, it's 32 + 16 = 48
    top[0]->Reshape(top_shape);
}


template<typename Dtype>
void PredictionHeatmapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    PredictionHeatmapParameter prediction_heatmap_param = this->layer_param_.prediction_heatmap_param();
    CHECK(prediction_heatmap_param.has_num_joints()) << "Must specify num_joints";
    CHECK_EQ(prediction_heatmap_param.num_joints(),bottom[0]->channels()) << "'num_joints' must equals to final number of channels";
}

template <typename Dtype>
void PredictionHeatmapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
{
    int visualise_channel = this->layer_param_.visualise_channel();
    bool visualise = this->layer_param_.visualise();
    Dtype* top_data = top[0]->mutable_cpu_data();
    
    const Dtype* bottom_pred = bottom[0]->cpu_data(); // predictions for all images
    const int num_images = bottom[0]->num();
    const int num_channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();


    DLOG(INFO) << "bottom size: " << bottom[0]->height() << " " << bottom[0]->width() << " " << bottom[0]->channels();

    const int channel_size = height * width;
    const int img_size = channel_size * num_channels;
    cv::Mat bottom_img;  // Initialise opencv images for visualisation
    cv::Mat heatmap;
    if (visualise)
    {
        cv::namedWindow("bottom", CV_WINDOW_AUTOSIZE);
        cv::namedWindow("visualisation_bottom", CV_WINDOW_AUTOSIZE);
        bottom_img = cv::Mat::zeros(height, width, CV_32FC1);
    }

    // Loop over images
    for (int idx_img = 0; idx_img < num_images; idx_img++)
    {
        std::vector<cv::Mat> heatmaps;  //heatmaps will store the heatmap of each person joint
        // Each channel represents a point
        for (int idx_ch = 0; idx_ch < num_channels; idx_ch++)
        {
            heatmap = cv::Mat::zeros(height, width, CV_32FC1);;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    // Store visualisation for all channels                  
                    int image_idx = idx_img * img_size + idx_ch * channel_size + i * width + j;
                    if (idx_ch == visualise_channel && visualise)
                    {
                        bottom_img.at<float>((int)j, (int)i) += (float) bottom_pred[image_idx];
                    }
                    heatmap.at<float>((int)j, (int)i) = (float) bottom_pred[image_idx];
                }           
            }
            heatmaps.push_back(heatmap);

            // Store the point and score of each heatmap
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(heatmap, &minVal, &maxVal, &minLoc, &maxLoc);
            top_data[3*num_channels*idx_img + 3*idx_ch + 0] = maxLoc.y;//It's x in real space
            top_data[3*num_channels*idx_img + 3*idx_ch + 1] = maxLoc.x;//It's y in real space
            top_data[3*num_channels*idx_img + 3*idx_ch + 2] = maxVal;
        }
        // Plot visualisation
        if (visualise)
        {
//            DLOG(INFO) << "num_images=" << num_images << " idx_img=" << idx_img;
//            DLOG(INFO) << "sum bottom: " << cv::sum(bottom_img) << "  sum gt: " << cv::sum(gt_img);
            int visualisation_size = 256;
            cv::Size size(visualisation_size, visualisation_size);            
            this->Visualise(bottom_img, size);
            this->VisualiseBottom(bottom, idx_img, heatmaps, size);
            cv::waitKey(0);     // Wait forever a key is pressed
        }
    }
}


template <typename Dtype>
void PredictionHeatmapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
{
    Forward_cpu(bottom, top);
}

// Visualise predicted heatmap
template <typename Dtype>
void PredictionHeatmapLayer<Dtype>::Visualise(cv::Mat bottom_img, cv::Size size)
{
    // Resize all images to fixed size
    PrepVis(bottom_img, size);
    cv::resize(bottom_img, bottom_img, size);

    // Show visualisation images
    cv::imshow("bottom", bottom_img);
}

// Plot another visualisation image overlaid with ground truth & prediction locations
// (particularly useful e.g. if you set this to the original input image)
template <typename Dtype>
void PredictionHeatmapLayer<Dtype>::VisualiseBottom(const vector<Blob<Dtype>*>& bottom, int idx_img, std::vector<cv::Mat>& heatmaps, cv::Size size)
{
    // Determine which layer to visualise
    Blob<Dtype>* visualisation_bottom = bottom[1];
    DLOG(INFO) << "visualisation_bottom: " << visualisation_bottom->channels() << " " << visualisation_bottom->height() << " " << visualisation_bottom->width();

    // Format as RGB / gray
    bool isRGB = visualisation_bottom->channels() == 3;
    cv::Mat visualisation_bottom_img;
    if (isRGB)
        visualisation_bottom_img = cv::Mat::zeros(visualisation_bottom->height(), visualisation_bottom->width(), CV_32FC3);
    else
        visualisation_bottom_img = cv::Mat::zeros(visualisation_bottom->height(), visualisation_bottom->width(), CV_32FC1);

    // Convert frame from Caffe representation to OpenCV image
    for (int idx_ch = 0; idx_ch < visualisation_bottom->channels(); idx_ch++)
    {
        for (int i = 0; i < visualisation_bottom->height(); i++)
        {
            for (int j = 0; j < visualisation_bottom->width(); j++)
            {
                int image_idx = idx_img * visualisation_bottom->width() * visualisation_bottom->height() * visualisation_bottom->channels() + idx_ch * visualisation_bottom->width() * visualisation_bottom->height() + i * visualisation_bottom->height() + j;
                if (isRGB && idx_ch < 3) {
                    visualisation_bottom_img.at<cv::Vec3f>((int)j, (int)i)[idx_ch] = (float) visualisation_bottom->cpu_data()[image_idx];
                } else if (idx_ch == 0)
                {
                    visualisation_bottom_img.at<float>((int)j, (int)i) = (float) visualisation_bottom->cpu_data()[image_idx];
                }
            }
        }

    }
    PrepVis(visualisation_bottom_img, size);

    // Convert colouring if RGB
    if (isRGB)
        cv::cvtColor(visualisation_bottom_img, visualisation_bottom_img, CV_RGB2BGR);

    // Plot prediction
    for(int joint_id = 0; joint_id < bottom[0]->channels(); joint_id++)
    {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(heatmaps[joint_id], &minVal, &maxVal, &minLoc, &maxLoc);
        cv::circle(visualisation_bottom_img, maxLoc, 4, cv::Scalar(0, 255, 0), -1);
    }

    // Show visualisation
    cv::imshow("visualisation_bottom", visualisation_bottom_img);
}



// Convert from Caffe representation to OpenCV img
template <typename Dtype>
void PredictionHeatmapLayer<Dtype>::PrepVis(cv::Mat img, cv::Size size)
{
    cv::transpose(img, img);
    cv::flip(img, img, 1);
}

#ifdef CPU_ONLY
STUB_GPU(PredictionHeatmapLayer);
#endif

INSTANTIATE_CLASS(PredictionHeatmapLayer);
REGISTER_LAYER_CLASS(PredictionHeatmap);


}  // namespace caffe