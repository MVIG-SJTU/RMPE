#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/euclidean_loss_heatmap_layer.hpp"


// Euclidean loss layer that computes loss on a [x] x [y] x [ch] set of heatmaps,
// and enables visualisation of inputs, GT, prediction and loss.


namespace caffe {

template <typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[1]->height());
    CHECK_EQ(bottom[0]->width(), bottom[1]->width());
    diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                  bottom[0]->height(), bottom[0]->width());
}


template<typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    if (this->layer_param_.loss_weight_size() == 0) {
        this->layer_param_.add_loss_weight(Dtype(1));
    }

}

template <typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
{
    Dtype loss = 0;

    int visualise_channel = this->layer_param_.visualise_channel();
    bool visualise = this->layer_param_.visualise();

    const Dtype* bottom_pred = bottom[0]->cpu_data(); // predictions for all images
    const Dtype* gt_pred = bottom[1]->cpu_data();    // GT predictions
    const int num_images = bottom[1]->num();
    const int label_height = bottom[1]->height();
    const int label_width = bottom[1]->width();
    const int num_channels = bottom[0]->channels();

    DLOG(INFO) << "bottom size: " << bottom[0]->height() << " " << bottom[0]->width() << " " << bottom[0]->channels();

    const int label_channel_size = label_height * label_width;
    const int label_img_size = label_channel_size * num_channels;
    cv::Mat bottom_img, gt_img, diff_img;  // Initialise opencv images for visualisation

    if (visualise)
    {
        cv::namedWindow("bottom", CV_WINDOW_AUTOSIZE);
        cv::namedWindow("gt", CV_WINDOW_AUTOSIZE);
        cv::namedWindow("diff", CV_WINDOW_AUTOSIZE);
        cv::namedWindow("overlay", CV_WINDOW_AUTOSIZE);
        cv::namedWindow("visualisation_bottom", CV_WINDOW_AUTOSIZE);
        bottom_img = cv::Mat::zeros(label_height, label_width, CV_32FC1);
        gt_img = cv::Mat::zeros(label_height, label_width, CV_32FC1);
        diff_img = cv::Mat::zeros(label_height, label_width, CV_32FC1);
    }

    // Loop over images
    for (int idx_img = 0; idx_img < num_images; idx_img++)
    {
        // Compute loss
        for (int idx_ch = 0; idx_ch < num_channels; idx_ch++)
        {
            for (int i = 0; i < label_height; i++)
            {
                for (int j = 0; j < label_width; j++)
                {
                    int image_idx = idx_img * label_img_size + idx_ch * label_channel_size + i * label_height + j;
                    float diff = (float)bottom_pred[image_idx] - (float)gt_pred[image_idx];
                    loss += diff * diff;

                    // Store visualisation for given channel
                    if (idx_ch == visualise_channel && visualise)
                    {
                        bottom_img.at<float>((int)j, (int)i) = (float) bottom_pred[image_idx];
                        gt_img.at<float>((int)j, (int)i) = (float) gt_pred[image_idx];
                        diff_img.at<float>((int)j, (int)i) = (float) diff * diff;
                    }

                }
            }
        }
        // Plot visualisation
        if (visualise)
        {
//            DLOG(INFO) << "num_images=" << num_images << " idx_img=" << idx_img;
//            DLOG(INFO) << "sum bottom: " << cv::sum(bottom_img) << "  sum gt: " << cv::sum(gt_img);
            int visualisation_size = 256;
            cv::Size size(visualisation_size, visualisation_size);            
            std::vector<cv::Point> points;
            this->Visualise(loss, bottom_img, gt_img, diff_img, points, size);
            this->VisualiseBottom(bottom, idx_img, visualise_channel, points, size);
            cv::waitKey(0);     // Wait forever a key is pressed
        }
    }

    DLOG(INFO) << "total loss: " << loss;
    loss /= (num_images * num_channels * label_channel_size);
    DLOG(INFO) << "total normalised loss: " << loss;

    top[0]->mutable_cpu_data()[0] = loss;
}



// Visualise GT heatmap, predicted heatmap, input image and max in heatmap
// bottom: predicted heatmaps
// gt: ground truth gaussian heatmaps
// diff: per-pixel loss
// overlay: prediction with GT location & max of prediction
// visualisation_bottom: additional visualisation layer (defined as the last 'bottom' in the loss prototxt def)
template <typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::Visualise(float loss, cv::Mat bottom_img, cv::Mat gt_img, cv::Mat diff_img, std::vector<cv::Point>& points, cv::Size size)
{
    DLOG(INFO) << loss;

    // Definitions
    double minVal, maxVal;
    cv::Point minLocGT, maxLocGT;
    cv::Point minLocBottom, maxLocBottom;
    cv::Point minLocThird, maxLocThird;
    cv::Mat overlay_img_orig, overlay_img;

    // Convert prediction (bottom) into 3 channels, call 'overlay'
    overlay_img_orig = bottom_img.clone() - 1;
    cv::Mat in[] = {overlay_img_orig, overlay_img_orig, overlay_img_orig};
    cv::merge(in, 3, overlay_img);

    // Resize all images to fixed size
    PrepVis(bottom_img, size);
    cv::resize(bottom_img, bottom_img, size);
    PrepVis(gt_img, size);
    cv::resize(gt_img, gt_img, size);
    PrepVis(diff_img, size);
    cv::resize(diff_img, diff_img, size);
    PrepVis(overlay_img, size);
    cv::resize(overlay_img, overlay_img, size);

    // Get and plot GT position & prediction position in new visualisation-resized space
    cv::minMaxLoc(gt_img, &minVal, &maxVal, &minLocGT, &maxLocGT);
    DLOG(INFO) << "gt min: " << minVal << "  max: " << maxVal;
    cv::minMaxLoc(bottom_img, &minVal, &maxVal, &minLocBottom, &maxLocBottom);
    DLOG(INFO) << "bottom min: " << minVal << "  max: " << maxVal;
    cv::circle(overlay_img, maxLocGT, 5, cv::Scalar(0, 255, 0), -1);
    cv::circle(overlay_img, maxLocBottom, 3, cv::Scalar(0, 0, 255), -1);

    // Show visualisation images
    cv::imshow("bottom", bottom_img - 1);
    cv::imshow("gt", gt_img - 1);
    cv::imshow("diff", diff_img);
    cv::imshow("overlay", overlay_img - 1);

    // Store max locations
    points.push_back(maxLocGT);
    points.push_back(maxLocBottom);
}

// Plot another visualisation image overlaid with ground truth & prediction locations
// (particularly useful e.g. if you set this to the original input image)
template <typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::VisualiseBottom(const vector<Blob<Dtype>*>& bottom, int idx_img, int visualise_channel, std::vector<cv::Point>& points, cv::Size size)
{
    // Determine which layer to visualise
    Blob<Dtype>* visualisation_bottom = bottom[2];
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
                    visualisation_bottom_img.at<cv::Vec3f>((int)j, (int)i)[idx_ch] = 4 * (float) visualisation_bottom->cpu_data()[image_idx] / 255;
                } else if (idx_ch == visualise_channel)
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

    // Plot max of GT & prediction
    cv::Point maxLocGT = points[0];
    cv::Point maxLocBottom = points[1];    
    cv::circle(visualisation_bottom_img, maxLocGT, 5, cv::Scalar(0, 255, 0), -1);
    cv::circle(visualisation_bottom_img, maxLocBottom, 3, cv::Scalar(0, 0, 255), -1);

    // Show visualisation
    cv::imshow("visualisation_bottom", visualisation_bottom_img - 1);
}



// Convert from Caffe representation to OpenCV img
template <typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::PrepVis(cv::Mat img, cv::Size size)
{
    cv::transpose(img, img);
    cv::flip(img, img, 1);
}


template <typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    const int count = bottom[0]->count();
    const int channels = bottom[0]->channels();

    caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), diff_.mutable_cpu_data());

    // strictly speaking, should be normalising by (2 * channels) due to 1/2 multiplier in front of the loss
    Dtype loss = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data()) / Dtype(channels);

    // copy the gradient
    memcpy(bottom[0]->mutable_cpu_diff(), diff_.cpu_data(), sizeof(Dtype) * count);
    memcpy(bottom[1]->mutable_cpu_diff(), diff_.cpu_data(), sizeof(Dtype) * count);

}


template <typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
{
    Forward_cpu(bottom, top);
}

template <typename Dtype>
void EuclideanLossHeatmapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    Backward_cpu(top, propagate_down, bottom);
}



#ifdef CPU_ONLY
STUB_GPU(EuclideanLossHeatmapLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossHeatmapLayer);
REGISTER_LAYER_CLASS(EuclideanLossHeatmap);


}  // namespace caffe
