// Copyright 2015 Tomas Pfister

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <stdint.h>

#include <cmath>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/data_heatmap.hpp"
#include "caffe/util/benchmark.hpp"
#include <unistd.h>


namespace caffe
{

template <typename Dtype>
DataHeatmapLayer<Dtype>::~DataHeatmapLayer<Dtype>() {
    this->StopInternalThread();
}


template<typename Dtype>
void DataHeatmapLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    HeatmapDataParameter heatmap_data_param = this->layer_param_.heatmap_data_param();

    // Shortcuts
    const std::string labelindsStr = heatmap_data_param.labelinds();
    const int batchsize = heatmap_data_param.batchsize();
    const int label_width = heatmap_data_param.label_width();
    const int label_height = heatmap_data_param.label_height();
    const int outsize = heatmap_data_param.outsize();
    const int label_batchsize = batchsize;
    scale_width = heatmap_data_param.scale_width();
    scale_height = heatmap_data_param.scale_height();
    sample_per_cluster_ = heatmap_data_param.sample_per_cluster();
    root_img_dir_ = heatmap_data_param.root_img_dir();


    // initialise rng seed
    const unsigned int rng_seed = caffe_rng_rand();
    srand(rng_seed);

    // get label inds to be used for training
    std::istringstream labelss(labelindsStr);
    LOG(INFO) << "using joint inds:";
    while (labelss)
    {
        std::string s;
        if (!std::getline(labelss, s, ',')) break;
        labelinds_.push_back(atof(s.c_str()));
        LOG(INFO) << atof(s.c_str());
    }

    // load GT
    std::string gt_path = heatmap_data_param.source();
    LOG(INFO) << "Loading annotation from " << gt_path;

    std::ifstream infile(gt_path.c_str());
    string img_name, labels, cropInfos, clusterClassStr;
    if (!sample_per_cluster_)
    {
        // sequential sampling
        while (infile >> img_name >> labels >> cropInfos >> clusterClassStr)
        {
            // read comma-separated list of regression labels
            std::vector <float> label;
            std::istringstream ss(labels);
            int labelCounter = 1;
            while (ss)
            {
                std::string s;
                if (!std::getline(ss, s, ',')) break;
                if (labelinds_.empty() || std::find(labelinds_.begin(), labelinds_.end(), labelCounter) != labelinds_.end())
                {
                    label.push_back(atof(s.c_str()));
                }
                labelCounter++;
            }

            // read cropping info
            std::vector <float> cropInfo;
            std::istringstream ss2(cropInfos);
            while (ss2)
            {
                std::string s;
                if (!std::getline(ss2, s, ',')) break;
                cropInfo.push_back(atof(s.c_str()));
            }

            int clusterClass = atoi(clusterClassStr.c_str());

            img_label_list_.push_back(std::make_pair(img_name, std::make_pair(label, std::make_pair(cropInfo, clusterClass))));
        }

        // initialise image counter to 0
        cur_img_ = 0;
    }
    else
    {
        // uniform sampling w.r.t. classes
        while (infile >> img_name >> labels >> cropInfos >> clusterClassStr)
        {
            int clusterClass = atoi(clusterClassStr.c_str());

            if (clusterClass + 1 > img_list_.size())
            {
                // expand the array
                img_list_.resize(clusterClass + 1);
            }

            // read comma-separated list of regression labels
            std::vector <float> label;
            std::istringstream ss(labels);
            int labelCounter = 1;
            while (ss)
            {
                std::string s;
                if (!std::getline(ss, s, ',')) break;
                if (labelinds_.empty() || std::find(labelinds_.begin(), labelinds_.end(), labelCounter) != labelinds_.end())
                {
                    label.push_back(atof(s.c_str()));
                }
                labelCounter++;
            }

            // read cropping info
            std::vector <float> cropInfo;
            std::istringstream ss2(cropInfos);
            while (ss2)
            {
                std::string s;
                if (!std::getline(ss2, s, ',')) break;
                cropInfo.push_back(atof(s.c_str()));
            }

            img_list_[clusterClass].push_back(std::make_pair(img_name, std::make_pair(label, std::make_pair(cropInfo, clusterClass))));
        }

        const int num_classes = img_list_.size();

        // init image sampling
        cur_class_ = 0;
        cur_class_img_.resize(num_classes);

        // init image indices for each class
        for (int idx_class = 0; idx_class < num_classes; idx_class++)
        {
            if (sample_per_cluster_)
            {
                cur_class_img_[idx_class] = rand() % img_list_[idx_class].size();
                LOG(INFO) << idx_class << " size: " << img_list_[idx_class].size();
            }
            else
            {
                cur_class_img_[idx_class] = 0;
            }
        }
    }


    if (this->layer_param_.heatmap_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(img_label_list_.begin(), img_label_list_.end(), prefetch_rng);
    }
    //  no mean, assume input images are RGB (3 channels)
    this->datum_channels_ = 3;


    // init data
    this->transformed_data_.Reshape(batchsize, this->datum_channels_, outsize, outsize);
    top[0]->Reshape(batchsize, this->datum_channels_, outsize, outsize);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i)
        this->prefetch_[i].data_.Reshape(batchsize, this->datum_channels_, outsize, outsize);
    this->datum_size_ = this->datum_channels_ * outsize * outsize;

    // init label
    int label_num_channels;
    if (!sample_per_cluster_)
        label_num_channels = img_label_list_[0].second.first.size();
    else
        label_num_channels = img_list_[0][0].second.first.size();
    label_num_channels /= 2;
    top[1]->Reshape(label_batchsize, label_num_channels, label_height, label_width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i)
        this->prefetch_[i].label_.Reshape(label_batchsize, label_num_channels, label_height, label_width);

    LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
    LOG(INFO) << "output label size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();
    LOG(INFO) << "number of label channels: " << label_num_channels;
    LOG(INFO) << "datum channels: " << this->datum_channels_;

}









template<typename Dtype>
void DataHeatmapLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

    CPUTimer batch_timer;
    batch_timer.Start();
    CHECK(batch->data_.count());
    HeatmapDataParameter heatmap_data_param = this->layer_param_.heatmap_data_param();

    // Pointers to blobs' float data
    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* top_label = batch->label_.mutable_cpu_data();

    cv::Mat img, img_res, img_annotation_vis, img_vis, img_res_vis, seg, segTmp;

    // Shortcuts to params
    const bool visualise = this->layer_param_.visualise();
    const int batchsize = heatmap_data_param.batchsize();
    const int label_height = heatmap_data_param.label_height();
    const int label_width = heatmap_data_param.label_width();
    const float angle_max = heatmap_data_param.angle_max();
    const int multfact = heatmap_data_param.multfact();
    const bool segmentation = heatmap_data_param.segmentation();
    const int outsize = heatmap_data_param.outsize();
    const int num_aug = 1;
    const bool data_augment = heatmap_data_param.data_augment();
    const string bbox_or_centerscale = heatmap_data_param.bbox_or_centerscale();

    // Shortcuts to global vars
    const int channels = this->datum_channels_;


    if (visualise)
    {
        cv::namedWindow("original image", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("cropped image", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("cropped image with padding", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("resulting image", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("heat map", cv::WINDOW_AUTOSIZE);
    }

    // collect "batchsize" images
    std::vector<float> cur_label, cur_cropinfo;
    std::string img_name;
    int cur_class;

    // loop over non-augmented images
    for (int idx_img = 0; idx_img < batchsize; idx_img++)
    {
        // get image name and class
        this->GetCurImg(img_name, cur_label, cur_cropinfo, cur_class);

        // get number of channels for image label
        int label_num_channels = cur_label.size();

        std::string img_path = this->root_img_dir_ + img_name;
        DLOG(INFO) << "img: " << img_path;
        img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);


        // show image
        if (visualise)
        {
            img_annotation_vis = img.clone();
            this->VisualiseAnnotations(img_annotation_vis, label_num_channels, cur_label, multfact);
            cv::imshow("original image", img_annotation_vis);
        }

        // use if seg exists
        if (segmentation)
        {
            std::string seg_path = this->root_img_dir_ + "segs/" + img_name;
            std::ifstream ifile(seg_path.c_str());

            // Skip this file if segmentation doesn't exist
            if (!ifile.good())
            {
                LOG(INFO) << "file " << seg_path << " does not exist!";
                idx_img--;
                this->AdvanceCurImg();
                continue;
            }
            ifile.close();
            seg = cv::imread(seg_path, CV_LOAD_IMAGE_GRAYSCALE);
        }

        int width = img.cols;
        int height = img.rows;
        float cropImgLength = 0;
        // convert from BGR to RGB
        cv::cvtColor(img, img, CV_BGR2RGB);

        // to float
        img.convertTo(img, CV_32FC3);

        if (segmentation)
        {
            segTmp = cv::Mat::zeros(img.rows, img.cols, CV_32FC3);
            int threshold = 40;
            seg = (seg > threshold);
            segTmp.copyTo(img, seg);
        }

        if (visualise)
            img_vis = img.clone();

        DLOG(INFO) << "Entering jitter loop.";

        // loop over the jittered versions
        for (int idx_aug = 0; idx_aug < num_aug; idx_aug++)
        {
            // augmented image index in the resulting batch
            const int idx_img_aug = idx_img * num_aug + idx_aug;
            std::vector<float> cur_label_aug = cur_label;

            if (data_augment)
            {
                if(bbox_or_centerscale == "centerscale")
                {
                        // random sampling
                        DLOG(INFO) << "Using centerscale, do data augment";
                
                        // left-top coordinates of the crop [0;x_border] x [0;y_border]
                        const float x_center = cur_cropinfo[0], y_center = cur_cropinfo[1];
                        const float scale = cur_cropinfo[2];
                        const float length = 200*scale;//for MPII, they scale person into 200 pixel height.
                        cropImgLength = length;

                        const float x_min = x_center - length/2;
                        const float y_min = y_center - length/2;
                        
                        const float left = std::max(x_min,(float)0);
                        const float top = std::max(y_min,(float)0);
                        
                        const float pad_left = std::max(0-x_min,(float)0);
                        const float pad_top = std::max(0-y_min,(float)0);
                        const float pad_right = std::max(left+length-width-pad_left,(float)0);
                        const float pad_bottom = std::max(top+length-height-pad_top,(float)0);

                        // do crop
                        cv::Rect crop((int)left, (int)top, (int)std::min(length - pad_left,width - left), (int)std::min(length - pad_top,height - top));
                        // NOTE: no full copy performed, so the original image buffer is affected by the transformations below
                        cv::Mat img_crop(img, crop);
                        // show image
                        if (visualise)
                        {
                            DLOG(INFO) << "cropped image";
                            cv::Mat img_vis_crop(img_vis, crop);
                            cv::Mat img_res_vis = img_vis_crop / 255;
                            cv::cvtColor(img_res_vis, img_res_vis, CV_RGB2BGR);
                            cv::imshow("cropped image", img_res_vis);
                        }

                        //do pad
                        cv::copyMakeBorder(img_crop, img_res, (int)pad_top, (int)pad_bottom, (int)pad_left, (int)pad_right, cv::BORDER_CONSTANT, 0 );

                        // "crop" annotations
                        for (int i = 0; i < label_num_channels; i += 2)
                        {
                            cur_label_aug[i] -= x_min / (float) multfact;
                            cur_label_aug[i + 1] -= y_min / (float) multfact;
                        }
                
                        // rotations
                        float angle = Uniform(-angle_max, angle_max);
                        cv::Mat M = this->RotateImage(img_res, angle);
                
                        // also rotate labels
                        for (int i = 0; i < label_num_channels; i += 2)
                        {
                            // convert to image space
                            float x = cur_label_aug[i] * (float) multfact;
                            float y = cur_label_aug[i + 1] * (float) multfact;
                
                            // rotate
                            cur_label_aug[i] = M.at<double>(0, 0) * x + M.at<double>(0, 1) * y + M.at<double>(0, 2);
                            cur_label_aug[i + 1] = M.at<double>(1, 0) * x + M.at<double>(1, 1) * y + M.at<double>(1, 2);
                
                            // convert back to joint space
                            cur_label_aug[i] /= (float) multfact;
                            cur_label_aug[i + 1] /= (float) multfact;
                        }
                
                }
                else if(bbox_or_centerscale == "bbox")
                {
                        DLOG(INFO) << "Using bbox, do data augment";
                
                        float x_min = cur_cropinfo[0], y_min = cur_cropinfo[1], x_max = cur_cropinfo[2], y_max = cur_cropinfo[3];
                        float W = x_max-x_min, H = y_max-y_min;
                        x_min = std::max((float)0,x_min-W*(scale_width-1)/2);
                        y_min = std::max((float)0,y_min-H*(scale_height-1)/2);
                        x_max = std::min((float)width,x_max+W*(scale_width-1)/2);
                        y_max = std::min((float)height,y_max+H*(scale_height-1)/2);
                        W = x_max-x_min; H = y_max-y_min;                        
                        const float length = std::max(W,H);
                        cropImgLength = length;

                        const float pad_left = std::max((float)0,length-W)/2;
                        const float pad_top = std::max((float)0,length-H)/2;
                        const float pad_right = std::max((float)0,length-W)/2;
                        const float pad_bottom = std::max((float)0,length-H)/2;

                        // do crop
                        cv::Rect crop((int)x_min, (int)y_min, (int)(x_max-x_min), (int)(y_max-y_min));
                        // NOTE: no full copy performed, so the original image buffer is affected by the transformations below
                        cv::Mat img_crop(img, crop);
                        cv::Mat imgCrop = img_crop.clone();
                        // show image
                        if (visualise)
                        {
                            DLOG(INFO) << "cropped image";
                            cv::Mat img_vis_crop(img_vis, crop);
                            cv::Mat img_res_vis = img_vis_crop / 255;
                            cv::cvtColor(img_res_vis, img_res_vis, CV_RGB2BGR);
                            cv::imshow("cropped image", img_res_vis);
                        }

                        //do pad
                        cv::copyMakeBorder(imgCrop, img_res, (int)pad_top, (int)pad_bottom, (int)pad_left, (int)pad_right, cv::BORDER_CONSTANT, 0 );

                        // "crop" annotations
                        for (int i = 0; i < label_num_channels; i += 2)
                        {
                            cur_label_aug[i] -= (x_min - pad_left)/ (float) multfact;
                            cur_label_aug[i + 1] -= (y_min - pad_top) / (float) multfact;
                        }
                
                        // rotations
                        float angle = Uniform(-angle_max, angle_max);
                        cv::Mat M = this->RotateImage(img_res, angle);
                
                        // also rotate labels
                        for (int i = 0; i < label_num_channels; i += 2)
                        {
                            // convert to image space
                            float x = cur_label_aug[i] * (float) multfact;
                            float y = cur_label_aug[i + 1] * (float) multfact;
                
                            // rotate
                            cur_label_aug[i] = M.at<double>(0, 0) * x + M.at<double>(0, 1) * y + M.at<double>(0, 2);
                            cur_label_aug[i + 1] = M.at<double>(1, 0) * x + M.at<double>(1, 1) * y + M.at<double>(1, 2);
                
                            // convert back to joint space
                            cur_label_aug[i] /= (float) multfact;
                            cur_label_aug[i + 1] /= (float) multfact;
                        }
                }
                else
                {
                    CHECK(false) << "crop type only support \"bbox\" or \"centerscale\"" << std::endl;
                }
            } 
            else 
            {
                if(bbox_or_centerscale == "centerscale")
                {
                        DLOG(INFO) << "Using centerscale, no data augment";
                
                        // left-top coordinates of the crop [0;x_border] x [0;y_border]
                        const float x_center = cur_cropinfo[0], y_center = cur_cropinfo[1];
                        const float scale = cur_cropinfo[2];
                        const float length = 200*scale;//for MPII, they scale person into 200 pixel height.
                        cropImgLength = length;

                        const float x_min = x_center - length/2;
                        const float y_min = y_center - length/2;
                        
                        const float left = std::max(x_min,(float)0);
                        const float top = std::max(y_min,(float)0);

                        const float pad_left = std::max(0-x_min,(float)0);
                        const float pad_top = std::max(0-y_min,(float)0);
                        const float pad_right = std::max(left+length-width-pad_left,(float)0);
                        const float pad_bottom = std::max(top+length-height-pad_top,(float)0);

                        // do crop
                        cv::Rect crop((int)left, (int)top, (int)std::min(length - pad_left,width - left), (int)std::min(length - pad_top,height - top));
                        // NOTE: no full copy performed, so the original image buffer is affected by the transformations below
                        cv::Mat img_crop(img, crop);
                        // show image
                        if (visualise)
                        {
                            DLOG(INFO) << "cropped image";
                            cv::Mat img_vis_crop(img_vis, crop);
                            cv::Mat img_res_vis = img_vis_crop / 255;
                            cv::cvtColor(img_res_vis, img_res_vis, CV_RGB2BGR);
                            cv::imshow("cropped image", img_res_vis);
                        }
                        //do pad
                        cv::copyMakeBorder(img_crop, img_res, (int)pad_top, (int)pad_bottom, (int)pad_left, (int)pad_right, cv::BORDER_CONSTANT, 0 );

                        // "crop" annotations
                        for (int i = 0; i < label_num_channels; i += 2)
                        {
                            cur_label_aug[i] -= x_min / (float) multfact;
                            cur_label_aug[i + 1] -= y_min / (float) multfact;
                        }
                }
                else if(bbox_or_centerscale == "bbox")
                {
                        DLOG(INFO) << "Using bbox, no data augment";
                
                        float x_min = cur_cropinfo[0], y_min = cur_cropinfo[1], x_max = cur_cropinfo[2], y_max = cur_cropinfo[3];
                        float W = x_max-x_min, H = y_max-y_min;
                        x_min = std::max((float)0,x_min-W*(scale_width-1)/2);
                        y_min = std::max((float)0,y_min-H*(scale_height-1)/2);
                        x_max = std::min((float)width,x_max+W*(scale_width-1)/2);
                        y_max = std::min((float)height,y_max+H*(scale_height-1)/2);
                        W = x_max-x_min; H = y_max-y_min;
                        const float length = std::max(W,H);
                        cropImgLength = length;

                        const float pad_left = std::max((float)0,length-W)/2;
                        const float pad_top = std::max((float)0,length-H)/2;
                        const float pad_right = std::max((float)0,length-W)/2;
                        const float pad_bottom = std::max((float)0,length-H)/2;

                        // do crop
                        cv::Rect crop((int)x_min, (int)y_min, (int)(x_max-x_min), (int)(y_max-y_min));
                        // NOTE: no full copy performed, so the original image buffer is affected by the transformations below
                        cv::Mat img_crop(img, crop);
                        cv::Mat imgCrop = img_crop.clone();
                        // show image
                        if (visualise)
                        {
                            DLOG(INFO) << "cropped image";
                            cv::Mat img_vis_crop(img_vis, crop);
                            cv::Mat img_res_vis = img_vis_crop / 255;
                            cv::cvtColor(img_res_vis, img_res_vis, CV_RGB2BGR);
                            cv::imshow("cropped image", img_res_vis);
                        }

                        //do pad
                        cv::copyMakeBorder(imgCrop, img_res, (int)pad_top, (int)pad_bottom, (int)pad_left, (int)pad_right, cv::BORDER_CONSTANT, 0 );

                        // "crop" annotations
                        for (int i = 0; i < label_num_channels; i += 2)
                        {
                            cur_label_aug[i] -= (x_min - pad_left)/ (float) multfact;
                            cur_label_aug[i + 1] -= (y_min - pad_top) / (float) multfact;
                        }
                }
                else
                {
                    CHECK(false) << "crop type only support \"bbox\" or \"centerscale\"" << std::endl;
                }
                // determinsitic sampling
                DLOG(INFO) << "deterministic crop sampling (centre)";
            }

            // show image
            if (visualise)
            {
                cv::Mat img_res_vis = img_res / 255;
                cv::cvtColor(img_res_vis, img_res_vis, CV_RGB2BGR);
                this->VisualiseAnnotations(img_res_vis, label_num_channels, cur_label_aug, multfact);
                cv::imshow("cropped image with padding", img_res_vis);
            }

            DLOG(INFO) << "Resizing output image.";

            // resize to output image size
            cv::Size s(outsize, outsize);
            cv::resize(img_res, img_res, s);

            // "resize" annotations
            const float resizeFact = outsize/cropImgLength;
            for (int i = 0; i < label_num_channels; i++)
                cur_label_aug[i] *= resizeFact;

            // show image
            if (visualise)
            {
                cv::Mat img_res_vis = img_res / 255;
                cv::cvtColor(img_res_vis, img_res_vis, CV_RGB2BGR);
                this->VisualiseAnnotations(img_res_vis, label_num_channels, cur_label_aug, multfact);
                cv::imshow("resulting image", img_res_vis);
            }

            // resulting image dims
            const int channel_size = outsize * outsize;
            const int img_size = channel_size * channels;

            // store image data
            DLOG(INFO) << "storing image";
            for (int c = 0; c < channels; c++)
            {
                for (int i = 0; i < outsize; i++)
                {
                    for (int j = 0; j < outsize; j++)
                    {
                        top_data[idx_img_aug * img_size + c * channel_size + i * outsize + j] = img_res.at<cv::Vec3f>(i, j)[c]/255.0;
                    }
                }
            }

            // store label as gaussian
            DLOG(INFO) << "storing labels";
            const int label_channel_size = label_height * label_width;
            const int label_img_size = label_channel_size * label_num_channels / 2;
            cv::Mat dataMatrix = cv::Mat::zeros(label_height, label_width, CV_32FC1);
            float label_resize_fact = (float) label_height / (float) outsize;
            float sigma = 1;

            for (int idx_ch = 0; idx_ch < label_num_channels / 2; idx_ch++)
            {
                float x = label_resize_fact * cur_label_aug[2 * idx_ch] * multfact;
                float y = label_resize_fact * cur_label_aug[2 * idx_ch + 1] * multfact;
                if((int)cur_label[2 * idx_ch] <= 0 and (int)cur_label[2 * idx_ch + 1] <= 0)
                {
                    for (int i = 0; i < label_height; i++)
                    {
                        for (int j = 0; j < label_width; j++)
                        {
                            int label_idx = idx_img_aug * label_img_size + idx_ch * label_channel_size + i * label_width + j;
                            top_label[label_idx] = 0;

                            dataMatrix.at<float>((int)i, (int)j) += 0;
                        }
                    }
                }
                else
                {    
                    for (int i = 0; i < label_height; i++)
                    {
                        for (int j = 0; j < label_width; j++)
                        {
                            int label_idx = idx_img_aug * label_img_size + idx_ch * label_channel_size + i * label_width + j;
                            float gaussian = exp( -0.5 * ( pow(i - y, 2.0) + pow(j - x, 2.0) ) * pow(1 / sigma, 2.0) );
                            top_label[label_idx] = gaussian;

                            dataMatrix.at<float>((int)i, (int)j) += gaussian;
                        }
                    }
                }
            }
            if (visualise)
            {
                cv::Size s(outsize, outsize);
                cv::resize(dataMatrix, dataMatrix, s);
                cv::imshow("heat map", dataMatrix);
            }

        } // jittered versions loop

        DLOG(INFO) << "next image";

        // move to the next image
        this->AdvanceCurImg();

        if (visualise)
            cv::waitKey(0);


    } // original image loop

    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}



template<typename Dtype>
void DataHeatmapLayer<Dtype>::GetCurImg(string& img_name, std::vector<float>& img_label, std::vector<float>& crop_info, int& img_class)
{

    if (!sample_per_cluster_)
    {
        img_name = img_label_list_[cur_img_].first;
        img_label = img_label_list_[cur_img_].second.first;
        crop_info = img_label_list_[cur_img_].second.second.first;
        img_class = img_label_list_[cur_img_].second.second.second;
    }
    else
    {
        img_class = cur_class_;
        img_name = img_list_[img_class][cur_class_img_[img_class]].first;
        img_label = img_list_[img_class][cur_class_img_[img_class]].second.first;
        crop_info = img_list_[img_class][cur_class_img_[img_class]].second.second.first;
    }
}

template<typename Dtype>
void DataHeatmapLayer<Dtype>::AdvanceCurImg()
{
    if (!sample_per_cluster_)
    {
        if (cur_img_ < img_label_list_.size() - 1)
            cur_img_++;
        else{
            cur_img_ = 0;
	    if (this->layer_param_.heatmap_data_param().shuffle()) {
           	    LOG(INFO) << "Shuffling data";
    		    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    		    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    		    caffe::rng_t* prefetch_rng =
      			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    		    shuffle(img_label_list_.begin(), img_label_list_.end(), prefetch_rng);
     	    }
	}
    }
    else
    {
        const int num_classes = img_list_.size();

        if (cur_class_img_[cur_class_] < img_list_[cur_class_].size() - 1)
            cur_class_img_[cur_class_]++;
        else
            cur_class_img_[cur_class_] = 0;

        // move to the next class
        if (cur_class_ < num_classes - 1)
            cur_class_++;
        else
            cur_class_ = 0;
    }

}


template<typename Dtype>
void DataHeatmapLayer<Dtype>::VisualiseAnnotations(cv::Mat img_annotation_vis, int label_num_channels, std::vector<float>& img_class, int multfact)
{
    // colors
    const static cv::Scalar colors[] = {
        CV_RGB(0, 0, 255),
        CV_RGB(0, 128, 255),
        CV_RGB(0, 255, 255),
        CV_RGB(0, 255, 0),
        CV_RGB(255, 128, 0),
        CV_RGB(255, 255, 0),
        CV_RGB(255, 0, 0),
        CV_RGB(128, 0, 255),
        CV_RGB(0, 0, 128),
        CV_RGB(0, 128, 128),
        CV_RGB(0, 128, 128),
        CV_RGB(0, 128, 0),
        CV_RGB(128, 128, 0),
        CV_RGB(128, 255, 0),
        CV_RGB(128, 0, 0),
        CV_RGB(128, 0, 255)
    };

    int numCoordinates = int(label_num_channels / 2);

    // points
    cv::Point centers[numCoordinates];
    for (int i = 0; i < label_num_channels; i += 2)
    {
        int coordInd = int(i / 2);
        centers[coordInd] = cv::Point(img_class[i] * multfact, img_class[i + 1] * multfact);
        cv::circle(img_annotation_vis, centers[coordInd], 1, colors[coordInd], 3);
    }

    // connecting lines
    cv::line(img_annotation_vis, centers[0], centers[1], CV_RGB(0, 255, 0), 1, CV_AA);
    cv::line(img_annotation_vis, centers[1], centers[2], CV_RGB(255, 255, 0), 1, CV_AA);
    cv::line(img_annotation_vis, centers[2], centers[6], CV_RGB(0, 0, 255), 1, CV_AA);
    cv::line(img_annotation_vis, centers[3], centers[4], CV_RGB(0, 255, 255), 1, CV_AA);
    cv::line(img_annotation_vis, centers[3], centers[6], CV_RGB(0, 122, 255), 1, CV_AA);
    cv::line(img_annotation_vis, centers[4], centers[5], CV_RGB(0, 255, 122), 1, CV_AA);
    cv::line(img_annotation_vis, centers[6], centers[8], CV_RGB(122, 0, 255), 1, CV_AA);
    cv::line(img_annotation_vis, centers[8], centers[9], CV_RGB(0, 122, 0), 1, CV_AA);
    cv::line(img_annotation_vis, centers[13], centers[8], CV_RGB(122, 0, 122), 1, CV_AA);
    cv::line(img_annotation_vis, centers[10], centers[11], CV_RGB(255, 0, 122), 1, CV_AA);
    cv::line(img_annotation_vis, centers[11], centers[12], CV_RGB(255, 255, 122), 1, CV_AA);
    cv::line(img_annotation_vis, centers[12], centers[8], CV_RGB(122, 122, 0), 1, CV_AA);
    cv::line(img_annotation_vis, centers[13], centers[14], CV_RGB(0, 0, 122), 1, CV_AA);
    cv::line(img_annotation_vis, centers[14], centers[15], CV_RGB(0, 122, 122), 1, CV_AA);

}


template <typename Dtype>
float DataHeatmapLayer<Dtype>::Uniform(const float min, const float max) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = max - min;
    float r = random * diff;
    return min + r;
}

template <typename Dtype>
cv::Mat DataHeatmapLayer<Dtype>::RotateImage(cv::Mat src, float rotation_angle)
{
    cv::Mat rot_mat(2, 3, CV_32FC1);
    cv::Point center = cv::Point(src.cols / 2, src.rows / 2);
    double scale = 1;

    // Get the rotation matrix with the specifications above
    rot_mat = cv::getRotationMatrix2D(center, rotation_angle, scale);

    // Rotate the warped image
    cv::warpAffine(src, src, rot_mat, src.size());

    return rot_mat;
}

INSTANTIATE_CLASS(DataHeatmapLayer);
REGISTER_LAYER_CLASS(DataHeatmap);

} // namespace caffe
