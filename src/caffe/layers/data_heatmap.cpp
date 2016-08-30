// Copyright 2015 Tomas Pfister

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <stdint.h>

#include <cmath>

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
    const int size = heatmap_data_param.cropsize();
    const int outsize = heatmap_data_param.outsize();
    const int label_batchsize = batchsize;
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

    if (!heatmap_data_param.has_meanfile())
    {
        // if no mean, assume input images are RGB (3 channels)
        this->datum_channels_ = 3;
        sub_mean_ = false;
    } else {
        // Implementation of per-video mean removal

        sub_mean_ = true;
        string mean_path = heatmap_data_param.meanfile();

        LOG(INFO) << "Loading mean file from " << mean_path;
        BlobProto blob_proto, blob_proto2;
        Blob<Dtype> data_mean;
        ReadProtoFromBinaryFile(mean_path.c_str(), &blob_proto);
        data_mean.FromProto(blob_proto);
        LOG(INFO) << "mean file loaded";

        // read config
        this->datum_channels_ = data_mean.channels();
        num_means_ = data_mean.num();
        LOG(INFO) << "num_means: " << num_means_;

        // copy the per-video mean images to an array of OpenCV structures
        const Dtype* mean_buf = data_mean.cpu_data();

        // extract means from beginning of proto file
        const int mean_height = data_mean.height();
        const int mean_width = data_mean.width();
        int mean_heights[num_means_];
        int mean_widths[num_means_];

        // offset in memory to mean images
        const int meanOffset = 2 * (num_means_);
        for (int n = 0; n < num_means_; n++)
        {
            mean_heights[n] = mean_buf[2 * n];
            mean_widths[n] = mean_buf[2 * n + 1];
        }

        // save means as OpenCV-compatible files
        for (int n = 0; n < num_means_; n++)
        {
            cv::Mat mean_img_tmp_;
            mean_img_tmp_.create(mean_heights[n], mean_widths[n], CV_32FC3);
            mean_img_.push_back(mean_img_tmp_);
            LOG(INFO) << "per-video mean file array created: " << n << ": " << mean_heights[n] << "x" << mean_widths[n] << " (" << size << ")";
        }

        LOG(INFO) << "mean: " << mean_height << "x" << mean_width << " (" << size << ")";

        for (int n = 0; n < num_means_; n++)
        {
            for (int i = 0; i < mean_heights[n]; i++)
            {
                for (int j = 0; j < mean_widths[n]; j++)
                {
                    for (int c = 0; c < this->datum_channels_; c++)
                    {
                        mean_img_[n].at<cv::Vec3f>(i, j)[c] = mean_buf[meanOffset + ((n * this->datum_channels_ + c) * mean_height + i) * mean_width + j]; //[c * mean_height * mean_width + i * mean_width + j];
                    }
                }
            }
        }

        LOG(INFO) << "mean file converted to OpenCV structures";
    }


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

    cv::Mat img, img_res, img_annotation_vis, img_mean_vis, img_vis, img_res_vis, mean_img_this, seg, segTmp;

    // Shortcuts to params
    const bool visualise = this->layer_param_.visualise();
    const Dtype scale = heatmap_data_param.scale();
    const int batchsize = heatmap_data_param.batchsize();
    const int label_height = heatmap_data_param.label_height();
    const int label_width = heatmap_data_param.label_width();
    const float angle_max = heatmap_data_param.angle_max();
    const bool dont_flip_first = heatmap_data_param.dont_flip_first();
    const bool flip_joint_labels = heatmap_data_param.flip_joint_labels();
    const int multfact = heatmap_data_param.multfact();
    const bool segmentation = heatmap_data_param.segmentation();
    const int size = heatmap_data_param.cropsize();
    const int outsize = heatmap_data_param.outsize();
    const int num_aug = 1;
    const float resizeFact = (float)outsize / (float)size;
    const bool random_crop = heatmap_data_param.random_crop();

    // Shortcuts to global vars
    const bool sub_mean = this->sub_mean_;
    const int channels = this->datum_channels_;

    // What coordinates should we flip when mirroring images?
    // For pose estimation with joints assumes i=0,1 are for head, and i=2,3 left wrist, i=4,5 right wrist etc
    //     in which case dont_flip_first should be set to true.
    int flip_start_ind;
    if (dont_flip_first) flip_start_ind = 2;
    else flip_start_ind = 0;

    if (visualise)
    {
        cv::namedWindow("original image", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("cropped image", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("interim resize image", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("resulting image", cv::WINDOW_AUTOSIZE);
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
        int x_border = width - size;
        int y_border = height - size;

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

        // subtract per-video mean if used
        int meanInd = 0;
        if (sub_mean)
        {
            std::string delimiter = "/";
            std::string img_name_subdirImg = img_name.substr(img_name.find(delimiter) + 1, img_name.length());
            std::string meanIndStr = img_name_subdirImg.substr(0, img_name_subdirImg.find(delimiter));
            meanInd = atoi(meanIndStr.c_str()) - 1;

            // subtract the cropped mean
            mean_img_this = this->mean_img_[meanInd].clone();

            DLOG(INFO) << "Image size: " << width << "x" << height;
            DLOG(INFO) << "Crop info: " << cur_cropinfo[0] << " " <<  cur_cropinfo[1] << " " << cur_cropinfo[2] << " " << cur_cropinfo[3] << " " << cur_cropinfo[4];
            DLOG(INFO) << "Crop info after: " << cur_cropinfo[0] << " " <<  cur_cropinfo[1] << " " << cur_cropinfo[2] << " " << cur_cropinfo[3] << " " << cur_cropinfo[4];
            DLOG(INFO) << "Mean image size: " << mean_img_this.cols << "x" << mean_img_this.rows;
            DLOG(INFO) << "Cropping: " << cur_cropinfo[0] - 1 << " " << cur_cropinfo[2] - 1 << " " << width << " " << height;

            // crop and resize mean image
            cv::Rect crop(cur_cropinfo[0] - 1, cur_cropinfo[2] - 1, cur_cropinfo[1] - cur_cropinfo[0], cur_cropinfo[3] - cur_cropinfo[2]);
            mean_img_this = mean_img_this(crop);
            cv::resize(mean_img_this, mean_img_this, img.size());

            DLOG(INFO) << "Cropped mean image.";

            img -= mean_img_this;

            DLOG(INFO) << "Subtracted mean image.";

            if (visualise)
            {
                img_vis -= mean_img_this;
                img_mean_vis = mean_img_this.clone() / 255;
                cv::cvtColor(img_mean_vis, img_mean_vis, CV_RGB2BGR);
                cv::imshow("mean image", img_mean_vis);
            }
        }

        // pad images that aren't wide enough
        if (x_border < 0)
        {
            DLOG(INFO) << "padding " << img_path << " -- not wide enough.";

            cv::copyMakeBorder(img, img, 0, 0, 0, -x_border, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            width = img.cols;
            x_border = width - size;

            // add border offset to joints
            for (int i = 0; i < label_num_channels; i += 2)
                cur_label[i] = cur_label[i] + x_border;

            DLOG(INFO) << "new width: " << width << "   x_border: " << x_border;
            if (visualise)
            {
                img_vis = img.clone();
                cv::copyMakeBorder(img_vis, img_vis, 0, 0, 0, -x_border, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            }
        }

        DLOG(INFO) << "Entering jitter loop.";

        // loop over the jittered versions
        for (int idx_aug = 0; idx_aug < num_aug; idx_aug++)
        {
            // augmented image index in the resulting batch
            const int idx_img_aug = idx_img * num_aug + idx_aug;
            std::vector<float> cur_label_aug = cur_label;

            if (random_crop)
            {
                // random sampling
                DLOG(INFO) << "random crop sampling";

                // horizontal flip
                if (rand() % 2)
                {
                    // flip
                    cv::flip(img, img, 1);

                    if (visualise)
                        cv::flip(img_vis, img_vis, 1);

                    // "flip" annotation coordinates
                    for (int i = 0; i < label_num_channels; i += 2)
                        cur_label_aug[i] = (float)width / (float)multfact - cur_label_aug[i];

                    // "flip" annotation joint numbers
                    // assumes i=0,1 are for head, and i=2,3 left wrist, i=4,5 right wrist etc
                    // where coordinates are (x,y)
                    if (flip_joint_labels)
                    {
                        float tmp_x, tmp_y;
                        for (int i = flip_start_ind; i < label_num_channels; i += 4)
                        {
                            CHECK_LT(i + 3, label_num_channels);
                            tmp_x = cur_label_aug[i];
                            tmp_y = cur_label_aug[i + 1];
                            cur_label_aug[i] = cur_label_aug[i + 2];
                            cur_label_aug[i + 1] = cur_label_aug[i + 3];
                            cur_label_aug[i + 2] = tmp_x;
                            cur_label_aug[i + 3] = tmp_y;
                        }
                    }
                }

                // left-top coordinates of the crop [0;x_border] x [0;y_border]
                int x0 = 0, y0 = 0;
                x0 = rand() % (x_border + 1);
                y0 = rand() % (y_border + 1);

                // do crop
                cv::Rect crop(x0, y0, size, size);

                // NOTE: no full copy performed, so the original image buffer is affected by the transformations below
                cv::Mat img_crop(img, crop);

                // "crop" annotations
                for (int i = 0; i < label_num_channels; i += 2)
                {
                    cur_label_aug[i] -= (float)x0 / (float) multfact;
                    cur_label_aug[i + 1] -= (float)y0 / (float) multfact;
                }

                // show image
                if (visualise)
                {
                    DLOG(INFO) << "cropped image";
                    cv::Mat img_vis_crop(img_vis, crop);
                    cv::Mat img_res_vis = img_vis_crop / 255;
                    cv::cvtColor(img_res_vis, img_res_vis, CV_RGB2BGR);
                    this->VisualiseAnnotations(img_res_vis, label_num_channels, cur_label_aug, multfact);
                    cv::imshow("cropped image", img_res_vis);
                }

                // rotations
                float angle = Uniform(-angle_max, angle_max);
                cv::Mat M = this->RotateImage(img_crop, angle);

                // also flip & rotate labels
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

                img_res = img_crop;
            } else {
                // determinsitic sampling
                DLOG(INFO) << "deterministic crop sampling (centre)";

                // centre crop
                const int y0 = y_border / 2;
                const int x0 = x_border / 2;

                DLOG(INFO) << "cropping image from " << x0 << "x" << y0;

                // do crop
                cv::Rect crop(x0, y0, size, size);
                cv::Mat img_crop(img, crop);

                DLOG(INFO) << "cropping annotations.";

                // "crop" annotations
                for (int i = 0; i < label_num_channels; i += 2)
                {
                    cur_label_aug[i] -= (float)x0 / (float) multfact;
                    cur_label_aug[i + 1] -= (float)y0 / (float) multfact;
                }

                if (visualise)
                {
                    cv::Mat img_vis_crop(img_vis, crop);
                    cv::Mat img_res_vis = img_vis_crop.clone() / 255;
                    cv::cvtColor(img_res_vis, img_res_vis, CV_RGB2BGR);
                    this->VisualiseAnnotations(img_res_vis, label_num_channels, cur_label_aug, multfact);
                    cv::imshow("cropped image", img_res_vis);
                }
                img_res = img_crop;
            }

            // show image
            if (visualise)
            {
                cv::Mat img_res_vis = img_res / 255;
                cv::cvtColor(img_res_vis, img_res_vis, CV_RGB2BGR);
                this->VisualiseAnnotations(img_res_vis, label_num_channels, cur_label_aug, multfact);
                cv::imshow("interim resize image", img_res_vis);
            }

            DLOG(INFO) << "Resizing output image.";

            // resize to output image size
            cv::Size s(outsize, outsize);
            cv::resize(img_res, img_res, s);

            // "resize" annotations
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

            // show image
            if (visualise && sub_mean)
            {
                cv::Mat img_res_meansub_vis = img_res / 255;
                cv::cvtColor(img_res_meansub_vis, img_res_meansub_vis, CV_RGB2BGR);
                cv::imshow("mean-removed image", img_res_meansub_vis);
            }

            // multiply by scale
            if (scale != 1.0)
                img_res *= scale;

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
                        top_data[idx_img_aug * img_size + c * channel_size + i * outsize + j] = img_res.at<cv::Vec3f>(i, j)[c];
                    }
                }
            }

            // store label as gaussian
            DLOG(INFO) << "storing labels";
            const int label_channel_size = label_height * label_width;
            const int label_img_size = label_channel_size * label_num_channels / 2;
            cv::Mat dataMatrix = cv::Mat::zeros(label_height, label_width, CV_32FC1);
            float label_resize_fact = (float) label_height / (float) outsize;
            float sigma = 1.5;

            for (int idx_ch = 0; idx_ch < label_num_channels / 2; idx_ch++)
            {
                float x = label_resize_fact * cur_label_aug[2 * idx_ch] * multfact;
                float y = label_resize_fact * cur_label_aug[2 * idx_ch + 1] * multfact;
                for (int i = 0; i < label_height; i++)
                {
                    for (int j = 0; j < label_width; j++)
                    {
                        int label_idx = idx_img_aug * label_img_size + idx_ch * label_channel_size + i * label_height + j;
                        float gaussian = ( 1 / ( sigma * sqrt(2 * M_PI) ) ) * exp( -0.5 * ( pow(i - y, 2.0) + pow(j - x, 2.0) ) * pow(1 / sigma, 2.0) );
                        gaussian = 4 * gaussian;
                        top_label[label_idx] = gaussian;

                        if (idx_ch == 0)
                            dataMatrix.at<float>((int)j, (int)i) = gaussian;
                    }
                }
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
        else
            cur_img_ = 0;
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
        CV_RGB(255, 0, 255)
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
    cv::line(img_annotation_vis, centers[1], centers[3], CV_RGB(0, 255, 0), 1, CV_AA);
    cv::line(img_annotation_vis, centers[2], centers[4], CV_RGB(255, 255, 0), 1, CV_AA);
    cv::line(img_annotation_vis, centers[3], centers[5], CV_RGB(0, 0, 255), 1, CV_AA);
    cv::line(img_annotation_vis, centers[4], centers[6], CV_RGB(0, 255, 255), 1, CV_AA);
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
