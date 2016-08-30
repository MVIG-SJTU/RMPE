// Copyright 2014 Tomas Pfister

#ifndef CAFFE_HEATMAP_HPP_
#define CAFFE_HEATMAP_HPP_

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


template<typename Dtype>
class DataHeatmapLayer: public BasePrefetchingDataLayer<Dtype>
{

public:

    explicit DataHeatmapLayer(const LayerParameter& param)
        : BasePrefetchingDataLayer<Dtype>(param) {}
    virtual ~DataHeatmapLayer();
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "DataHeatmap"; }

    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int ExactNumTopBlobs() const { return 2; }


protected:
    virtual void load_batch(Batch<Dtype>* batch);

    // Filename of current image
    inline void GetCurImg(string& img_name, std::vector<float>& img_class, std::vector<float>& crop_info, int& cur_class);

    inline void AdvanceCurImg();

    // Visualise point annotations
    inline void VisualiseAnnotations(cv::Mat img_annotation_vis, int numChannels, std::vector<float>& cur_label, int width);

    // Random number generator
    inline float Uniform(const float min, const float max);

    // Rotate image for augmentation
    inline cv::Mat RotateImage(cv::Mat src, float rotation_angle);

    // Global vars
    shared_ptr<Caffe::RNG> rng_data_;
    shared_ptr<Caffe::RNG> prefetch_rng_;
    vector<std::pair<std::string, int> > lines_;
    int lines_id_;    
    int datum_channels_;
    int datum_height_;
    int datum_width_;
    int datum_size_;
    int num_means_;
    int cur_class_;
    vector<int> labelinds_;
    vector<cv::Mat> mean_img_;
    bool sub_mean_;  // true if the mean should be subtracted
    bool sample_per_cluster_; // sample separately per cluster?
    string root_img_dir_;
    vector<float> cur_class_img_; // current class index
    int cur_img_; // current image index
    vector<int> img_idx_map_; // current image indices for each class

    // array of lists: one list of image names per class
    vector< vector< pair<string, pair<vector<float>, pair<vector<float>, int> > > > > img_list_;

    // vector of (image, label) pairs
    vector< pair<string, pair<vector<float>, pair<vector<float>, int> > > > img_label_list_;    
};

}

#endif /* CAFFE_HEATMAP_HPP_ */
