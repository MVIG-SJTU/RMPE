# RMPE: Regional Multi-person Pose Estimation

By [Haoshu Fang](https://fang-haoshu.github.io), Shuqin Xie, [Cewu Lu](https://cvsjtu.wordpress.com/).

### Introduction

RMPE is a two steps framework for the task of multi-person pose estimation. You can use the code to train/evaluate a model for pose estimation task. For more details, please refer to our [arXiv paper]().

<p align="center">
<img src="https://github.com/Fang-Haoshu/Fang-Haoshu.github.io/blob/master/images/publications/rmpe/framework.jpg" alt="RMPE Framework" width="600px">
</p>

<center>

| Method | MPII full test *mAP* | s/frame |
|:-------|:-----:|:-------:|:-------:|
| [Iqbal&Gall, ECCVw'16](http://arxiv.org/abs/1608.08526) | 43.1 | 10 |
| [DeeperCut](http://pose.mpi-inf.mpg.de/) | 59.5 | 485 | 
| **[RMPE](https://fang-haoshu.github.io/publications/rmpe/)** | **69.2** | **0.8** |

</center>

### Citing RMPE

Please cite RMPE in your publications if the code or paper helps your research:

    @article{fang16rmpe,
      Title = {{RMPE}: Regional Multi-person Pose Estimation},
      Author = {Haoshu Fang, Shuqin Xie and Cewu Lu },
      Journal = {},
      Year = {2016}
    }

### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Demo](#demo)
4. [Train/Eval](#traineval)
5. [Acknowledgements](#acknowledgements)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  git clone https://github.com/weiliu89/caffe.git
  cd caffe
  git checkout rmpe
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  make py
  make test -j8
  make runtest -j8
  # If you have multiple GPUs installed in your machine, make runtest might fail. If so, try following:
  export CUDA_VISIBLE_DEVICES=0; make runtest -j8
  # If you have error: "Check failed: error == cudaSuccess (10 vs. 0)  invalid device ordinal",
  # first make sure you have the specified GPUs, or try following if you have multiple GPUs:
  unset CUDA_VISIBLE_DEVICES
  ```

### Preparation
#### For demo only
1. Download pre-trained human detector([Google drive](https://drive.google.com/open?id=0BxORzoJl8N9Pck8tZW1wMVlWNjA)|[Baidu cloud](http://pan.baidu.com/s/1hssOFWS)) and SPPE+SSTN model([Google drive](https://drive.google.com/open?id=0BxORzoJl8N9PejV6OUZhUllzQWs)|[Baidu cloud](http://pan.baidu.com/s/1geVQkKj)). By default, we assume the models are stored in `$CAFFE_ROOT/models/VGG_SSD/` and `$CAFFE_ROOT/models/SPPE/` accordingly.

#### For train/eval
1. Download [fully convolutional reduced (atrous) VGGNet](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6). By default, we assume the model is stored in `$CAFFE_ROOT/models/VGG_SSD/`

2. Download [MPII images](http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz) and [COCO14 training set](http://msvocds.blob.core.windows.net/coco2014/train2014.zip). By default, we assume the images are stored in `/data/MPII_COCO14/images/`.

3. Download MPII_COCO14 Annotations([Google drive](https://drive.google.com/open?id=0BxORzoJl8N9PWFhfdzR6S1g1bHc)|[Baidu cloud](http://pan.baidu.com/s/1i4911zB)). By default, we assume the XMLs are stored in the `/data/MPII_COCO14/Annotations/`.

### Demo
Our experiments use both Caffe and Torch7. But we implement the whole framework in Caffe so you can run the demo easily.
1. Run the ipython notebook. It will show you how our whole framework works

  ```Shell
  cd $CAFFE_ROOT
  # make a soft link to the images
  ln -s /data/MPII_COCO14/images/ data/MPII/images
  jupyter notebook examples/rmpe/Regional\ Multi-person\ Pose\ Estimation.ipynb
  # run the python program for more results
  python examples/rmpe/demo.py
  ```

### Train/Eval
1. Train human detector. 
We use the data in MPII and COCO14 to train our human detector. We have already create the train/val list in `CAFFE_ROOT/data/MPII_COCO14` and release our script in `CAFFE_ROOT/examples/rmpe`, so basically what you need to do will be something like
  ```Shell
  # First create the LMDB file.
  cd $CAFFE_ROOT
  # You can modify the parameters in create_data.sh if needed.
  # It will create lmdb files for trainval and test with encoded original image:
  #   - /data/MPII_COCO14/lmdb/MPII_COCO14_trainval_lmdb
  #   - /data/MPII_COCO14/lmdb/MPII_COCO14_test_lmdb
  # and make soft links at examples/MPII_COCO14/
  ./data/MPII_COCO14/create_data.sh
  # It will create model definition files and save snapshot models in:
  #   - $CAFFE_ROOT/models/VGG_SSD/MPII_COCO14/SSD_500x500/
  # and job file, log file, and the python script in:
  #   - $CAFFE_ROOT/jobs/VGG_SSD/MPII_COCO14/SSD_500x500/
  # and save temporary evaluation results in:
  #   - $HOME/data/MPII_COCO14/results/SSD_500x500/
  # It should reach 85.* mAP at 60k iterations.
  python examples/rmpe/ssd_pascal_MPII_COCO14VGG.py
  ```

2. Train SPPE+SSTN.
This part of our model is implemented in Torch7. Please refer to [this repo]() for more details.
We will call the directory that you cloned the repo into `$SPPE_ROOT`.
Note that I am currently working on an implementation in Caffe. The script may come out soon.


3. Evaluate the model. You can modify line 45 in `demo.py` to evaluate our framework on whole test set. But the results may be slightly different from our work. To reproduce our results reported in our paper:
  ```Shell
  # First get the result of human detector
  cd $CAFFE_ROOT
  jupyter notebook examples/rmpe/human_detection.ipynb
  # Then move the results to $SPPE_ROOT/predict/annot/
  mv examples/rmpe/mpii-test0.09 $SPPE_ROOT/predict/annot/
  # Next, do single person human estimation
  cd $SPPE_ROOT/predict
  th main.lua predict-test
  #Finally, do pose NMS
  python batch_nms.py
  #our result is stored in txt format, to evaluate, Download MPII [toolkit](http://human-pose.mpi-inf.mpg.de/#evaluation) and put it in current directory
  matlab
  #In matlab
  set_pred()
  ```

### Acknowledgements

Thanks to [Wei Liu](https://github.com/weiliu89/caffe/tree/ssd), [Alejandro Newell](https://github.com/anewell/pose-hg-train), [Pfister, T.](https://github.com/tpfister/caffe-heatmap), [Kaichun Mo](https://github.com/daerduoCarey/SpatialTransformerLayer), [Maxime Oquab](https://github.com/qassemoquab/stnbhwd) for contributing their codes. 
Thanks to the authors of Caffe and Torch7!