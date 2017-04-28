# RMPE: Regional Multi-person Pose Estimation
By [Haoshu Fang](https://fang-haoshu.github.io), Shuqin Xie, Yuwing Tai, [Cewu Lu](https://cvsjtu.wordpress.com/).

### Introduction

RMPE is a two steps framework for the task of multi-person pose estimation. You can use the code to train/evaluate a model for pose estimation task. For more details, please refer to our arxiv [paper](https://arxiv.org/abs/1612.00137).

<p align="center">
<img src="https://github.com/fang-haoshu/RMPE/blob/master/readme/new-framework.jpg" alt="RMPE Framework" width="600px">
</p>

### Results
<p align="left">
<img src="https://github.com/Fang-Haoshu/RMPE/blob/master/readme/demo.gif", width="720">
</p>

Video results available [here](https://www.youtube.com/watch?v=RHNdbEY5xn4)

Results on MPII dataset:
<center>

| Method | MPII full test *mAP* | s/frame |
|:-------|:-----:|:-------:|
| [Iqbal & Gall, ECCVw'16](http://arxiv.org/abs/1608.08526) | 43.1 | 10 |
| [DeeperCut, ECCV16](http://pose.mpi-inf.mpg.de/) | 59.5 | 485 |
| **[RMPE](https://github.com/fang-haoshu/RMPE)** | **76.7** | **1.5** |

</center>

Results on COCO test-dev 2015:
<center>

| Method | AP @0.5:0.95 | AP @0.5 | AP @0.75 |
|:-------|:-----:|:-------:|:-------:|
| **[RMPE](https://github.com/fang-haoshu/RMPE)** | **61.8** | **83.7** | **69.8** |

</center>

### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Demo](#demo)
4. [Train/Eval](#traineval)
5. [Acknowledgements](#acknowledgements)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  git clone https://github.com/fang-haoshu/rmpe.git
  cd rmpe
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
1. Download pre-trained human detector([Google drive](https://drive.google.com/open?id=0BxORzoJl8N9Pck8tZW1wMVlWNjA)|[Baidu cloud](http://pan.baidu.com/s/1hssOFWS)) and SPPE+SSTN caffe model([Google drive](https://drive.google.com/open?id=0BxORzoJl8N9PejV6OUZhUllzQWs)|[Baidu cloud](http://pan.baidu.com/s/1geVQkKj)). By default, we assume the models are stored in `$CAFFE_ROOT/models/VGG_SSD/` and `$CAFFE_ROOT/models/SPPE/` accordingly.

#### For train/eval
This part of our model is implemented in Torch7. Please refer to [this repo](https://github.com/fang-haoshu/multi-human-pose) for more details.

### Demo
Our experiments use both Caffe and Torch7. But we implement the whole framework in Caffe so you can run the demo easily.
_Note: The current caffe model of SPPE use the 2-stacked hourglass network which has a lower precision. We will be grateful if anyone can help to transfer [new torch model](https://pan.baidu.com/s/1i4LJn97) to caffe._

1. Run the ipython notebook. It will show you how our whole framework works

  ```Shell
  cd $CAFFE_ROOT
  # it shows how our framework works
  jupyter notebook examples/rmpe/Regional\ Multi-person\ Pose\ Estimation.ipynb
  ```  
  
2. Run the python program for more results

  ```Shell
  python examples/rmpe/demo.py
  ```  

### Train/Eval

1. Train SPPE+SSTN.
This part of our model is implemented in Torch7. Please refer to [this repo](https://github.com/fang-haoshu/multi-human-pose) for more details.
We will call the directory that you cloned the repo into `$SPPE_ROOT`.
I have written an implementation in Caffe. You can email me for the script.


2. Evaluate the model. You can modify line 45 in `demo.py` to evaluate our framework on whole test set. But the results will be different. To reproduce our results reported in our paper:
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
  #our result is stored in txt format, to evaluate, Download MPII toolkit and put it in current directory
  matlab
  #In matlab
  setpred()
  ```

### Acknowledgements

Thanks to [Wei Liu](https://github.com/weiliu89/caffe/tree/ssd), [Alejandro Newell](https://github.com/anewell/pose-hg-train), [Pfister, T.](https://github.com/tpfister/caffe-heatmap), [Kaichun Mo](https://github.com/daerduoCarey/SpatialTransformerLayer), [Maxime Oquab](https://github.com/qassemoquab/stnbhwd) for contributing their codes. 
Thanks to the authors of Caffe and Torch7!
