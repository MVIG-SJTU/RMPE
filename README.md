# RMPE: Regional Multi-person Pose Estimation

By [Haoshu Fang](https://fang-haoshu.github.io), Shuqin Xie, [Cewu Lu](https://cvsjtu.wordpress.com/).

### Introduction

RMPE is a two steps framework for the task of multi-person pose estimation. You can use the code to train/evaluate a model for pose estimation task. For more details, please refer to our [arXiv paper](http://arxiv.org/abs/1512.02325).

<p align="center">
<img src="https://github.com/Fang-Haoshu/Fang-Haoshu.github.io/blob/master/images/publications/rmpe/framework.jpg" alt="RMPE Framework" width="600px">
</p>

<center>

| Method | MPII full test *mAP* | s/frame |
|:-------|:-----:|:-------:|:-------:|
| [Iqbal&Gall, ECCVw'16](http://arxiv.org/abs/1608.08526) | 43.1 | 7 |
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
3. [Train/Eval](#traineval)
4. [Models](#models)

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
1. Download [fully convolutional reduced (atrous) VGGNet](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6). By default, we assume the model is stored in `$CAFFE_ROOT/models/VGGNet/`

2. Download [MPII dataset](http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz). By default, untar the file and make a soft link to `$CAFFE_ROOT/data/MPII/images`. You can jump this step if you only need to run the demo.

### Demo
Our experiments use both Caffe and Torch7. But we implement the whole framework in Caffe so you can run the demo easily.
1. Run the ipython notebook
  ```Shell
  # It will show how our whole framework works
  cd $CAFFE_ROOT
  jupyter notebook examples/rmpe/Regional Multi-person Pose Estimation.ipynb
  python examples/ssd/ssd_pascal.py
  ```
  If you don't have time to train your model, you can download a pre-trained model at [here](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_300x300.tar.gz).

2. Run the python program
  ```Shell
  # It is basically the same as the ipython notebook, shows more results with a loop.
  python examples/rmpe/demo.py
  ```

3. Test your model using a webcam. Note: press <kbd>esc</kbd> to stop.
  ```Shell
  # If you would like to attach a webcam to a model you trained, you can do:
  python examples/ssd/ssd_pascal_webcam.py
  ```
  [Here](https://drive.google.com/file/d/0BzKzrI_SkD1_R09NcjM1eElLcWc/view) is a demo video of running a SSD500 model trained on [MSCOCO](http://mscoco.org) dataset.

4. Check out `examples/ssd_detect.ipynb` or `examples/ssd/ssd_detect.cpp` on how to detect objects using a SSD model.

5. To train on other dataset, please refer to data/OTHERDATASET for more details.
We currently add support for MSCOCO and ILSVRC2016.


### Train/Eval
1. Train your model and evaluate the model on the fly.
  ```Shell
  # It will create model definition files and save snapshot models in:
  #   - $CAFFE_ROOT/models/VGGNet/VOC0712/SSD_300x300/
  # and job file, log file, and the python script in:
  #   - $CAFFE_ROOT/jobs/VGGNet/VOC0712/SSD_300x300/
  # and save temporary evaluation results in:
  #   - $HOME/data/VOCdevkit/results/VOC2007/SSD_300x300/
  # It should reach 72.* mAP at 60k iterations.
  python examples/ssd/ssd_pascal.py
  ```
  If you don't have time to train your model, you can download a pre-trained model at [here](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_300x300.tar.gz).

2. Evaluate the most recent snapshot.
  ```Shell
  # If you would like to test a model you trained, you can do:
  python examples/ssd/score_ssd_pascal.py
  ```

3. Test your model using a webcam. Note: press <kbd>esc</kbd> to stop.
  ```Shell
  # If you would like to attach a webcam to a model you trained, you can do:
  python examples/ssd/ssd_pascal_webcam.py
  ```
  [Here](https://drive.google.com/file/d/0BzKzrI_SkD1_R09NcjM1eElLcWc/view) is a demo video of running a SSD500 model trained on [MSCOCO](http://mscoco.org) dataset.

4. Check out `examples/ssd_detect.ipynb` or `examples/ssd/ssd_detect.cpp` on how to detect objects using a SSD model.

5. To train on other dataset, please refer to data/OTHERDATASET for more details.
We currently add support for MSCOCO and ILSVRC2016.

### Models
1. Models trained on VOC0712: [SSD300](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_300x300.tar.gz), [SSD500](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_500x500.tar.gz)

2. Models trained on MSCOCO trainval35k: [SSD300](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_coco_SSD_300x300.tar.gz), [SSD500](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_coco_SSD_500x500.tar.gz)

3. Models trained on ILSVRC2015 trainval1: [SSD300](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_ilsvrc15_SSD_300x300.tar.gz), [SSD500](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_ilsvrc15_SSD_500x500.tar.gz) (46.4 mAP on val2)
