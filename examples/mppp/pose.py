from __future__ import print_function
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess
import sys

# Add STN layers on top of a "base" network (e.g. Stacked Hourglass).
def AddSTN(net,from_layer,freeze=False):
    # Add locnet, we use SqueezeNet here
    SqueezeNetBody(net,from_layer,for_HG_module=True, freeze=freeze)
    bottom = []
    bottom.append(net[from_layer])
    bottom.append(net.theta)
    st_param = {
        'to_compute_dU': False,
    }
    net.stn = L.SpatialTransformer(*bottom,st_param=st_param)
    return net

# Add SDTN layers on top of a "base" network (e.g. Stacked Hourglass).
def AddSDTN(net,from_layer):
    st_param = {
        'to_compute_dU': True,
        'de_transform': True,
    }
    bottom = []
    bottom.append(net[from_layer])
    bottom.append(net.theta)
    net.sdtn = L.SpatialTransformer(*bottom,st_param=st_param)
    return net
### Modify the following parameters accordingly ###
# The directory which contains the caffe code.
# We assume you are running the script at the CAFFE_ROOT.
caffe_root = os.getcwd()

# Set true if you want to start training right after generating all files.
run_soon = True
# Set true if you want to load from most recently saved snapshot.
# Otherwise, we will load from the pretrain_model defined below.
resume_training = False
# If true, Remove old model files.
remove_old_models = False

# The database file for training data. Created by data/VOC0712/create_data.sh
train_data = "data/MPII/train_bbox.txt"
# The database file for testing data. Created by data/VOC0712/create_data.sh
test_data = "data/MPII/train_bbox.txt"
# The root dir of your images
img_dir = "data/MPII/images/"
# Specify the batch sampler.
resize_width = 256
resize_height = 256
resize = "{}x{}".format(resize_width, resize_height)

# The name of the model. Modify it if you want.
model_name = "SHG_MPII_{}".format(resize)

# Directory which stores the model .prototxt file, the snapshot of models, the job script and log file, the detection results.
save_dir = "models/SHG/MPII/{}".format(model_name)

# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
# snapshot prefix.
snapshot_prefix = "{}/{}".format(save_dir, model_name)
# job script path.
job_file = "{}/{}.sh".format(save_dir, model_name)


# The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
pretrain_model = "/media/fred/linux/patch0.73.caffemodel"

# Solver parameters.
# Defining which GPUs to use.
gpus = "0"
gpulist = gpus.split(",")
num_gpus = len(gpulist)

# Divide the mini-batch to different GPUs.
batch_size = 1
accum_batch_size = 1
iter_size = accum_batch_size / batch_size
solver_mode = P.Solver.CPU
device_id = 0
batch_size_per_device = batch_size
if num_gpus > 0:
  batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
  iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
  solver_mode = P.Solver.GPU
  device_id = int(gpulist[0])

# Evaluate on whole test set.
num_test_image = 3
test_batch_size = 1
test_iter = num_test_image / test_batch_size


train_heatmap_data_param = {
    'source': train_data,
    'bbox_or_centerscale' : "bbox",
    'root_img_dir': img_dir,
    'batchsize': batch_size,
    'outsize': 256,
    'sample_per_cluster': False,
    'data_augment': True,
    'label_width': 64,
    'label_height': 64,
    'segmentation': False,
    'angle_max': 30,
    'multfact': 1,  # set to 282 if using preprocessed data from website
  }
test_heatmap_data_param = {
    'source': test_data,
    'bbox_or_centerscale' : "bbox",
    'root_img_dir': img_dir,
    'batchsize': test_batch_size,
    'outsize': 256,
    'sample_per_cluster': False,
    'data_augment': False,
    'label_width': 64,
    'label_height': 64,
    'segmentation': False,
    'multfact': 1,  # set to 282 if using preprocessed data from website
  }

solver_param = {
    # Train parameters
    'base_lr': 2.5e-4,
    'weight_decay': 0.0,
    'lr_policy': "fixed",
    'momentum': 0.0,
    'iter_size': iter_size,
    'max_iter': 60000,
    'snapshot': 5000,
    'display': 1,
    'average_loss': 10,
    'type': "RMSProp",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    'test_iter': [test_iter],
    'test_interval': 1000,
    }


### Hopefully you don't need to change the following ###
# Check file.
check_if_exist(train_data)
check_if_exist(test_data)
check_if_exist(pretrain_model)
make_if_not_exist(save_dir)

# Create train net.
net = caffe.NetSpec()
net.data, net.label = CreateHeatmapDataLayer(output_label=True, train=True, visualise=False,
        heatmap_data_param=train_heatmap_data_param)

AddSTN(net,'data')
HGStacked(net, from_layer='stn', freeze=True)
AddSDTN(net,net.keys()[-1])

last_layer = [net[net.keys()[-1]]]
last_layer.append(net.label)
last_layer.append(net.data)
net.heatmap_loss = L.EuclideanLossHeatmap(*last_layer, visualise=False, visualise_channel=4,
        include=dict(phase=caffe_pb2.Phase.Value('TRAIN'))
        )

with open(train_net_file, 'w') as f:
    print('name: "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

# Create test net.
net = caffe.NetSpec()
net.data, net.label = CreateHeatmapDataLayer(output_label=True, train=False, visualise=False,
        heatmap_data_param=test_heatmap_data_param)

AddSTN(net,'data')
HGStacked(net, from_layer='stn', freeze=True)
AddSDTN(net,net.keys()[-1])

last_layer = [net[net.keys()[-1]]]
last_layer.append(net.data)
net.prediction_heatmap = L.PredictionHeatmap(*last_layer,
    num_joints=16, include=dict(phase=caffe_pb2.Phase.Value('TEST')))
net.accuracy_heatmap = L.AccuracyHeatmap(*[net.prediction_heatmap,net.label],
    threshold=0.5, include=dict(phase=caffe_pb2.Phase.Value('TEST')))

with open(test_net_file, 'w') as f:
    print('name: "{}_test"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

# Create deploy net.
# Remove the first and last layer from test net.
deploy_net = net
with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (AccuracyHeatmap) layer from test net.
    del net_param.layer[0]
    del net_param.layer[-1]
    net_param.name = '{}_deploy'.format(model_name)
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
    print(net_param, file=f)

# Create solver.
solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        test_net=[test_net_file],
        snapshot_prefix=snapshot_prefix,
        **solver_param)

with open(solver_file, 'w') as f:
    print(solver, file=f)

max_iter = 0
# Find most recent snapshot.
for file in os.listdir(save_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format(model_name))[1])
    if iter > max_iter:
      max_iter = iter

train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
if resume_training:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

if remove_old_models:
  # Remove any snapshots smaller than max_iter.
  for file in os.listdir(save_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(save_dir, file))
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(save_dir, file))

# Create job file.
with open(job_file, 'w') as f:
  f.write('cd {}\n'.format(caffe_root))
  f.write('./build/tools/caffe train \\\n')
  f.write('--solver="{}" \\\n'.format(solver_file))
  f.write(train_src_param)
  if solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, save_dir, model_name))
  else:
    f.write('2>&1 | tee {}/{}.log\n'.format(save_dir, model_name))

# Copy the python script to save_dir.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, save_dir)

# Run the job.
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
  subprocess.call(job_file, shell=True)
