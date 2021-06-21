#! /bin/bash

root=$(pwd)
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip install tensorflow-gpu==1.15.0 tensorboardX==2.1 matplotlib
pip install open3d==0.9.0
#pip install scikit-learn
#pip install tqdm

#now compile the chamfer distance
#exporting CUDA variables - TODO: check that paths match your CUDA installation
export CUDA_HOME="/usr/local/cuda-10.0"
export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64":$LD_LIBRARY_PATH
export PATH="/usr/local/cuda-10.0/bin":$PATH
export CPATH="/usr/local/cuda-10.0/include":$CPATH
cd torch-nndistance || exit
rm -i -r ${root}/torch-nndistance/build
python build.py install
