# SurfelMeshing #

## github
https://github.com/puzzlepaint/surfelmeshing.git

## Environment
```
ubuntu 20.04
gcc 9.4.0
g++ 9.4.0
NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7
nvcc V10.1.243
```

## How to build this project: 
```
mkdir build
cmake -DCMAKE_CUDA_ARCHITECTURES=50 -DCMAKE_CUDA_FLAGS="-gencode arch=compute_50,code=sm_50" -B build/
cd build
make -j8 SurfelMeshing
```
## RealSense
```
./applications/surfel_meshing/SurfelMeshing /home/roboticslab/r09522848/datasets/data0130_outdoor_ntu_small_loop SurfelMeshing_RealSenseD435i.yaml associated_trajectory7.txt --hide_input_images --follow_input_camera false --export_point_cloud ../results/pcd0320.ply --export_mesh ../results/mesh0320.obj

```

## TUM
```
./applications/surfel_meshing/SurfelMeshing /home/roboticslab/r09522848/datasets/rgbd_dataset_freiburg1_desk2 SurfelMeshing_TUM1.yaml groundtruth.txt --follow_input_camera false
```

## Overview ##

SurfelMeshing is an approach for real-time surfel-based mesh reconstruction from
RGB-D video, described in the following article:

T. Schöps, T. Sattler, M. Pollefeys, "SurfelMeshing: Online Surfel-Based Mesh Reconstruction", PAMI 2019. \[[pdf](http://arxiv.org/abs/1810.00729)\]\[[video](https://youtu.be/CzMtNxuQ0OY)\]
