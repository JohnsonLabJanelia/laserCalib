# laserCalib 

## Install

Use conda to manage python virutal environment 

```
conda create -n lasercalib python=3.9
conda activate lasercalib
conda install numpy
conda install -c anaconda scipy 
conda install -c conda-forge matplotlib 
conda install scikit-image
conda install -c menpo opencv
conda install seaborn
python -m pip install -U prettytable
```

## Basic Workflow
1. Collect images  
- Use syncrhonized cameras. Turn off all the lights in the rig. This code is currently written for calibrating color cameras. If you are using monochrome cameras, please adjust the centroid-finding code to accept single channel images. If using color cameras, please use a green laser pointer. If you'd rather use a blue or red laser pointer, please adjust the centroid-finding code to use the red or blue image channel (right now it is using the green channel to find the green laser pointer centroid in each image). Use short (~100-500 microsecond) camera exposures and low gain such that the only bright spot in the image is from the laser pointer. The goal is to collect several thousand images on each camera of the laser pointer spot. You can shine the spot onto the floor of the arena (see example below).

## TODO 
- [ ] FFmpeg decoder 
- [ ] about 20 minutes to run the centroid extraction, can be parallelized 

## Reference 
The library is built on pySBA -- Python Bundle Adjustment library: https://github.com/jahdiel/pySBA

Camera visualizer -- https://github.com/demul/extrinsic2pyramid

Scipy bundle adjustment cookbook: http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

FFmpeg python wrapper: https://github.com/kkroening/ffmpeg-python 
