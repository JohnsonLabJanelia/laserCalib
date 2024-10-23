# laserCalib

`laserCalib` is a python package to calibrate mulitple, synchronized cameras. We estimate intrinsics and extrinsics for all the cameras using sparse bundle adjustment.

## overview
Here is a quick overview of steps involved and what the package does:

1. setup synchronized recording from multiple cameras 
2. provide initial estimates of calibration parameters (camera intrinsics and extrinsics) for all cameras
3. record videos of a moving laser point that is preferably visible in all cameras 
4. detect the 2d coordinates of the laser point in each frame across all camera views
5. estimate the 3d coordinates of the laser point using 2d coordinates (from step 4) and initial camera calibration parameters (step 2)
6. use sparse bundle adjustment to iteratively refine the estimates of camera calibration parameters and 3d laser point coordinates and until they converge to an optimal estimate
7. register the camera positions and orientations relative to fixed landmarks in the recording arena (world coordinates) to get 3d triangulation results in world coordinate origin


## install

setup a python virtual environment using conda

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

- we use Linux. 
- you will also need to install `ffmpeg` on your machine such that the `ffmpeg` command works from the terminal. 


In laserCalib folder, install this library as a dev package
```
git clone git@github.com:JohnsonLabJanelia/laserCalib.git
git checkout mouse_rig
conda activate lasercalib
pip install -e  .
```


### Collect datasets 
1. Laser pointer videos 


- Use synchronized cameras. Turn off all the lights in the rig. This code is currently written for calibrating color cameras. If you are using monochrome cameras, please adjust the centroid-finding code to accept single channel images. If using color cameras, please use a green laser pointer. If you'd rather use a blue or red laser pointer, please adjust the centroid-finding code to use the red or blue image channel (right now it is using the green channel to find the green laser pointer centroid in each image). Use short (~100-500 microsecond) camera exposures and low gain such that the only bright spot in the image is from the laser pointer. The goal is to collect several thousand images on each camera of the laser pointer spot. You can shine the spot onto the floor of the arena (see example below).
- Collect laser point videos for two planes. Please also provide the ground truth z-plane in the calibration config file in the world frame. The ground truth z-planes are used for unprojecting 2d point to 3d.


2. Collect short videos of aruco markers. Aruco markers are used as global landmarks for the world frame registration. The center of the markers are used for registration. Users must provide ground truth 3d coordinates in the calibration config file. .


3. Create a config file for calibration. Example config file is provided in the example folder.


4. Initial estimation of the cameras are required due to many local minimums in multiview bundle adjustment. Future work will use two-view geometry to remove this constraint. Please put the folder that contains camera initial parameter estimation in the same folder as the config file. 

<strong>Example provided in the example folder</strong>. 


### Calibration steps

0. Prepare a directory with recorded videos, initial calibration parameters, and a `config` file 

Let's say it is the directory `/home/ro/exp/calib/2024_10_22` and it looks like below: 

```
.
├── 2024_10_22_16_32_02            <-- first set of laser video (at z=z1)
│   ├── Cam2002486_meta.csv
│   ├── Cam2002486.mp4
│   ├── ....
│   ├── Cam2008670_meta.csv
│   └── Cam2008670.mp4
├── 2024_10_22_16_35_36           <-- second set of laser video (at z=z2)
│   ├── Cam2002486_meta.csv
│   ├── Cam2002486.mp4
│   ├── ....
│   ├── Cam2008670_meta.csv
│   └── Cam2008670.mp4
├── 2024_10_22_16_43_05           <-- set of aruco marker videos
│   ├── Cam2002486_meta.csv
│   ├── Cam2002486.mp4
│   ├── ....
│   ├── Cam2008670_meta.csv
│   └── Cam2008670.mp4
├── calib_init                    <-- initial camera calibration parameters
│   ├── Cam2002486.yaml
│   ├── ....
│   └── Cam2008670.yaml
└── config.json                   <-- configuration file
    
```

Go to the scripts subdirectory in the `lasercalib` package,

```
cd scripts 
export calib_dataset=/home/ro/exp/calib/2024_10_22
```


1. Extract 2d laser points 
```
python detect_laser_points.py -c $calib_dataset -i 0
python detect_laser_points.py -c $calib_dataset -i 1
```
`-i` specifies the dataset index. Order is specified in the `config.json` file. 

to run on all the laser datasets listed in the config in sequence:

```
python detect_laser_points.py -c $calib_dataset
```


2. infer 3d points
```
python get_points3d.py -c $calib_dataset
```


3. run calibration
```
python calibrate_camera.py -c $calib_dataset
```


4. extract aruco markers (landmarks), press `q` to exit. 
```
python run_viewers.py -c $calib_dataset -m aruco
```
- Use aruco markers from the dictionary `DICT_4X4_100` -- see TODO for some examples. 
- specify the marker ids and physical 3d coordinates of the marker centers in the `config.json` file.


5. triangulate aruco markers
```
python triangulate_aruco.py -c $calib_dataset
```


6. register to the world coordinate
```
python register_world.py -c $calib_dataset
```


7. To check on the results
```
python verify_world.py -c $calib_dataset
```


Final results are saved in $calib_dataset/results/calibration_rig/




### Some utility function:
1. `movie_viewer.py`: visualize laser detection, `python movie_viewer.py --n_cams=16 --root_dir=/home/user/Calibration/16cam --mode laser`.
2. `get_video_pixel.py`: find the pixel rgb value for mouse selectable pixel.


## Example of a suitable input image 
This is a suitable image. The green laser pointer is the brightest and largest green spot in the image. Good job. As written, the program will reject laser pointer points that are >1300 pixels from the center of the image.  
![suitable_input_image](README_images/suitable_input_image.png) 


## Example of calibrated cameras and their pose in world frame
![camera_pose_with_frames](README_images/cameras_pose_with_frames.svg)
## Reference 
The library is built on pySBA -- Python Bundle Adjustment library: https://github.com/jahdiel/pySBA 


Camera visualizer -- https://github.com/demul/extrinsic2pyramid 


Scipy bundle adjustment cookbook: http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html 


FFmpeg python wrapper: https://github.com/kkroening/ffmpeg-python 


Multiple view Camera calibration tool: https://github.com/cvlab-epfl/multiview_calib


## Acknowledgements 
We thank he authors of pyBSA (https://github.com/jahdiel), the Scipy Cookbook, and the Camera Visualizer Tool (https://github.com/demul) used in this project. We adapted their code (see Reference Section). We thank Selmaan Chettih (https://twitter.com/selmaanchettih) for advice and help in getting this project started with pyBSA. We thank David Schauder (https://github.com/porterbot) and Ratan Othayoth (https://github.com/othayoth) for help in synchronizing cameras. 



