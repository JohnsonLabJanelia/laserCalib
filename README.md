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
We are using Linux. You will need to install `ffmpeg` on your machine such that the `ffmpeg` command works from the terminal.  

## Basic Workflow  
1. Collect videos  
- Use syncrhonized cameras. Turn off all the lights in the rig. This code is currently written for calibrating color cameras. If you are using monochrome cameras, please adjust the centroid-finding code to accept single channel images. If using color cameras, please use a green laser pointer. If you'd rather use a blue or red laser pointer, please adjust the centroid-finding code to use the red or blue image channel (right now it is using the green channel to find the green laser pointer centroid in each image). Use short (~100-500 microsecond) camera exposures and low gain such that the only bright spot in the image is from the laser pointer. The goal is to collect several thousand images on each camera of the laser pointer spot. You can shine the spot onto the floor of the arena (see example below).  
2. Place all the laser videos in a folder called `movies`, under the parent root folder. 
3. Run the `find_laser_points.py` script to extract centroids from these images and save them to a `/results/*.pkl` file. For instance: 
```
python find_laser_points.py --root_dir=/home/user/Calibration/16cam --frame_range 0 5000 --width 3208 --height 2200
```
4. Run `prepare_points3d.py` to prepare datasets for calibration. It will use one camera initial parameters, and ground truth z plane value to infer points 3d locations. Please select a camera that doesn't move much. 
```
python prepare_points3d.py -f /media/user/data0/laser_calibrate_2024_4_1/2024_04_01_17_24_00/ /media/user/data0/laser_calibrate_2024_4_1/2024_04_01_17_09_58/ -z 105 0 -o /media/user/data0/laser_calibrate_2024_4_1/2024_04_01_17_09_58/ -n Cam710038
```
5. Run `calibrate.py` to calibrate the cameras (edit to adjust for cam number, cam param initialization, and 3D point initialization.  
```
python calibrate.py --root_dir=/home/user/Calibration/16cam
```
The calibration result is saved in `results/sba.pkl`.
Note: 
- It is probably important to initialize these camera parameters in the ballpark of the expected output. These calibration parameters could be estimated from a previous calibration or from a drawing/digital twin of the rig. To innitialize camera postions from blender model, put the `camera_dicts.pkl` in `results`.  
- You will also need to provide an initial guess of the 3D world coordinates for each laser pointer observation. We simply used the pixel location of the extracted laser pointer centroids from one of our cameras (the central ceiling camera; blue camera in image below).  
6. Use the `movie_viewer.py` to run aruco based feature detection. 
```
python movie_viewer.py --n_cams=16 --root_dir=/home/user/Calibration/16cam --mode aruco
```
Press `p` to pause, press `q` to quit, and the aruco corner will be saved. Make sure all the corners are detected. 

7. Run `aruco_triangulate.py` to triangulate corresponding points from aruco marker. It triangulates the centers of aruco markers  and all the corners of the aruco markers. Also, it epirically estiamtes the scaling factor using ground truth side length of aruco marker (side length in millimeter).  
```
python aruco_triangulate.py --n_cams=16 --root_dir=/home/user/Calibration/16cam --side_len=120
```

8. Run `label2world.py` fit a rigid body transformation to change the coordinate to desired world coordinate. 
```
python label2world.py --n_cams=16 --root_dir=/home/user/Calibration/16cam --use_scale=1
```
If use_scale is true, it will use the scale factor estimated from step 7, otherwise it is set to 1 (no scaling).

9. Run `verify_world_transform.py`` to double check the alignment of step 8. 
```
python verify_world_transform.py --n_cams=17 --root_dir=/media/user/data0/laser_calibrate_2024_4_1/2024_04_01_17_09_58/ --side_len=120 --yaml_dir_name=/media/user/data0/laser_calibrate_2024_4_1/2024_04_01_17_09_58/results/rigspace/calibration_rig/
```
10. Run 'plot_from_yaml.py' to visualize the camera extrinsics. 
```
python plot_from_yaml.py --yaml_dir=/media/user/data0/laser_calibrate_2024_4_1/2024_04_01_17_09_58/results/rigspace/calibration_rig/
```

Some utility function:
1. `movie_viewer.py`: visualize laser detection algorithm, `python movie_viewer.py --n_cams=16 --root_dir=/home/user/Calibration/16cam --mode laser`. 
2. `get_video_pixel.py`: find the pixel rgb value for mouse selectable pixel. 

## Example of a suitable input image  
This is a suitable image. The green laser pointer is the brightest and largest green spot in the image. Good job. As written, the program will reject laser pointer points that are >1300 pixels from the center of the image.   
![suitable_input_image](README_images/suitable_input_image.png)  

## Example of calibrated cameras and extracted laser points in world space  
![laser_points_and_cam_positions](README_images/laser_points_and_cam_positions.png)  

## Reference  
The library is built on pySBA -- Python Bundle Adjustment library: https://github.com/jahdiel/pySBA  

Camera visualizer -- https://github.com/demul/extrinsic2pyramid  

Scipy bundle adjustment cookbook: http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html  

FFmpeg python wrapper: https://github.com/kkroening/ffmpeg-python  

## Acknowledgements  
We thank he authors of pyBSA (https://github.com/jahdiel), the Scipy Cookbook, and the Camera Visualizer Tool (https://github.com/demul) used in this project. We adapted their code (see Reference Section). We thank Selmaan Chettih (https://twitter.com/selmaanchettih) for advice and help in getting this project started with pyBSA. We thank David Schauder (https://github.com/porterbot) and Ratan Othayoth (https://github.com/othayoth) for help in synchronizing cameras.  
