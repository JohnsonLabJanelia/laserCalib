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
2. Place all the laser videos in a folder called `laser_movies`, under the parent root folder. 
3. Run the `find_laser_points.py` script to extract centroids from these images and save them to a `/results/*.pkl` file. For instance: 
```
python find_laser_points.py --n_cams=8 --root_dir=/home/user/Calibration/16cam --frame_range 0 5000 --width 3208 --height 2200
```
5. Run `calibrate.py` to calibrate the cameras (edit to adjust for cam number, cam param initialization, and 3D point initialization.  
```
python calibrate.py --root_dir=/home/user/Calibration/16cam --cam_id_for_3d_init 4
```
The calibration result is saved in `results/sba_blender.pkl`.
Note: 
- It is probably important to initialize these camera parameters in the ballpark of the expected output. These calibration parameters could be estimated from a previous calibration or from a drawing/digital twin of the rig. To innitialize camera postions from blender model, put the `camera_dicts.pkl` in `results`.  
- You will also need to provide an initial guess of the 3D world coordinates for each laser pointer observation. We simply used the pixel location of the extracted laser pointer centroids from one of our cameras (the central ceiling camera; blue camera in image below).  
6. Run the `save_cam_params.py` to save out your calibration parameters as a `.csv` file for our labeling tool `red`, and for aruco calibration for transfering the coordinates to a desired world coordiate.
```
python save_cam_params.py --n_cams=16 --root_dir=/home/user/Calibration/16cam
```
7. Use the `movie_viewer.py` to run aruco based feature detection. 
```
python movie_viewer.py --n_cams=16 --root_dir=/home/user/Calibration/16cam --mode aruco
```
Press `p` to pause, press `q` to quit, and the aruco corner will be saved. Make sure all the corners are detected. 

8. Run `aruco_triangulate_*.py` to triangulate corresponding points from aruco marker. `aruco_triangulate_center.py` triangulate the centers of aruco markers, while `aruco_triangulate_corners.py` triangulate all the corners of the aruco markers, and epirically estiamte the scaling factor using ground truth side length of aruco marker (side length in millimeter).  
```
python aruco_triangulate_corners.py --n_cams=16 --root_dir=/home/user/Calibration/16cam --side_len=120
```
```
python aruco_triangulate_centers.py --n_cams=16 --root_dir=/home/user/Calibration/16cam
```
9. Run `label2world.py` fit a rigid body transformation to change the coordinate to desired world coordinate. 
```
python label2world.py --n_cams=16 --root_dir=/home/user/Calibration/16cam --refit=1
```
Set refit to 1 if want to refit the model with transformed points. 

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
