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

### Collect datasets  
1. Laser pointer videos  
- Use syncrhonized cameras. Turn off all the lights in the rig. This code is currently written for calibrating color cameras. If you are using monochrome cameras, please adjust the centroid-finding code to accept single channel images. If using color cameras, please use a green laser pointer. If you'd rather use a blue or red laser pointer, please adjust the centroid-finding code to use the red or blue image channel (right now it is using the green channel to find the green laser pointer centroid in each image). Use short (~100-500 microsecond) camera exposures and low gain such that the only bright spot in the image is from the laser pointer. The goal is to collect several thousand images on each camera of the laser pointer spot. You can shine the spot onto the floor of the arena (see example below). 
- Collect laser point videos for two planes. Please also provide the ground truth z-plane in the calibration config file in the world frame. This is used for unprojecting 2d point to 3d. 
2. Aruco marker videos 
Aruco markers are used as global landmarks for the world frame registration. The center of the markers are used for registration. User must provide ground truth 3d coordinates in the calibration config file. Collect short video of aruco markers.
3. Create a config file for calibration. Example config file is provided in example folder. 
4. Initial estimation of the cameras are required due to many local minimum in multiview bundle adjustment. Future work will use two-view geometry to remove this constraint. Please put the folder that contains camera initial parameter estimation in the same folder as config file. Example provided in the example folder.  

### Calibration steps
Assuming the config.json is in the folder /media/user/data0/laser_calib_2024_05_02_tutorial/calib/results/calibration_rig/

In the script folder, run 
1. Extract laser points
```
python detect_laser_points.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/ -i 0
python detect_laser_points.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/ -i 1
```

2. infer 3d points
```
python get_points3d.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/

```

3. run calibration
```
python calibrate_camera.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/
```

4. run viewers to extract aruco markers, press `q` to exit.  
```
python run_viewers.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/ -m aruco
```

5. triangulate aurco markers
```
python triangulate_aruco.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/
```

6. register to the world coordinate
```
python register_world.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/
```

7. To check on the results
```
python verify_world.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/
```

Final reasults in /media/user/data0/laser_calib_2024_05_02_tutorial/calib/results/calibration_rig/


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
