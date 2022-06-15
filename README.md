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
*This is a work in progress*  
1. Collect images  
- Use syncrhonized cameras. Turn off all the lights in the rig. This code is currently written for calibrating color cameras. If you are using monochrome cameras, please adjust the centroid-finding code to accept single channel images. If using color cameras, please use a green laser pointer. If you'd rather use a blue or red laser pointer, please adjust the centroid-finding code to use the red or blue image channel (right now it is using the green channel to find the green laser pointer centroid in each image). Use short (~100-500 microsecond) camera exposures and low gain such that the only bright spot in the image is from the laser pointer. The goal is to collect several thousand images on each camera of the laser pointer spot. You can shine the spot onto the floor of the arena (see example below).  
2. Place all the images from each camera in it's own directory. Place all these camera directories in a parent directory.  
3. Run the `find_laser_points.py` script to extract centroids from these images and save them to a `.pkl` file  
5. Run `calibrate.py` to calibrate the cameras (edit to adjust for cam number, cam param initialization, and 3D point initialization.  
- In our example, we have 7 cameras. We used a previous calibration to initialize the camera parameters for each camera. It is probably important to initialize these camera parameters in the ballpark of the expected output. These calibration parameters could be estimated from a previous calibration or from a drawing of the rig. We haven't tested this code for a situation in which no initial guess of the camera parameters is provided.   
- You will also need to provide an initial guess of the 3D world coordinates for each laser pointer observation. We simply used the pixel location of the extracted laser pointer centroids from one of our cameras (the central ceiling camera; blue camera in image below).  
6. Run the `save_cam_params.py` to save out your calibration parameters as a `.csv` file.   

## Example of a suitable input image  
This is a suitable image. The green laser pointer is the brightest and largest green spot in the image. Good job.   
![suitable_input_image](README_images/suitable_input_image.png)  


## Example of an unsuitable input image  
In the image below, note that the Universal Robots tablets on the right side have green power buttons that are visible in the image. This can be a problem for easily locating the green laser pointer spot. In this release, the laser-pointer detection algorithm implements a distance threshold such that any contours found more than 1500 pixels away from the image center are excluded. To map the entire camera space, please take suitable images and remove this distance constraint from the code (see centroid_extraction.py).  
![unsuitable_input_image](README_images/unsuitable_input_image.png)  


## Example of calibrated cameras and extracted laser points in world space  
![laser_points_and_cam_positions](README_images/laser_points_and_cam_positions.png)  


## TODO  
- [ ] FFmpeg decoder  
- [ ] about 20 minutes to run the centroid extraction, can be parallelized  

## Reference  
The library is built on pySBA -- Python Bundle Adjustment library: https://github.com/jahdiel/pySBA  

Camera visualizer -- https://github.com/demul/extrinsic2pyramid  

Scipy bundle adjustment cookbook: http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html  

FFmpeg python wrapper: https://github.com/kkroening/ffmpeg-python  

## Acknowledgements  
We thank he authors of pyBSA (https://github.com/jahdiel), the Scipy Cookbook, and the Camera Visualizer Tool (https://github.com/demul) used in this project. We adapted their code (see Reference Section). We thank Selmaan Chettih (https://twitter.com/selmaanchettih) for advice and help in getting this project started with pyBSA. We thank David Schauder (https://github.com/porterbot) and Ratan Othayoth (https://github.com/othayoth) for help in synchronizing cameras.  
