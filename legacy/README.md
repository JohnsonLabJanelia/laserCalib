1. Run the `find_laser_points.py` script to extract centroids from these images and save them to a `/results/*.pkl` file. For instance: 
```
python find_laser_points.py --root_dir=/home/user/Calibration/16cam --frame_range 0 5000
```
<em>New faster implementation</em>.
```
python extract_laser_points.py --root_dir=/media/user/data0/laser_calib_2024_06_20/2024_06_20_17_12_14/ --frame_range 0 1500
```
2. Run `prepare_points3d.py` to prepare datasets for calibration. It will use one camera initial parameters, and ground truth z plane value to infer points 3d locations. Please select a camera that doesn't move much. 
```
python prepare_points3d.py -f /media/user/data0/laser_calibrate_2024_4_1/2024_04_01_17_24_00/ /media/user/data0/laser_calibrate_2024_4_1/2024_04_01_17_09_58/ -z 105 0 -o /media/user/data0/laser_calibrate_2024_4_1/2024_04_01_17_09_58/ -n Cam710038
```
Note: 
- It is probably important to initialize these camera parameters in the ballpark of the expected output. These calibration parameters could be estimated from a previous calibration or from a drawing/digital twin of the rig. To innitialize camera postions from blender model, put the `camera_dicts.pkl` in `results`.  
- You will also need to provide an initial guess of the 3D world coordinates for each laser pointer observation. We simply used the pixel location of the extracted laser pointer centroids from one of our cameras (the central ceiling camera; blue camera in image below).
3. Run `calibrate.py` to calibrate the cameras
```
python calibrate.py --root_dir=/home/user/Calibration/16cam
```
The calibration result is saved in `results/sba.pkl`.  
4. Use the `movie_viewer.py` to run aruco based feature detection. 
```
python movie_viewer.py --n_cams=16 --root_dir=/home/user/Calibration/16cam --mode aruco
```
Press `p` to pause, press `q` to quit, and the aruco corner will be saved. Make sure all the corners are detected. 

5. Run `aruco_triangulate.py` to triangulate corresponding points from aruco marker. It triangulates the centers of aruco markers  and all the corners of the aruco markers. Also, it epirically estiamtes the scaling factor using ground truth side length of aruco marker (side length in millimeter).  
```
python aruco_triangulate.py --n_cams=16 --root_dir=/home/user/Calibration/16cam --side_len=120
```

6. Run `label2world.py` fit a rigid body transformation to change the coordinate to desired world coordinate. 
```
python label2world.py --n_cams=16 --root_dir=/home/user/Calibration/16cam --use_scale=1
```
If use_scale is true, it will use the scale factor estimated from step 7, otherwise it is set to 1 (no scaling).

7. Run `verify_world_transform.py`` to double check the alignment of step 8. 
```
python verify_world_transform.py --n_cams=17 --root_dir=/media/user/data0/laser_calibrate_2024_4_1/2024_04_01_17_09_58/ --side_len=120 --yaml_dir_name=/media/user/data0/laser_calibrate_2024_4_1/2024_04_01_17_09_58/results/rigspace/calibration_rig/
```