

1. Extract laser points
```
python detect_laser_points.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/ -i 0
python detect_laser_points.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/ -i 1
```

2. infer 3d points
python get_points3d.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/

3. run calibration
python calibrate_camera.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/

4. run viewers to extract aruco markers 
python run_viewers.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/ -m aruco

Press `q` to exit. 

5. triangulate aurco markers
python triangulate_aruco.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/

6. transform to the world coordinate
python transform_world.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/

Final reasults in root_dir/calib/results/calibration_rig/