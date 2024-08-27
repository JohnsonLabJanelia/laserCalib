## Calibration steps

1. Extract laser points
```
python detect_laser_points.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/ -i 0
python detect_laser_points.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/ -i 1
```

2. infer 3d points
python get_points3d.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/

3. run calibration
python calibrate_camera.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/

4. run viewers to extract aruco markers, press `q` to exit.  
python run_viewers.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/ -m aruco

5. triangulate aurco markers
python triangulate_aruco.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/

6. register to the world coordinate
python register_world.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/

7. To check on the results
python verify_world.py -c /media/user/data0/laser_calib_2024_05_02_tutorial/calib/

Final reasults in /media/user/data0/laser_calib_2024_05_02_tutorial/calib/results/calibration_rig/
