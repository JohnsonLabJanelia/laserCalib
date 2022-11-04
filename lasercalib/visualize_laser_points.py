import cv2 
import numpy as np

from skimage import morphology
from skimage import measure



video_file = "/home/red/Calibration/laser_calib_4_cams/laser_movies/Cam0.mp4"
cap = cv2.VideoCapture(video_file)


while(cap.isOpened()):
    ret, frame_original = cap.read() # frame shape (2200, 3208, 3)    


    # rob's green laser point detector 
    green = frame_original[:,:,1]
    cc = green > 80
    cc = morphology.binary_erosion(cc,  morphology.disk(1))
    cc = morphology.binary_dilation(cc, morphology.disk(1))
    cc = morphology.binary_closing(cc, morphology.disk(4))

    ## roi 
    mask_roi = np.zeros_like(green)
    mask_roi = cv2.circle(mask_roi, (1604,1100), 1000, 255, -1)

    cc = mask_roi * cc

    labels = measure.label(cc, background=0, return_num=False)
    props = measure.regionprops(labels)

    idx = []
    for i in range(len(props)):
        dist = ((props[i].centroid[0] - 1100)**2 + (props[i].centroid[1] - 1604)**2)**0.5
        if dist > 1000:
            continue
        idx.append(i)
        
        
    if len(idx) == 1:
        frame_original = cv2.circle(frame_original, (int(props[0].centroid[1]), int(props[0].centroid[0])), 20, (0, 255, 0), 2)



    frame_ds = cv2.resize(frame_original, (802, 550))    


    cv2.imshow('frame', frame_ds)


    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break