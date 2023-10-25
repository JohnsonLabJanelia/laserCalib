from skimage import morphology
from skimage import measure
import cv2
import numpy as np

def green_laser_finder(img, 
            laser_intensity_thresh=70, 
            centroid_dist_thresh=1100, 
            small_footprint=morphology.disk(1), 
            big_footprint=morphology.disk(4)):
    
    green = img[:,:,1]
    cc = green > laser_intensity_thresh

    # cc = morphology.binary_erosion(cc, small_footprint)
    cc = morphology.binary_dilation(cc, small_footprint)
    cc = morphology.binary_closing(cc, big_footprint)

    ## roi 
    mask_roi = np.zeros_like(green)
    mask_roi = cv2.circle(mask_roi, (1604,1100), 1100, 255, -1)
    # cc = mask_roi * cc

    labels = measure.label(cc, background=0, return_num=False)
    props = measure.regionprops(labels)

    # exclude connected components that are too far from image center
    n = len(props)
    idx = []
    
    for i in range(n):
        # dist = ((props[i].centroid[0] - 1100)**2 + (props[i].centroid[1] - 1604)**2)**0.5
        # if dist > centroid_dist_thresh:
        #     continue            
        idx.append(i)

    if len(idx) == 1:
        return props[idx[0]].centroid
    else:
        return None
