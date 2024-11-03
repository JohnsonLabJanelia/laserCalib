import numpy as np
import cv2

def probe_monotonicity(K, dist, newcameramtx, image_shape, N=100, M=100):
    
    # calculate the region in which to probe the monotonicity
    pts_undist = np.array([
        [0,0],
        [0,image_shape[0]],
        [image_shape[1],0],
        [image_shape[1], image_shape[0]]
    ])
    pts_norm = (pts_undist-newcameramtx[[0,1],[2,2]][None])/newcameramtx[[0,1],[0,1]][None]

    xmin, ymin = pts_norm.min(0)
    xmax, ymax = pts_norm.max(0)
    r_max = np.sqrt(xmax**2+ymax**2)

    # create points used to compute the sign after distortion
    alphas = np.linspace(0,np.pi/2, N//4+2)[1:-1]
    alphas = np.concatenate([alphas, alphas+np.pi/2, alphas+np.pi, alphas+np.pi*3/2])
    
    ds = r_max/M

    ptss = []
    sign = []
    for r in np.linspace(0, r_max, M):
        pts= np.vstack([r*np.cos(alphas), r*np.sin(alphas)]).T
        ptsp = np.vstack([(r+ds)*np.cos(alphas), (r+ds)*np.sin(alphas)]).T

        mask1 = np.logical_and(pts[:,0]>=xmin, pts[:,0]<xmax)
        mask2 = np.logical_and(pts[:,1]>=ymin, pts[:,1]<ymax)
        mask = np.logical_and(mask1, mask2)

        if np.all(mask==False):
            continue

        pts, ptsp = pts[mask],ptsp[mask]

        ptss.append((pts,ptsp))
        sign.append(np.sign(pts-ptsp))
        
    # distort the points
    grid, gridp = zip(*ptss)
    grid, gridp = np.vstack(grid), np.vstack(gridp)

    grid_ = np.hstack([grid, np.zeros((len(grid),1))])
    gridp_ = np.hstack([gridp, np.zeros((len(gridp),1))])

    proj1 = cv2.projectPoints(grid_, np.eye(3), np.zeros(3), np.eye(3), dist)[0].reshape(-1,2)
    proj2 = cv2.projectPoints(gridp_, np.eye(3), np.zeros(3), np.eye(3), dist)[0].reshape(-1,2)

    # probe 
    is_monotonic = np.sign(proj1-proj2)==np.vstack(sign)
    is_monotonic = np.logical_and(*is_monotonic.T)
    
    return grid, is_monotonic