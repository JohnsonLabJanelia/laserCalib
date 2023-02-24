import cv2 
import numpy as np

root_dir  = "/home/jinyao/Calibration/aruco_detection"
video_file = root_dir + "/Cam0.mp4"
cap = cv2.VideoCapture(video_file)


fs = cv2.FileStorage("/home/jinyao/Calibration/newrig8/results/calibration_aruco/Cam0.yaml", cv2.FILE_STORAGE_READ)
m_matrix_coefficients = fs.getNode("intrinsicMatrix")
m_distortion_coefficients = fs.getNode("distortionCoefficients")

matrix_coefficients = m_matrix_coefficients.mat().T
distortion_coefficients = m_distortion_coefficients.mat()[0]



while(cap.isOpened()):
    ret, frame = cap.read() # frame shape (2200, 3208, 3)    
    frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
    img_height, img_width, _ = frame.shape

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    aruco_params = cv2.aruco.DetectorParameters_create() 
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame_gray, aruco_dict, parameters=aruco_params)

    len(corners)

    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # pose estimation
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorner, 100, matrix_coefficients,
                                                                           distortion_coefficients)
            

            # extract the marker corners (which are always returned
            # in top-left, top-right, bottom-right, and bottom-left
            # order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the
            # ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the frame
            cv2.putText(frame, str(markerID),
                (topLeft[0], topLeft[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)


    #import pdb; pdb.set_trace()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("e"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
