import numpy as np
import time
import cv2 as cv
import matplotlib.pyplot as plt
import os
import math

ARUCO_DICT = {
	"DICT_4X4_50": cv.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
}



def Rmatrix(rvec):
    [dst,jacobian] = cv.Rodrigues(rvec)
    return dst

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])



def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.aruco_dict = cv.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv.aruco.DetectorParameters()
    
    corners, ids, rejected_img_points = cv.aruco.detectMarkers(gray, cv.aruco_dict,parameters=parameters)
    rvec0,rvec1,rvec2 = (0.,0.,0.)

    if len(corners) > 0:
        for i in range(0, len(ids)):
        
            rvec, tvec, markerPoints = cv.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)
            R=Rmatrix(np.array(rvec))
            theta = rotationMatrixToEulerAngles(R)[2]
            theta = theta*180/math.pi
            #cv.aruco.drawDetectedMarkers(frame, corners) 

            #cv.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01) 


    return frame, theta

def aruco_display(corners, ids, rejected, image):

    h, w, _ = image.shape #y,x = h,w
    grids = 50
    r = int(h/grids)
    c = int(w/grids)
    mat = np.zeros([r,c])
    x,y=0,0

    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            
            x = cX//grids
            y = cY//grids
            mat[y,x] = 0
    return image, x, y, mat

def get_pose():
    aruco_type = "DICT_4X4_250"
    arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    arucoParams = cv.aruco.DetectorParameters()
    intrinsic_camera = np.array(((933.15867, 0, 657.59),(0,933.1586, 400.36993),(0,0,1)))
    distortion = np.array((-0.43948,0.18514,0,0))

    #Capture Video
    cap = cv.VideoCapture(os.path.abspath('tarp_test_video.mp4'))
    #cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, img = cap.read()
        h, w, _ = img.shape
        width = 1000
        height = (width*h)//w
        output, theta = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)
        corners, ids, rejected = cv.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
        detected_markers_output, xmat, ymat, mat = aruco_display(corners, ids, rejected, img)
        #print (x,y,rvec1)
        #cv.imshow("Video", detected_markers_output)
        break
        #key = cv.waitKey(1) & 0xFF
        #if key == ord('q'):
        #    break

    cv.destroyAllWindows()
    cap.release()
    return(xmat,ymat,theta,mat,img)