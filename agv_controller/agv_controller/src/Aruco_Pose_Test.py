import numpy as np
import time
import cv2 as cv
import matplotlib.pyplot as plt
import os
import time
import math

# ARUCO_DICT = {
# 	"DICT_4X4_50": cv.aruco.DICT_4X4_50,
# 	"DICT_4X4_100": cv.aruco.DICT_4X4_100,
# 	"DICT_4X4_250": cv.aruco.DICT_4X4_250,
# 	"DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
# 	"DICT_5X5_50": cv.aruco.DICT_5X5_50,
# 	"DICT_5X5_100": cv.aruco.DICT_5X5_100,
# 	"DICT_5X5_250": cv.aruco.DICT_5X5_250,
# 	"DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
# 	"DICT_6X6_50": cv.aruco.DICT_6X6_50,
# 	"DICT_6X6_100": cv.aruco.DICT_6X6_100,
# 	"DICT_6X6_250": cv.aruco.DICT_6X6_250,
# 	"DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
# 	"DICT_7X7_50": cv.aruco.DICT_7X7_50,
# 	"DICT_7X7_100": cv.aruco.DICT_7X7_100,
# 	"DICT_7X7_250": cv.aruco.DICT_7X7_250,
# 	"DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
# 	"DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
# 	"DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
# 	"DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
# 	"DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
# 	"DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
# }


class aruco:
    
    def __init__(self):
        # try:
        #     #self.cap = cv.VideoCapture(cam)
        # except:
            #raise Exception("Cannot Create Multiple Instances with the same camera")
        self.arucotype = cv.aruco.DICT_4X4_250
        self.arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
        self.arucoParams = cv.aruco.DetectorParameters()
        #self.arucoParams.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        self.arucoParams.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR
        self.intrinsic_camera = np.array(((933.15867, 0, 657.59),(0,933.1586, 400.36993),(0,0,1)))
        self.distortion = np.array((-0.43948,0.18514,0,0))
        # self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    
    def __del__(self):
        #self.cap.release()
        cv.destroyAllWindows()


    def __Rmatrix(self,rvec):
        [dst,jacobian] = cv.Rodrigues(rvec)
        return dst

    # Checks if a matrix is a valid rotation matrix.
    def __isRotationMatrix(self,R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6
    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def __rotationMatrixToEulerAngles(self,R) :
        assert(self.__isRotationMatrix(R))
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


    def __pose_estimation(self, frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        cv.aruco_dict = cv.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv.aruco.DetectorParameters()
        
        corners, ids, rejected_img_points = cv.aruco.detectMarkers(gray, cv.aruco_dict,parameters=parameters)
        #print(np.array(corners[0]).shape)
        theta = 400.0
        rvec = [[[0,0,0]]]
        if len(corners) > 0:
            for i in range(0, len(ids)):
                #rvec, tvec, markerPoints = cv.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)
                # for x in corners[i]:
                #     # np.append(list(x[0]),tvec[0][0][2])
                #     # np.append(x[1],tvec[0][0][2])
                #     # np.append(x[2],tvec[0][0][2])
                #     # np.append(x[3],tvec[0][0][2])
                #     # np.append(object_point,x)
                #     #list(x[0]).append(tvec[0][0][2])
                #     print(x[0].shape)
                # print(object_point)
                #ext=[[[[corners[0][0][0][0]+corners[0][0][1][0]/2,corners[0][0][0][1]+corners[0][0][1][1]/2],  [corners[0][0][2][0]+corners[0][0][3][0]/2,corners[0][0][2][1]+corners[0][0][3][1]/2]]]]
                #print(np.array(ext).shape)

                # corners=np.insert(corners[0][0], 3, ext[0][0], axis=0)
                # corners=np.array([[corners]])
                #print(np.array(corners).shape)
                marker_size=250
                model_points = np.array([
                              [-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
                # Camera internals
                size=frame.shape
                focal_length = size[1]
                center = (size[1]/2, size[0]/2)
                camera_matrix = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]],[0, 0, 1]], dtype = "double")
                success,rvec, tvec = cv.solvePnP(model_points, corners[i], camera_matrix,distortion_coefficients,False, cv.SOLVEPNP_IPPE_SQUARE)
                # print(rvec[0][0][2]*180/math.pi)
                #print(rvec*180/math.pi)
                # theta = theta*180/math.pi
                # theta = (math.pi/2)-theta
                R=self.__Rmatrix(np.array(rvec))
                theta = self.__rotationMatrixToEulerAngles(R)
                theta = theta*180/math.pi
                # theta = (math.pi/2)-theta   
                # theta = theta * 1.035525
                #if theta > math.pi :

                cv.aruco.drawDetectedMarkers(frame, corners) 

                cv.drawFrameAxes(frame, camera_matrix, distortion_coefficients, rvec, tvec, 1) 




                #print ("Rotation Vector:\n {0}".format(rvec[2]*180/math.pi))
                #print ("Translation Vector:\n {0}".format(tvec))
                        # if theta == 400.0:
                        #     print("Aruco Not Found")
                        #     return frame, theta
                        # else:
                
                return frame, theta

    def __aruco_display(self, corners, ids, rejected, image):

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
                mat[y,x] = 2
        return image, x, y, mat

    def get_pose(self,img):

        #if self.cap.isOpened():
            #ret, img = self.cap.read()
            img = img[:, 50:800]
            # gamma = 0.3
            # img = np.array(255*(img/255)**gamma)
            # img = np.var(img)
            h, w, _ = img.shape
            temp= self.__pose_estimation(img, self.arucotype, self.intrinsic_camera, self.distortion)
            if temp== None:
                output, theta=404,(404,404,404)
                print("Aruco Not Found")
            else:
                output,theta=temp
            #print(theta)
            corners, ids, rejected = cv.aruco.detectMarkers(img, self.arucoDict, parameters=self.arucoParams)
            detected_markers_output, xmat, ymat, mat = self.__aruco_display(corners, ids, rejected, img)
            #print (theta)
            cv.imshow("Video", detected_markers_output)
            #key = cv.waitKey(1) & 0xFF
            #if key == ord('q'):
            #    break
        # else:
        #     raise Exception("Camera Not Initailized")
        
        #result = (xmat,ymat,theta,mat,img)
            # if len(result < 1:
            #     print("Invalid")
            #     #result = (0,0,0,0,0)
            #     result=404
            return theta[1]
cap = cv.VideoCapture(0)
ap = aruco()
prev_time=0
prev_theta=0
max_theta=-1000
min_theta=1000
while True:
    ret, img = cap.read()
    delT=time.time()-prev_time
    delTheta=ap.get_pose(img)-prev_theta
    prev_time=time.time() 

    prev_theta=ap.get_pose(img)

    max_theta=max(ap.get_pose(img),max_theta)
    min_theta=min(ap.get_pose(img),min_theta)
    if ap.get_pose(img)!=404:
     change=delTheta/(delT*200)
     print(change)
     #print(ap.get_pose(img))
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
            break
print(max_theta)
print(min_theta)
del ap
