import numpy as np
import time
import cv2 as cv
import matplotlib.pyplot as plt


def yolo(img,mat,grid,cX_mat,cY_mat):
  thres = 0.2
  nms_threshold=0.01
  #img=cv.imread('tarp1.jpeg')
  classNames= []
  classFile = 'coco.names'
  with open(classFile,'rt') as f:
   classNames = f.read().rstrip('\n').split('\n')
  configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
  weightsPath = 'frozen_inference_graph.pb'
  net = cv.dnn_DetectionModel(weightsPath,configPath)
  net.setInputSize(320,320)
  net.setInputScale(1.0/ 127.5)
  net.setInputMean((127.5, 127.5, 127.5))
  net.setInputSwapRB(True)
  classIds, confs, bbox = net.detect(img,confThreshold=thres)
  bbox=list(bbox)
  confs=list(np.array(confs).reshape(1,-1)[0])
  confs=list(map(float,confs))
  #print(type(confs[0]))
  #print(classIds,bbox)
  indices=cv.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
  for i in indices:
      box=bbox[i]
      x,y,w,h=box[0],box[1],box[2],box[3]
      cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,0),thickness=5)
      x_mat,y_mat,w_mat,h_mat = x//grid,y//grid,w//grid,h//grid
      if ((x_mat>cX_mat or cX_mat>x_mat+w_mat)or(y_mat>cY_mat or cY_mat>y_mat+h_mat)):
         print(x//grid,y//grid,w//grid,h//grid)
         mat[(y//grid):((y+h)//grid),(x//grid):((x+w)//grid)] = 1
  #if len(classIds) != 0:
  #  for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
  #    cv.rectangle(img,box,color=(0,0,0),thickness=20)
  #    cv.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
  #    cv.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),cv.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
  #plt.imshow(img)
  #plt.show()
  return img,mat


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


def aruco_display(corners, ids, rejected, image):

    h, w, _ = image.shape #y,x = h,w
    grids = 50
    for i in range(0,h,grids):
        cv.line(image, (0,i), (w,i), (0, 0, 0), 1)
    for i in range(0,w,grids):
        cv.line(image, (i,0), (i,h), (0, 0, 0), 1)
    
    #print("No Rows: {r} \nNo Cols: {c}".format(r=int(h/50), c=int(w/50)))
    r = int(h/grids)
    c = int(w/grids)
    mat = np.zeros([r,c])

    

    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            
            cv.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            cv.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(markerID))
            print("Centre coordinate: {x},{y}".format(x=cX//grids,y=cY//grids))

            #mat[cY//grids, cX//grids] = 5
            #mat[10,1] = 5;
            image,mat = yolo(image,mat,grids,cX//grids,cY//grids)
            #plt.imshow(mat)
            #plt.show()
            # start = (cY//grids, cX//grids)
            # end = (10,18)
            # path = np.array(Node.search(mat,1,start,end))
            # path_mat = path+1+mat
            #plt.imshow(path_mat)
            #plt.show()
            
            #print(mat)
    
    return image

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.aruco_dict = cv.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv.aruco.DetectorParameters()


    corners, ids, rejected_img_points = cv.aruco.detectMarkers(gray, cv.aruco_dict,parameters=parameters)

        
    if len(corners) > 0:
        for i in range(0, len(ids)):
           
            rvec, tvec, markerPoints = cv.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                       distortion_coefficients)
            
            cv.aruco.drawDetectedMarkers(frame, corners) 

            cv.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  

    return frame


intrinsic_camera = np.array(((933.15867, 0, 657.59),(0,933.1586, 400.36993),(0,0,1)))
distortion = np.array((-0.43948,0.18514,0,0))

aruco_type = "DICT_4X4_250"

arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

arucoParams = cv.aruco.DetectorParameters()


cap = cv.VideoCapture(2)
#cap = cv.VideoCapture('tarp_test_video.mp4')
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)


while cap.isOpened():
    ret, img = cap.read()
    
    h, w, _ = img.shape
    
    width = 1000
    height = int(width*(h/w))
    img = cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC)
    corners, ids, rejected = cv.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
    detected_markers = aruco_display(corners, ids, rejected, img)
    pose = pose_estimation(detected_markers, ARUCO_DICT[aruco_type], intrinsic_camera , distortion)
    cv.imshow("Image", pose)
    
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv.destroyAllWindows()
cap.release()








