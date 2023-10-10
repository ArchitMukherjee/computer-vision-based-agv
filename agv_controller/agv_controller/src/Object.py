import cv2 as cv
import numpy as np
import os


def yolo(img,mat,grid,cX_mat,cY_mat):
  thres = 0.2
  nms_threshold=0.01
  #img=cv.imread('tarp1.jpeg')
  classNames= []
  classFile = '/home/tashmoy/tarp_ws/src/agv_controller/src/coco.names'
  with open(classFile,'rt') as f:
   classNames = f.read().rstrip('\n').split('\n')
  #f.close()
  configPath = '/home/tashmoy/tarp_ws/src/agv_controller/src/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
  weightsPath = '/home/tashmoy/tarp_ws/src/agv_controller/src/frozen_inference_graph.pb'
  #configPath = open('/home/tashmoy/tarp_ws/src/agv_controller/src/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt','r')
  #weightsPath = open('/home/tashmoy/tarp_ws/src/agv_controller/src/Object.py','r')
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
         #print(x//grid,y//grid,w//grid,h//grid)
        mat[(y//grid)-1:((y+h)//grid)+1,(x//grid)-1:((x+w)//grid)+1] = 1
  #if len(classIds) != 0:
  #  for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
  #    cv.rectangle(img,box,color=(0,0,0),thickness=20)
  #    cv.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
  #    cv.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),cv.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
  #plt.imshow(img)
  #plt.show()
  return mat,img