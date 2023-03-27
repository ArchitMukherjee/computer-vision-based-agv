import Aruco
import Object
import matplotlib.pyplot as plt
import Node
import numpy as np

x,y,theta,mat,img = Aruco.get_pose()
mat,img = Object.yolo(img,mat,50,x,y)

start = (y, x)
end = (13,24)
path = np.array(Node.search(mat,1,start,end))

path_mat = path+1+mat
print(path_mat)
plt.imshow(path_mat)
plt.show()

