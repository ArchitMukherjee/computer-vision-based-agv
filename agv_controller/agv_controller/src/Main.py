from Aruco import aruco
import Object
import Node
from time import sleep
from matplotlib import pyplot as plt
from astar import map
'''
def get_data():
    xmat,ymat,r0,mat,img = Aruco.get_pose()
    ob_mat,img = Object.yolo(img,mat,50,xmat,ymat)
    path_mat = Node.search(ob_mat, 1, (ymat,xmat), (13,24))
    if(path_mat[ymat][xmat+1]==1):
        return(ymat,xmat+1,r0)
    elif(path_mat[ymat][xmat-1]==1):
        return(ymat,xmat-1,r0)
    elif(path_mat[ymat+1][xmat]==1):
        return(ymat+1,xmat,r0)
    elif(path_mat[ymat-1][xmat]==1):
        return(ymat+1,xmat,r0)
    
    elif(path_mat[ymat-1][xmat-1]==1):
        return(ymat-1,xmat-1,r0)
    elif(path_mat[ymat-1][xmat+1]==1):
        return(ymat-1,xmat+1,r0)
    elif(path_mat[ymat+1][xmat-1]==1):
        return(ymat+1,xmat-1,r0)
    elif(path_mat[ymat+1][xmat+1]==1):
        return(ymat+1,xmat+1,r0)


    else:
        return(0,0,0)
'''
def get_current_pose(object):
    xmat,ymat,theta,mat,img = object.get_pose()
    return(xmat,ymat,theta,mat)

# def get_matrix(object):
#     _,_,_,mat,img = object.get_pose()
#     return(mat)

def get_map(object):
    xmat,ymat,theta,mat,img = object.get_pose()
    mat,img = Object.yolo(img,mat,50,xmat,ymat)
    return (mat, img)


obj = aruco(0)
mat, img = get_map(obj)
plt.imshow(mat)
plt.show()
plt.imshow(img)
plt.show()
x,y,_,_ = get_current_pose(obj)
start = (x,y)
print(f"Start: {start}")
end = (7,13)
mat[y][x] = 0
h, w = mat.shape
mp = map(mat, (w, h))
path = mp.astar(start, end)
res = path[len(path)-1].copy()
result = []
t = res.copy()
while True:
    result.append(t[0])
    if t[0] == start:
        break
    t = t[4]

for i in result:
    mat[i[0]][i[1]] = 3

plt.imshow(mat)
plt.show()


# obj = aruco(0)
# for i in range(200):
#     rad = get_current_pose(obj)
#     #rad = rad*180/3.14 
#     print(rad)
#     sleep(0.1)

# del obj



