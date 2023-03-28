import Aruco
import Object
import Node
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
def get_current_pose():
    xmat,ymat,theta,mat,img = Aruco.get_pose()
    return(xmat,ymat,theta)

def get_matrix():
    _,_,_,mat,img = Aruco.get_pose()
    return(mat)

def get_map():
    xmat,ymat,theta,mat,img = Aruco.get_pose()
    mat,img = Object.yolo(img,mat,50,xmat,ymat)
    return mat

