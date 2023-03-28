#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int8MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
import Astar
import Main

x = 0
y = 0
theta = 0
path=[]
flag =0
def callback1(message):
    global x
    global y
    global theta
    x = message.x
    y = message.y
    theta = message.theta
    #rospy.loginfo("update_path recieved: " + str(message))


def callback2(message):
    x_target = message.x
    y_target = message.y
    map_matrix = Main.get_map()
    start = int(x),int(y)
    end = int(x_target), int(y_target)
    global path
    global flag
    path = Astar.astar_diagonal(map_matrix,start,end)
    flag = 1
    #rospy.loginfo("update_path recieved: "+ str(message))
    #rospy.loginfo("update_path calculated: "+ str(path))


    
    
    
    
    

    
def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('planner', anonymous=True)    
    rospy.Subscriber("/pose", Pose2D, callback1)
    rospy.Subscriber("/target", Pose2D, callback2)
    pub = rospy.Publisher("/path", Int8MultiArray, queue_size=10)

    # spin() simply keeps python from exiting until this node is stopped
    r = rospy.Rate(0.2)
    while not rospy.is_shutdown():
        global flag
        global path
        if flag == 1:

            path_msg = Int8MultiArray()
            rows,cols = len(path), len(path[0])
            layout = MultiArrayLayout()
            dim1 = MultiArrayDimension()
            dim2 = MultiArrayDimension()
            dim1.label = "length"
            dim1.size = rows
            dim1.stride = rows*cols
            dim2.label = "points"
            dim2.size = cols
            dim2.stride = cols
            dim = []
            dim.append(dim1)
            dim.append(dim2)
            layout.dim = dim
            layout.data_offset = 2

            path_msg.layout = layout
            path_data=[item for sublist in path for item in sublist]
            path_msg.data = path_data
            #rospy.loginfo(type(path))
            #path_msg.data = path
            #path_msg.layout = layout
            pub.publish(path_msg)
            flag = 0
            r.sleep()

if __name__ == '__main__':
    listener()