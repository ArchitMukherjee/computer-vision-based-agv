#!/usr/bin/env python
import rospy
from std_msgs.msg import Int8MultiArray
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Twist
import math

x = 0
y = 0
theta = 0
path = []
flag = 0

def callback1(message):
    global path
    global flag
    final_path=[]
    path_list = message.data
    rospy.loginfo("Series"+str(len(path_list)))
    for x in range(len(path_list)//2):
        final_path.append([path_list[2*x],path_list[2*x+1]])
    path = final_path
    rospy.loginfo("In /path callback: " + str(path))
    flag = 1

def callback2(message):
    global x
    global y
    global theta

    x = message.x
    y = message.y
    theta = message.theta
    
  
def listener():
    
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('controller', anonymous=True)

    rospy.Subscriber("/path", Int8MultiArray, callback1)
    rospy.Subscriber("/pose", Pose2D, callback2)
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    speed = Twist()
    
    r = rospy.Rate(0.2)  
    
    while not rospy.is_shutdown():
        global path
        global x
        global y
        global theta
        global flag
        if flag == 1:
            iter=0
            rospy.loginfo("In while loop: "+str(path))
            x_target,y_target = path[iter][0],path[iter][1]
            while((x,y != x_target,y_target)or(not rospy.is_shutdown())):
                inc_x = x_target -x
                inc_y = y_target -y

                angle_to_goal = math.atan2(inc_y, inc_x)

                if abs(angle_to_goal - theta) > 0.1:
                    speed.linear.x = 0.0
                    speed.angular.z = 0.3
                elif abs(math.sqrt((x_target-x)**2 + (y_target-y)**2)<0.2):
                    speed.linear.x = 0.0
                    speed.linear.y = 0.0
                    speed.angular.z = 0.0
                else:
                    speed.linear.x = 0.2
                    speed.angular.z = 0.0

                pub.publish(speed)
                flag = 0        
            iter = iter + 1

        r.sleep()



if __name__ == '__main__':
    listener()