#!/usr/bin/env python
import rospy
from std_msgs.msg import Int8MultiArray
from geometry_msgs.msg import Pose2D

def callback(message):
    rospy.loginfo(message.data)
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('controller', anonymous=True)

    rospy.Subscriber("/path", Int8MultiArray, callback)
    #rospy.Subscriber("/pose", Pose2D, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()