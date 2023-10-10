#!/usr/bin/env python
# license removed for brevity
import Main
import rospy
from geometry_msgs.msg import Pose2D
import time
from Aruco import aruco
def pose_updater(object):
    pub = rospy.Publisher('/pose', Pose2D, queue_size=10)
    rospy.init_node('updater', anonymous=True)

    pose = Pose2D()
    
    rate = rospy.Rate(10) # 1hz
    #time.sleep(5)
    while not rospy.is_shutdown():
        x_bot,y_bot,theta,mat=Main.get_current_pose(object)
        pose.x = x_bot
        pose.y = y_bot
        pose.theta = theta
        rospy.loginfo(pose)
        pub.publish(pose)
        rate.sleep()
    

if __name__ == '__main__':
    time.sleep(2)
    obj = aruco(2)
    try:
        pose_updater(obj)
    except rospy.ROSInterruptException:
        del obj
    
    del obj