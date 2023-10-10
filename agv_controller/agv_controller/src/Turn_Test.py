#!/usr/bin/env python

import rospy
import Aruco_Pose_Test as apr
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2  

ap=apr.aruco()
bridge = CvBridge()

def callback(message):
    try:
        # Convert your ROS Image message to OpenCV2
        global img
        global ap
        img = bridge.imgmsg_to_cv2(message, "bgr8")
    except CvBridgeError as e:
        print(e)
        del ap
def talker():
    # global x
    # global y
    # global z
    global curr_theta
    speed=Twist()
    global prev_time
    global detT
    global prev_theta
    global delTheta
    global curr_time


    prev_time=0
    prev_theta=0

    rospy.Subscriber("cam_feed", Image, callback)
    pub = rospy.Publisher('\cmd_vel',Twist, queue_size=10)
    rospy.init_node('orient', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
            
            # if theta!=404:
            #   if theta<0:
                
                curr_theta=ap.get_pose(img)
                curr_time=rospy.get_time()
                if curr_theta!=404:
                 delT=curr_time-prev_time

                 ########
                 delTheta=curr_theta-prev_theta
                 ########
                 prev_time=curr_time

                 prev_theta=curr_theta
                 change=delTheta/(delT*200)
                 if change>-1 and change <1:
                    speed.angular.x=0
                    speed.angular.y=0
                    speed.angular.z=change
                    rate.sleep()
                    speed.angular.x=0
                    speed.angular.y=0
                    speed.angular.z=0
                 else:
                    i=int(change)
                    if change>0:    
                        speed.angular.x=0
                        speed.angular.y=0
                        speed.angular.z=change-i
                        rate.sleep()
                        speed.angular.x=0
                        speed.angular.y=0
                        speed.angular.z=0
                    else:
                        speed.angular.x=0
                        speed.angular.y=0
                        speed.angular.z=abs(i)+change
                        rate.sleep()
                        speed.angular.x=0
                        speed.angular.y=0
                        speed.angular.z=0
                 pub.publish(speed)
    rate.sleep()
 
if __name__ == '__main__':
    try:
     talker()
    except rospy.ROSInterruptException:
     pass