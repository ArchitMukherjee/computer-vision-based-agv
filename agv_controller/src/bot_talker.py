#!/usr/bin/env python
# license removed for brevity
import Main
import rospy
from std_msgs.msg import String
import time

def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(1) # 1hz
    #time.sleep(5)
    x_bot,y_bot,theta=Main.get_data()
    while not rospy.is_shutdown():
        hello_str = "Bot Talked: target_x="+ str(x_bot) + " target_y=" + str(y_bot) + " theta=" + str(theta)
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass