#!/usr/bin/env python

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
import math

from inputs import get_gamepad

global arm
arm = False


def joy_callback(joy):
    global arm
    arm = joy.buttons[1]

def twist_callback(msg):
    
    ack_msg = AckermannDriveStamped()

    ack_msg.drive.speed = msg.linear.x
    ack_msg.drive.steering_angle = msg.angular.z / 2
    
    global arm
    if arm:
        ack_pub.publish(ack_msg)
        
    else:
        ack_pub.publish(AckermannDriveStamped())
        
 
if __name__ == "__main__":

    rospy.init_node("twist_to_ackermann")

    twist_sub = rospy.Subscriber("/cmd_vel", Twist, twist_callback, queue_size=10)
    joy_sub = rospy.Subscriber("/joy", Joy, joy_callback, queue_size=10)

    ack_pub = rospy.Publisher("/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=10)

    rospy.spin()
