#!/usr/bin/env python

import rospy, math
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Joy


global arm
arm = False

def joy_callback(joy):
    global arm
    
    arm = True if joy.axes[5] == -1 else False


def convert_trans_rot_vel_to_steering_angle(v, omega, wheelbase):
    if omega == 0 or v == 0:
        return 0

    radius = v / omega
    return math.atan(wheelbase / radius)


def cmd_callback(data):
    global wheelbase
    global ackermann_cmd_topic
    global frame_id
    global pub

    v = data.linear.x
    #steering = convert_trans_rot_vel_to_steering_angle(v, data.angular.z, wheelbase)
    steering = data.angular.z
    msg = AckermannDriveStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.drive.steering_angle = steering
    msg.drive.speed = max(v, 0.45) if (v>0) else -0.3
    print(msg.drive.speed)

    global arm
    if arm:
        ack_pub.publish(msg)

    else:
        ack_pub.publish(AckermannDriveStamped())



if __name__ == '__main__': 
    try:
        rospy.init_node('cmd_vel_to_ackermann_drive')
            
        twist_cmd_topic = rospy.get_param('~twist_cmd_topic', '/cmd_vel') 
        ackermann_cmd_topic = "/ackermann_cmd_mux/input/navigation"
        wheelbase = rospy.get_param('~wheelbase', 1.0)
        frame_id = rospy.get_param('~frame_id', 'odom')


        rospy.Subscriber("/joy", Joy, joy_callback, queue_size=10)
        rospy.Subscriber(twist_cmd_topic, Twist, cmd_callback, queue_size=1)
        ack_pub = rospy.Publisher(ackermann_cmd_topic, AckermannDriveStamped, queue_size=1)

        rospy.loginfo("Node 'cmd_vel_to_ackermann_drive' started.\nListening to %s, publishing to %s. Frame id: %s, wheelbase: %f", "/cmd_vel", ackermann_cmd_topic, frame_id, wheelbase)

        rospy.spin()

    except rospy.ROSInterruptException:
        pass

