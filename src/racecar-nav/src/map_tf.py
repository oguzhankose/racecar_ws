#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist


class TransformToMap():
    def __init__(self, name, odom_topic, imu_topic):
        try:
            self.source_name = name
            
            self.timeout = 0.5


            self.source_frame = rospy.wait_for_message(odom_topic, Odometry, self.timeout).header.frame_id
            self.target_frame = "map"

            self.pub = rospy.Publisher("/map", Odometry, queue_size=10)
            self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=10)
            self.imu_sub = rospy.Subscriber(imu_topic, Imu, self.imu_callback, queue_size=10)
            

        except Exception as e:
            print(e)
            

    def imu_callback(self, msg):

        odom = Odometry()

        odom.header.stamp = rospy.Time.now() 
        odom.header.frame_id = "odom"

        # set the position
        odom.pose.pose = Pose()
        odom.pose.pose.position.x = 0
        odom.pose.pose.position.y = 0
        odom.pose.pose.position.z = 0

        odom.pose.pose.orientation.x = msg.orientation.x
        odom.pose.pose.orientation.y = msg.orientation.y
        odom.pose.pose.orientation.z = msg.orientation.z
        odom.pose.pose.orientation.w = msg.orientation.w

        # set the velocity
        odom.child_frame_id = "base_link"
        odom.twist.twist = Twist()

        self.pub.publish(odom)

    def odom_callback(self, msg):
        pass





if __name__ == '__main__':

    rospy.init_node('map_tf')

    TransformToMap(name="base_link", odom_topic="/odom", imu_topic="/imu/data")
    
    rospy.spin()