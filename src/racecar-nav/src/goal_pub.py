#!/usr/bin/env python

import rospy
from move_base_msgs.msg import MoveBaseActionGoal
from nav_msgs.msg import Path
import math
import rospkg
# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()
# get the file path for rospy_tutorials
pkg_path = rospack.get_path("lane_detection")
import sys
sys.path.insert(0, pkg_path + "/src")

from driver import DriverNode

def goal_callback(msg):
    action_goal = MoveBaseActionGoal()

    action_goal.header.frame_id = "base_link"
    action_goal.header.stamp = rospy.Time.now()

    action_goal.goal_id.id = str(msg.header.seq)
    action_goal.goal_id.stamp = rospy.Time.now()

    action_goal.goal.target_pose.header.frame_id = "base_link"
    action_goal.goal.target_pose.header.stamp = rospy.Time.now()

    action_goal.goal.target_pose.pose.position.x = msg.poses[0].pose.position.x / 100 *2
    #action_goal.goal.target_pose.pose.position.x = 1.0
    action_goal.goal.target_pose.pose.position.y = -msg.poses[0].pose.position.y / 100

    
    action_goal.goal.target_pose.pose.orientation.x = msg.poses[0].pose.orientation.x
    action_goal.goal.target_pose.pose.orientation.y = msg.poses[0].pose.orientation.y
    action_goal.goal.target_pose.pose.orientation.z = msg.poses[0].pose.orientation.z
    action_goal.goal.target_pose.pose.orientation.w = msg.poses[0].pose.orientation.w
    """
    action_goal.goal.target_pose.pose.orientation.x = 0
    action_goal.goal.target_pose.pose.orientation.y = 0
    action_goal.goal.target_pose.pose.orientation.z = 0
    action_goal.goal.target_pose.pose.orientation.w = 1
    """

    goal_pub.publish(action_goal)

 
if __name__ == "__main__":

    rospy.init_node("goal_publisher")

    goal_pub = rospy.Publisher("/move_base/goal", MoveBaseActionGoal, queue_size=10)
    rospy.Subscriber("/lane_detection/path", Path, goal_callback, queue_size=10)

    #sender = DriverNode(cmd_type="cmd")
    #sender.Pid.Kp = 1.0

    rospy.spin()
