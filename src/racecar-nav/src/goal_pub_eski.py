#!/usr/bin/env python

import rospy
from move_base_msgs.msg import MoveBaseActionGoal
import math



 
if __name__ == "__main__":

    rospy.init_node("goal_publisher")

    goal_pub = rospy.Publisher("/move_base/goal", MoveBaseActionGoal, queue_size=10)

    rate = rospy.Rate(1)

    while not (rospy.is_shutdown()):
        action_goal = MoveBaseActionGoal()

        action_goal.header.frame_id = "base_link"
        action_goal.header.stamp = rospy.Time.now()
        
        action_goal.goal_id.id = "target"
        action_goal.goal_id.stamp = rospy.Time.now()

        action_goal.goal.target_pose.header.frame_id = "base_link"
        action_goal.goal.target_pose.header.stamp = rospy.Time.now()

        action_goal.goal.target_pose.pose.position.x = 4.95
        action_goal.goal.target_pose.pose.position.y = 0

        action_goal.goal.target_pose.pose.orientation.x = 0
        action_goal.goal.target_pose.pose.orientation.y = 0
        action_goal.goal.target_pose.pose.orientation.z = 0
        action_goal.goal.target_pose.pose.orientation.w = 1

        goal_pub.publish(action_goal)
        #rospy.spin()
        #print(action_goal.goal)

        rate.sleep()
