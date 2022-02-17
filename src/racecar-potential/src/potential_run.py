#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Joy, Image
from nav_msgs.msg import Path

import math
import shutil
import os
from tf.transformations import euler_from_quaternion

import cv2
from cv_bridge import CvBridge

import numpy as np

import time


import rospkg
# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()
# get the file path for rospy_tutorials
pkg_path = rospack.get_path("lane_detection")
import sys
sys.path.insert(0, pkg_path + "/src")

from driver import DriverNode



class PotentialField():

    def __init__(self, laser_topic):

        rospy.Subscriber(laser_topic, LaserScan, self.scan_cb, queue_size=10)
        rospy.Subscriber("/joy", Joy, self.joy_callback, queue_size=10)
        rospy.Subscriber("/lane_detection/path", Path, self.goal_callback, queue_size=10)

        self.ack_pub = rospy.Publisher("/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=10)

        self.wheelbase = 0.325

        self.gap_angle = 45
        self.max_dist = 1.5

        self.min_vel_x = -0.2
        self.max_vel_x = 2.0
        self.min_theta = -20
        self.max_theta = 20
        self.ranges = []

        self.mode = "MANUAL"

        # Parameters
        self.KA = 3.0  # attractive potential gain
        self.KRX = 0.015  # repulsive potential gain X Axis
        self.KRY = 30.0  # repulsive potential gain Y Axis

        #self.KA = 0.0  # attractive potential gain
        #self.KRX = 0.0  # repulsive potential gain X Axis
        self.KRY = 30.0  # repulsive potential gain Y Axis



        self.vel_x = 0
        self.steering = 0

        self.map_height = int(self.max_dist*100)
        self.map_width = int(self.max_dist*100)
        self.map_origin = (self.map_width/2, self.map_height)

        self._map = _map = np.ones([self.map_width, self.map_height, 3], np.uint8)*255

        self.zed_img = np.zeros((376,672,3), dtype=np.int16)


        self.goal = [2, 0, 0]

        




    def goal_callback(self, goal_msg):

        start = time.time()

        goal_x = goal_msg.poses[0].pose.position.x / 100
        goal_y = goal_msg.poses[0].pose.position.y / 100

        ori = goal_msg.poses[0].pose.orientation
        goal_angle = math.degrees(euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])[2])

        self.goal = [goal_x, goal_y, goal_angle]

        self.calculate_potential()

        t = time.time() - start
        f = 1/t
        print("\n\ntime : {0}, frequency: {1}".format(t, f))



    def scan_cb(self, msg):


        self.ranges = []

        self.obstacles = []

        self.angle_increment = math.degrees(msg.angle_increment)

        self.min_ang = int(max(-self.gap_angle*self.angle_increment, math.degrees(msg.angle_min)))
        self.max_ang = int(min(self.gap_angle*self.angle_increment, math.degrees(msg.angle_max)))

        

        for i in range(self.min_ang, self.max_ang):

            angle = i * self.angle_increment

            if(msg.ranges[i] < self.max_dist):

                self.ranges.append(msg.ranges[i])
                
                px = math.cos(math.radians(angle))*msg.ranges[i] 
                py = math.sin(math.radians(angle))*msg.ranges[i] if i > 0 else math.sin(math.radians(angle))*msg.ranges[i]
                
                self.obstacles.append([px, py])

        


    
    def joy_callback(self, joy):
        
        if(joy.buttons[5]):
            self.mode = "AUTO"
        
        elif(joy.buttons[4]):
            self.mode = "MANUAL"

        else:
            self.mode = "DISABLED"




    
    def visualize_potential(self, vel_x, vel_y):

        _map = np.ones([self.map_width, self.map_height, 3], np.uint8)*255

        cv2.circle(_map, self.map_origin, 10, (0,0,0), -1)

        dy = int(math.tan(math.radians(self.max_ang))*self.map_height)

        cv2.line(_map, self.map_origin, (self.map_origin[0]+dy, 0), (0, 0, 0), 2)
        cv2.line(_map, self.map_origin, (self.map_origin[0]-dy, 0), (0, 0, 0), 2)
        cv2.line(_map, self.map_origin, (self.map_origin[0], 0), (25, 25, 25), 1)


        cv2.circle(_map, (int(self.map_origin[0]-self.goal[1]*100), int(self.map_origin[1]-self.goal[0]*100)), 5, (0,0,255), -1)

        for obs in self.obstacles:

            px = int(self.map_origin[1] - obs[0]*100)
            py = int(self.map_origin[0] - obs[1]*100)

            py = max(min(self.map_width-1, py), 0)
            px = max(min(self.map_height-1, px), 0)
            
            #_map[py, px] = (255,0,0)
            cv2.circle(_map, (py,px), 2, (255,0,0), -1)


        
        vy = self.map_origin[1] - int(vel_x*100)
        vx = self.map_origin[0] - int(vel_y*100)

        
        cv2.line(_map, self.map_origin, (vx, vy), (0, 0, 255), 2)


        self._map = _map

        nmap = cv2.resize(_map.copy(), (_map.shape[0]*2, _map.shape[1]*2))
        cv2.imshow("map", nmap)
        cv2.waitKey(1)


    
    def calc_attractive_potential(self, goal):
        return self.KA * np.hypot(-goal[0], -goal[1])


    def calc_repulsive_potential(self, obstacles):

        rx = 0
        ry = 0
        rya = 0

        f_hit = False

        for obs in obstacles:
            
            d = np.hypot(obs[0], obs[1])

            f_hit = True if d < np.hypot(self.goal[0], self.goal[1]) else False
                
            rx += obs[0] / d

            rya += abs(obs[1]) / d
            ry += obs[1] / d

        ry = np.sign(ry)*rya


        p_rx = self.KRX * rx ** 2
        p_ry = -self.KRY * (1.0 / ry) ** 2 * np.sign(ry)
        

        return p_rx, p_ry, f_hit
        



    def calculate_potential(self):

        
        pot_att = self.calc_attractive_potential(self.goal)

        pot_rep_x, pot_rep_y, hit = self.calc_repulsive_potential(self.obstacles) if len(self.obstacles) > 1 else (0,0,0)

        vel_x = pot_att - pot_rep_x
        
        

        if np.sign(self.goal[1]) == np.sign(pot_rep_y):
            vel_y = -pot_rep_y
        
        else:
            vel_y = self.goal[1] - pot_rep_y


        print("\n---------------------------------------------------")
        
        print("\nGoal Point : X = {0}, Y = {1}, YAW = {2}\n".format(self.goal[0], self.goal[1], self.goal[2]))

        print("Attractive Potential [X] : {}".format(pot_att))

        print("Repulsive Potential [X] : {}".format(pot_rep_x))
        print("Repulsive Potential [Y] : {}".format(pot_rep_y))

        print("\nTrans Velocity [X] : {}".format(vel_x))
        print("Trans Velocity [Y] : {}".format(vel_y))

        #if hit:
        if True:
            #self.pub_command(vel_x=vel_x, vel_y=vel_y)
            sender.set_pose_target(target_x=vel_x, target_y=-vel_y)
        

        self.visualize_potential(vel_x=vel_x, vel_y=vel_y)


    
    def convert_trans_rot_vel_to_steering_angle(self, v, omega, wheelbase):
        if omega == 0 or v == 0:
            return 0

        radius = v / omega
        return math.atan(wheelbase / radius)
        

    
    def pub_command(self, vel_x, vel_y):

        ack_msg = AckermannDriveStamped()

        ack_msg.header.frame_id = "odom"

        steer = self.convert_trans_rot_vel_to_steering_angle(vel_x, vel_y, self.wheelbase)

        print("\nCalculated Velocity : {}".format(vel_x))
        print("Calculated Steering Angle : {}".format(math.degrees(steer)))

        cmd_vel_x = max(min(self.max_vel_x, vel_x), self.min_vel_x)
        steering_angle = max(min(math.radians(self.max_theta), steer), math.radians(self.min_theta))

        self.vel_x = cmd_vel_x
        self.steering = steering_angle
        ack_msg.drive.speed = self.vel_x
        ack_msg.drive.steering_angle = self.steering

        print("\nVelocity : {}".format(cmd_vel_x))
        print("Steering Angle : {}".format(math.degrees(steering_angle)))


        if(self.mode == "AUTO"):
            self.ack_pub.publish(ack_msg)
        else:
            pass

    



if __name__ == "__main__":

    rospy.init_node("potential_field")


    sender = DriverNode(cmd_type="position")
    sender.Pid.Kp = 1.0


    potential_field = PotentialField(laser_topic="/rplidar/scan")


    rospy.spin()
