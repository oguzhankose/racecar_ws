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


class PotentialField():

    def __init__(self, laser_topic):

        rospy.Subscriber(laser_topic, LaserScan, self.scan_cb, queue_size=10)
        rospy.Subscriber("/joy", Joy, self.joy_callback, queue_size=10)
        rospy.Subscriber("/zed/zed_node/right/image_rect_color", Image, self.zed_callback, queue_size=10)

        rospy.Subscriber("/lane_detection/path", Path, self.goal_callback, queue_size=10)

        self.ack_pub = rospy.Publisher("/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=10)

        self.wheelbase = 0.325

        self.vel_x_gain = 1.2
        self.vel_y_gain = 1.2

        self.gap_angle = 60
        self.max_dist = 3

        self.min_vel_x = -0.2
        self.max_vel_x = 2.0
        self.min_theta = -20
        self.max_theta = 20
        self.ranges = []

        self.mode = "MANUAL"

        self.vel_x = 0
        self.steering = 0

        self.frame = 0

        self.map_height = self.max_dist*100
        self.map_width = self.max_dist*100
        self.map_origin = (self.map_width/2, self.map_height)

        self._map = _map = np.ones([self.map_width, self.map_height, 3], np.uint8)*255

        #shutil.rmtree('frames/', ignore_errors=True)
        #os.makedirs("/home/nvidia/frames")
        
        #self.video_writer = cv2.VideoWriter("output.avi",cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (672,752))



    def goal_callback(self, goal_msg):

        self.goal_x = goal_msg.poses[1].pose.position.x
        self.goal_y = goal_msg.poses[1].pose.position.y

        ori = goal_msg.poses[1].pose.orientation
        self.goal_angle = math.degrees(euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])[2])

        print(self.goal_x, self.goal_y, self.goal_angle)




    def scan_cb(self, msg):

        self.ranges = msg.ranges

        self.angle_increment = msg.angle_increment

        self.calculate_potential()


    
    def joy_callback(self, joy):
        
        if(joy.buttons[5]):
            self.mode = "AUTO"
        
        elif(joy.buttons[4]):
            self.mode = "MANUAL"

        else:
            self.mode = "DISABLED"


    
    def zed_callback(self, msg):
        """
        zed_img = CvBridge().imgmsg_to_cv2(msg).copy()
        zed_img=cv2.cvtColor(zed_img, cv2.COLOR_BGRA2BGR)

        new_map = cv2.resize(self._map.copy(), (zed_img.shape[0],zed_img.shape[0]))
        
        new_map = cv2.copyMakeBorder(
                        new_map, 
                        0, 
                        0, 
                        (zed_img.shape[1] - new_map.shape[1])/2, 
                        (zed_img.shape[1] - new_map.shape[1])/2, 
                        cv2.BORDER_CONSTANT, 
                        value=(128,128,128)
              )

        comb = np.concatenate((new_map, zed_img), axis=0)


        cv2.putText(comb, "Mode: {0}".format(self.mode), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(comb, "Velocity: {0}".format(round(self.vel_x, 2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(comb, "Steering: {0}".format(round(self.steering, 2)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (0, 0, 255), 1, cv2.LINE_AA)


        cv2.imshow("comb", comb)
        cv2.waitKey(1)

        cv2.imwrite("frames/frame{}.jpg".format(self.frame), comb)
        """

        pass
        #self.video_writer.write(comb)



    
    def visualize_potential(self, vel_x, def_vel_y, vel_y):

        _map = np.ones([self.map_width, self.map_height, 3], np.uint8)*255

        cv2.circle(_map, self.map_origin, 10, (0,0,0), -1)

        dy = int(math.tan(math.radians(self.gap_angle/2))*self.map_height)

        
        cv2.line(_map, self.map_origin, (self.map_origin[0]+dy, 0), (0, 0, 0), 2)
        cv2.line(_map, self.map_origin, (self.map_origin[0]-dy, 0), (0, 0, 0), 2)
        cv2.line(_map, self.map_origin, (self.map_origin[0], 0), (0, 0, 0), 1)

        for i in range(-180, 180):

            r_angle = math.radians(i/2)

            if self.ranges[i] < self.max_dist:

                px = int(self.map_origin[0] - math.sin(r_angle)*self.ranges[i]*100)

                py = int(self.map_origin[1] - math.cos(r_angle)*self.ranges[i]*100)

                px = max(min(self.map_width-1, px), 0)
                py = max(min(self.map_height-1, py), 0)
                
                #_map[py, px] = (255,0,0)
                cv2.circle(_map, (px,py), 2, (255,0,0), -1)

            else:
                pass

        
        vy = self.map_origin[1] - int(vel_x*100)
        vx = self.map_origin[0] - int(vel_y*100)
        vx_def = self.map_origin[0] - int(def_vel_y*100)

        
        cv2.line(_map, self.map_origin, (vx, vy), (255, 0, 0), 2)

        #default vector
        cv2.line(_map, self.map_origin, (vx_def, vy), (0, 0, 255), 2)

        self._map = _map

        cv2.imshow("map", _map)
        cv2.waitKey(1)

        

    def calculate_potential(self):

        self.frame += 1
        
        pot_x = 0

        pot_y = 0
        pos_pot_y = 0
        neg_pot_y = 0

        counter = 0
        d_c = 0
        
        for i in range(int(-self.gap_angle/2 * len(self.ranges)/360),
                       int(self.gap_angle/2 * len(self.ranges)/360)):
        
            _range = min(self.ranges[i], self.max_dist)
            angle = i/2

            counter += 1

            if(_range < self.max_dist):

                d_c += 1

                pot_x += math.cos(math.radians(angle)) / (_range / self.max_dist)

                if i > 0:
                    pos_pot_y += math.cos(math.radians(angle)) / (_range / self.max_dist)
                else:
                    neg_pot_y += -1 * math.cos(math.radians(angle)) / (_range / self.max_dist)

        
        trans_vel_x = self.max_vel_x - (pot_x / counter * self.vel_x_gain)

        if(abs(abs(pos_pot_y) - abs(neg_pot_y)) < 2*d_c and trans_vel_x < 2.0):

            pot_y = pos_pot_y if(abs(pos_pot_y)>abs(neg_pot_y)) else neg_pot_y
            
        else:
            pot_y = pos_pot_y + neg_pot_y

        def_vel_y = (pos_pot_y + neg_pot_y) / counter * -1 * self.vel_y_gain

        trans_vel_y = pot_y / counter * -1 * self.vel_y_gain



        #Goal Attractive Force Addition
        ###########

        ###########


        ###########



        print("\n---------------------------------------------------")
        print("Potential [X] : {}".format(pot_x))
        print("Positive Potential [Y] : {}".format(pos_pot_y))
        print("Negative Potential [Y] : {}".format(neg_pot_y))

        print("\nTrans Velocity [X] : {}".format(trans_vel_x))
        print("Trans Velocity [Y] : {}".format(trans_vel_y))

        self.pub_command(vel_x=trans_vel_x, vel_y=trans_vel_y)
        
        self.visualize_potential(vel_x=trans_vel_x, def_vel_y=def_vel_y, vel_y=trans_vel_y)


    
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

    potential_field = PotentialField(laser_topic="/rplidar/scan")

    rospy.spin()
