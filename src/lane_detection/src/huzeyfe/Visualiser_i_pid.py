#!/usr/bin/env python

import matplotlib.pyplot as plt
import rospy
import tf
from nav_msgs.msg import Odometry

import numpy as np
from matplotlib.animation import FuncAnimation
import time
from nav_msgs.msg import Path

from Pid import PIDController

from driver_i_pid import DriverNode
import datetime as dt

from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Joy

import os

import cv2
import csv



class visualiser():
    def __init__(self):


        self.controller = DriverNode().Pid
           

        #self.data = np.zeros((1, 4))
        #print(self.data)

        self.start = time.time()

        self.joy_steer = 0
        self.mode = 0
        self.prevTime  = 0.0
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1)
        ln, = self.ax1.plot([], [], 'r', lw=1)
        ln2, = self.ax2.plot([], [], 'b', lw=1)
        
        self.line = [ln, ln2]
        
        self.x_data, self.y_data, self.y_data2, self.input = [], [], [], []
        
        
        rospy.Subscriber("/lane_detection/path", Path, self.pose_callback, queue_size=10)
    
        rospy.Subscriber("/joy", Joy, self.joy_callback, queue_size=10)



        

    def plot_init(self):

        self.ax1.set_xlim(0, 60)
        self.ax1.set_ylim(-100, 100)
        self.ax1.title.set_text('Lateral Deviation')
        self.ax1.set_ylabel('Deviation (px)')
        self.ax2.set_xlim(0, 60)
        self.ax2.set_ylim(-5, 5)
        self.ax2.title.set_text('Controller Output')
        self.ax2.set_ylabel('Steering Angle (degree)')
       

        return self.line

    def joy_callback(self, joy):
    
        if(joy.axes[5] == -1):
            self.mode = 1
        
        elif(joy.buttons[4]):
            self.mode = 0

        else:
            self.mode = 0



    def pose_callback(self, msg):

        
        self.target_y = msg.poses[0].pose.position.y 
        self.steering_angle = self.controller.prev_output

        # Data Logging

        

        
        
    
            
        
    
    def update_plot(self, frame):
        self.line[0].set_data(self.x_data, self.y_data)
        self.line[1].set_data(self.x_data, self.y_data2)

        return self.line



if __name__ == "__main__":

    rospy.init_node('visual_node')


    vis = visualiser()

    ani = FuncAnimation(vis.fig, vis.update_plot, init_func=vis.plot_init)

    plt.show(block=True)

    
    r = rospy.Rate(20)

    time.sleep(1)
    while not rospy.is_shutdown():

        with open('io18.csv', 'wb+') as ioFile:
            
            try:
                vis.x_data.append(time.time()-vis.start)
                vis.input.append(vis.mode)
                vis.y_data.append(vis.target_y)
                vis.y_data2.append(vis.steering_angle)
                data = np.array([vis.y_data, vis.y_data2, vis.x_data, vis.input])
                data = data.T
                np.savetxt(ioFile, data, fmt=['%.2f', '%.2f', '%.2f', '%d'], delimiter="\t")
            except Exception as e:
                print("HATA")
                print(e)
        

        r.sleep()
    #rospy.spin()


