import matplotlib.pyplot as plt
import rospy
import tf
from nav_msgs.msg import Odometry

import numpy as np
from matplotlib.animation import FuncAnimation
import time
from nav_msgs.msg import Path

from Pid import PIDController

from run import DriverNode
import datetime as dt

from ackermann_msgs.msg import AckermannDriveStamped



class Visualiser:
    def __init__(self):

        self.controller = DriverNode().Pid

        self.start = time.time()

        self.joy_steer = 0

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3,1)
        ln, = self.ax1.plot([], [], 'r--', lw=1)
        ln2, = self.ax2.plot([], [], 'b--', lw=1)
        ln3, = self.ax3.plot([], [], 'g--', lw=1)
        self.line = [ln, ln2, ln3]
        
        self.x_data, self.y_data, self.y_data2, self.y_data3 = [] , [], [], []

        rospy.Subscriber("/lane_detection/path", Path, self.pose_callback, queue_size=10)
        rospy.Subscriber("/ackermann_cmd_mux/input/teleop", AckermannDriveStamped, self.teleop_callback, queue_size=10)

        

    def plot_init(self):

        self.ax1.set_xlim(0, 600)
        self.ax1.set_ylim(-100, 100)
        self.ax1.title.set_text('Lateral Deviation')
        self.ax1.set_ylabel('Deviation (px)')
        self.ax2.set_xlim(0, 600)
        self.ax2.set_ylim(-25, 25)
        self.ax2.title.set_text('Controller Output')
        self.ax2.set_ylabel('Steering Angle (degree)')
        self.ax3.set_xlim(0, 600)
        self.ax3.set_ylim(-100, 100)
        self.ax3.title.set_text('Joystick Input')
        self.ax3.set_ylabel('Deviation (px)')        

        return self.line


    def teleop_callback(self, msg):

        self.joy_steer = msg.drive.steering_angle



    def pose_callback(self, msg):

        if (time.time()-self.start)*10 >= 600:
            self.start = time.time()
            self.x_data, self.y_data, self.y_data2, self.y_data3 = [] , [], [], []

        self.x_data.append((time.time()-self.start)*10)

        target_y = msg.poses[0].pose.position.y 
        steering_angle = self.controller.output(goal=target_y)


        self.y_data.append(target_y)
        self.y_data2.append(steering_angle)
        self.y_data3.append(self.joy_steer)




    
    def update_plot(self, frame):
        self.line[0].set_data(self.x_data, self.y_data)
        self.line[1].set_data(self.x_data, self.y_data2)
        self.line[2].set_data(self.x_data, self.y_data3)
        return self.line




if __name__ == "__main__":

    rospy.init_node('visual_node')
    vis = Visualiser()

    ani = FuncAnimation(vis.fig, vis.update_plot, init_func=vis.plot_init)
    plt.show(block=True) 

    rospy.spin()


