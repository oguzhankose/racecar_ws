import rospy
import os
import traceback
import time
import math

from nav_msgs.msg import Path
from tf.transformations import euler_from_quaternion
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Joy
from Pid import PIDController



class DriverNode():

    def __init__(self, type="path"):

        rospy.Subscriber("/joy", Joy, self.joy_callback, queue_size=10)

        rospy.Subscriber("/lane_detection/path", Path, self.pose_callback, queue_size=10)

        self.ack_pub = rospy.Publisher("/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=10)

        self.ack_msg = AckermannDriveStamped()

        # Controller Coefficients 
        self.Pid = PIDController(Kp = 0.02, Ki = 0.0, Kd = 0.0)

        self.type = "path"
        self.speed = 1.0
        self.mode = "MANUAL"


    def joy_callback(self, joy):
    
        if(joy.buttons[5]):
            self.mode = "AUTO"
        
        elif(joy.buttons[4]):
            self.mode = "MANUAL"

        else:
            self.mode = "DISABLED"


    def pose_callback(self, msg):

        if self.type == "path":
            
            self.ack_msg.header.frame_id = "odom"

            target_x = msg.poses[0].pose.position.x
            target_y = msg.poses[0].pose.position.y 

            ori = msg.poses[0].pose.orientation

            target_ori = math.degrees(euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])[2])

            target_ori = 0      ####################

            self.ack_msg.drive.speed = self.speed
            #ack_msg.drive.acceleration = 0
            #goal = math.hypot(target_y + target_x)
            steering_angle = self.Pid.output(goal=target_y)
            
            # maybe translation2rot function
            
            self.ack_msg.drive.steering_angle = -steering_angle

            print(self.mode)
            print("\nVelocity : {}".format(self.speed))
            print("Steering Angle : {0} target y : {1}".format(-steering_angle, target_y))


            if(self.mode == "AUTO"):
                self.ack_pub.publish(self.ack_msg)
            else:
                pass






if __name__ == "__main__":

    sender = DriverNode()
    
    rospy.spin()

