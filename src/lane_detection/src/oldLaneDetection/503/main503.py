#!/usr/bin/env python

import os
import numpy as np
import cv2
import traceback
import time

from LaneDetection503 import LaneDetection
from PreProcessImg import PreProcessImg

import rospy
import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def img_callback(img, seq):

    start = time.time()


    try:
        cv2.imwrite("inputs/input_{}.png".format(str(msg.header.seq)), img) 
        
        detector = LaneDetection([img.copy(), seq])
        
        pre_process = PreProcessImg([img.copy(), seq])
        processedImg, left_img, right_img = pre_process.process_image()

       
        blended_img, out_img, mirror, path = detector.process_image(processedImg, left_img, right_img)
        
        ###########################################     output contatenation
        out_img = cv2.copyMakeBorder(
                    out_img, 
                    (img.shape[0] - out_img.shape[0])/2, 
                    (img.shape[0] - out_img.shape[0])/2, 
                    (img.shape[1] - out_img.shape[1])/2, 
                    (img.shape[1] - out_img.shape[1])/2, 
                    cv2.BORDER_CONSTANT, 
                    value=(128,128,128)
                )

        #comb = np.concatenate((img, out_img), axis=0)
        comb = np.concatenate((blended_img, out_img), axis=0)
        ###########################################     output contatenation
        
        
        org1 = (10, comb.shape[1]+35)
        org2 = (10, comb.shape[1]+55)
        org3 = (10, comb.shape[1]+75)

        org4 = (10, comb.shape[1]+15)

        # Using cv2.putText() method 
        cv2.putText(comb, "x:{0}, y:{1}, yaw:{2}".format(path[0][0], path[0][1], round(path[0][2], 1)), org1, cv2.FONT_HERSHEY_SIMPLEX,  
                0.5, (0, 0, 255), 1, cv2.LINE_AA) 
        cv2.putText(comb, "x:{0}, y:{1}, yaw:{2}".format(path[1][0], path[1][1], round(path[1][2], 1)), org2, cv2.FONT_HERSHEY_SIMPLEX,  
                0.5, (0, 255, 0), 1, cv2.LINE_AA) 
        cv2.putText(comb, "x:{0}, y:{1}, yaw:{2}".format(path[2][0], path[2][1], round(path[2][2], 1)), org3, cv2.FONT_HERSHEY_SIMPLEX,  
                0.5, (255, 0, 0), 1, cv2.LINE_AA) 

        
        cv2.putText(comb, "Mirroring: {}".format(mirror), org4, cv2.FONT_HERSHEY_SIMPLEX,  
                0.5, (120, 180, 75), 1, cv2.LINE_AA) 

        #cv2.imshow("output", comb)
        cv2.waitKey(1)
        cv2.imwrite("outputs/output_{}.png".format(str(msg.header.seq)), comb) 
        
        

        
    except Exception as e:

        print(traceback.format_exc())
        
        cv2.imshow("Failure", img.copy())
        cv2.waitKey(1)

        #quit()
        
        return False

    

    pose_list = []

    for i in range(len(path)):

        pose = PoseStamped()
        pose.pose.position.x = path[i][0]
        pose.pose.position.y = path[i][1]
        pose.pose.position.z = 0

        quat = tf.transformations.quaternion_from_euler(0,0,path[i][2])
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        pose_list.append(pose)


    path_msg = Path()
    path_msg.poses = pose_list

    t = time.time() - start
    f = 1/t
    print("time : {0}, frequency: {1}".format(t, f))






if __name__ == "__main__":

    folder = "/media/nvidia/sdcard/new_frames_1/"
    

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img_callback(img, filename)
            cv2.waitKey(0)

        
        


    
    os.chdir("/media/nvidia/sdcard/opencv_logs")
    os.system("rm -rf inputs/")
    os.system("mkdir inputs")

    os.system("rm -rf outputs/")
    os.system("mkdir outputs")
    
    rospy.spin()

