#!/usr/bin/env python

import os
import numpy as np
import cv2
import traceback
import time

from PreProcessImg import PreProcessImg
from LaneDetection502 import LaneDetection

import rospy
import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def img_callback(img, seq):

    start = time.time()


    pre = PreProcessImg([img.copy(), seq])
    new = pre.process_image()

    detector = LaneDetection([img.copy(), seq])
    old = detector.process_image()

    comb = np.concatenate((old[200:], new[200:]), axis=1)
    cv2.imshow("out", comb)
        


if __name__ == "__main__":

    folder = "/media/nvidia/sdcard/opencv_logs/inputs/"
    
    
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img_callback(img, filename)
            cv2.waitKey(0)

        
        


    """
    os.chdir("/media/nvidia/sdcard/opencv_logs")
    os.system("rm -rf inputs/")
    os.system("mkdir inputs")

    os.system("rm -rf outputs/")
    os.system("mkdir outputs")
    """
    rospy.spin()

