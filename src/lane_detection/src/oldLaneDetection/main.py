#!/usr/bin/env python

import os
import sys
import time
import matplotlib.image as mpimg
import cv2
import traceback

from LaneDetectionv2 import LaneDetection


if __name__ == "__main__":
    
    # Get all images
    test_img_path = '/home/nvidia/marc/src/lane_detection/src/test/'

    test_images = [mpimg.imread(test_img_path + i) for i in os.listdir(test_img_path)]
    test_images_names = os.listdir(test_img_path) 

    images = zip(test_images, test_images_names)

    os.system("rm -rf outputs/")
    os.system("mkdir outputs")
    
    for image in images:

        try:
            start = time.time()

            detector = LaneDetection(image)
            detector.verbose = False

            out_img, path = detector.process_image()
            print(path)

            cv2.imshow('Output', out_img)
            cv2.waitKey(0)

            t = time.time() - start
            f = 1/t
            print("time : {0}, frequency: {1}".format(t, f))

        except Exception as e:
            print(traceback.format_exc())
            quit()
            

