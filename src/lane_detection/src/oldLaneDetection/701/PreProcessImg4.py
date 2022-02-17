#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 06:24:33 2017

@author: oguzhankose
"""
import numpy as np
import cv2
import os
import sys
import math
import matplotlib.pyplot as plt
import rospy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import time


class PreProcessImg():

    def __init__(self):
        
        self.pr_img_pub = rospy.Publisher("/lane_detection/processed", Image, queue_size=1)
        rospy.Subscriber("/zed_camera/frame", Image, self.img_callback, queue_size=1, buff_size=52428800)




    def img_callback(self, msg):

        start = time.time()

        img = CvBridge().imgmsg_to_cv2(msg, "bgr8").copy()

        seq = msg.header.seq

        self.process_image(img.copy())

        t = time.time() - start
        f = 1/t
        print("time : {0}, frequency: {1}".format(t, f))
    


    def connected_components(self, image, threshold):

        #find all your connected components (white blobs in your image)
        """
        retval, labels, stats, centroids = cv.connectedComponentsWithStatsWithAlgorithm(image,connectivity,ltype,ccltype[,labels[,stats[,centroids]]]	)
        image	the 8-bit single-channel image to be labeled
        labels	destination labeled image
        stats	statistics output for each label, including the background label. Statistics are accessed via stats(label, COLUMN) where COLUMN is one of ConnectedComponentsTypes, selecting the statistic. The data type is CV_32S.
        centroids	centroid output for each label, including the background label. Centroids are accessed via centroids(label, 0) for x and centroids(label, 1) for y. The data type CV_64F.
        connectivity	8 or 4 for 8-way or 4-way connectivity respectively
        ltype	output image label type. Currently CV_32S and CV_16U are supported.
        ccltype	connected components algorithm type (see ConnectedComponentsAlgorithmsTypes).
        """
        nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

        w = stats[1:, cv2.CC_STAT_WIDTH]
        h = stats[1:, cv2.CC_STAT_HEIGHT]
        area = stats[1:, cv2.CC_STAT_AREA]
        out_img = np.zeros_like(labels,dtype=np.uint8)

 
        ratio = np.divide(w, h, dtype=np.float)

        counter = 0
        for i in range(1, nb_components):

            if (w[i-1] / h[i-1]) < 8 and area[i-1] > threshold:
                out_img[labels == i] = 255
                counter += 1

        try:
            contours, _ = cv2.findContours(out_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        except:
            _, contours, _ = cv2.findContours(out_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)



        counter = 0
        for cnt in contours:
            
            try:
                center, r = cv2.minEnclosingCircle(cnt)
                circ = (math.pi * r**2) / cv2.contourArea(cnt)
            except Exception as e:
                print(e)
                circ=0

            if circ < 2:
                counter += 1
                out_img = cv2.drawContours(out_img, [cnt], -1, 0, -1)
            else:
                pass

        return out_img 



    def bilateral_adaptive_threshold(self, img, ksize=85, C=19, mode='ceil', true_value=255, false_value=0):
        '''
        Perform adaptive color thresholding on a single-channel input image.
        The function uses a cross-shaped filter kernel of 1-pixel thickness.
        The intensity value of a given pixel needs to be relatively higher (or lower, depending on the mode)
        than the average intensity value of either both the left and right sides or both the upper and lower
        sides of the kernel cross. This means in order to exceed the threshold, a pixel does not just need
        to be brighter than its average neighborhood, but instead it needs to be brighter than both sides
        of its neighborhood independently in either horizontal or vertical direction.
        This is useful for filtering lane line pixels, because the area on both sides of a lane line is darker
        than the lane line itself.
        Arguments:
            img (image file): The input image for which a filter mask is to be created.
            ksize (int, optional): The radius of the filter cross excluding the center pixel,
                i.e. for a ksize of `k`, the diameter of the cross will be 2k+1. Defaults to 30.
            C (int, optional): The required difference between the intensity of a pixel and
                its neighborhood in order to pass the threshold. If C = c, a pixel's intensity
                value needs to be higher/lower than that of its neighborhood by c in order to pass
                the threshold.
            mode (string, optional): One of 'floor' or 'ceil'. If set to 'floor', only pixels brighter
                than their neighborhood by C will pass the threshold. If set to 'floor', only pixels
                darker than their neighborhood by C will pass the threshold. Defaults to 'floor'.
            true_value (int, optional): The value to which mask pixels will be set for image pixels
                that pass the threshold. Must be in [0, 255]. Defaults to 255.
            false_value (int, optional): The value to which mask pixels will be set for image pixels
                that do not pass the threshold. Must be in [0, 255]. Defaults to 0.
        Returns:
            A mask of the same shape as the input image containing `true_value` for all pixels that
            passed the filter threshold and `false_value` elsewhere.
        '''

        mask = np.full(img.shape, false_value, dtype=np.uint8) # This will be the returned mask

        # In order to increase the efficiency of the filter, we'll scale everything
        # so that we can work with integer math instead of floating point math.
        # Note that if `p` is the intensity value of the pixel to which the filter
        # is applied, then the following computations are equivalent:
        #
        # avg(kernel_l) > p  <==>  sum(kernel_l) > p * ksize  <==>  sum(kernel_l) - p * ksize > 0
        #
        # The latter equivalence is what you see implemented below.
        #
        kernel_l = np.array([[1] * (ksize) + [-ksize]], dtype=np.int16)
        kernel_r = np.array([[-ksize] + [1] * (ksize)], dtype=np.int16)
        kernel_u = np.array([[1]] * (ksize) + [[-ksize]], dtype=np.int16)
        kernel_d = np.array([[-ksize]] + [[1]] * (ksize), dtype=np.int16)

        # We have to scale C by ksize, too.
        if mode == 'floor':
            delta = C * ksize
        elif mode == 'ceil':
            delta = -C * ksize
        else: raise ValueError("Unexpected mode value. Expected value is 'floor' or 'ceil'.")

        left_thresh = cv2.filter2D(img, cv2.CV_16S, kernel_l, anchor=(ksize,0), delta=delta, borderType=cv2.BORDER_CONSTANT)
        right_thresh = cv2.filter2D(img, cv2.CV_16S, kernel_r, anchor=(0,0), delta=delta, borderType=cv2.BORDER_CONSTANT)
        up_thresh = cv2.filter2D(img, cv2.CV_16S, kernel_u, anchor=(0,ksize), delta=delta, borderType=cv2.BORDER_CONSTANT)
        down_thresh = cv2.filter2D(img, cv2.CV_16S, kernel_d, anchor=(0,0), delta=delta, borderType=cv2.BORDER_CONSTANT)

        if mode == 'floor':
            mask[((0 > left_thresh) & (0 > right_thresh)) | ((0 > up_thresh) & (0 > down_thresh))] = true_value
        elif mode == 'ceil':
            mask[((0 < left_thresh) & (0 < right_thresh)) | ((0 < up_thresh) & (0 < down_thresh))] = true_value

        return mask
        

    def process_image(self, img):

        #   ###
        #   2 Grayscale one color channel
        grayImg = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)


        #   ###
        #   3 Gaussian Blur
        """
        Use low pass filter to remove noise. Will remove high freq stuff like noise and edges
        kernel_size specifies width/height of kernel, should be positive and odd
        Also specify stand dev in X and Y direction, give zero to calculate from kernel size
        Can also use average, median, and bilarteral blurring techniques
        """
        
        smoothed = cv2.bilateralFilter(grayImg, 7, 95, 95)
        #smoothed = cv2.GaussianBlur(grayImg, (5,5), 0)
        #smoothed = grayImg

        #   ###
        #   4 Create mask to only keep area defined by four coners
        # Black out every area outside area
        vertices = np.array([[(0,img.shape[0]), (0, 210), (img.shape[1], 210), (img.shape[1],img.shape[0])]], dtype=np.int32)
        # defining a blank mask to start with, 0s with same shape of edgesImg
        # fill pixels inside the polygon defined by vertices"with the fill color
        mask = np.zeros_like(smoothed)  
        cv2.fillPoly(mask, vertices, 255)
        # create image only where mask and edge Detection image are the same
        maskedImg = cv2.bitwise_and(smoothed, mask)



        #   ###
        #   5 Adaptive Threshold for Filtering
        """
        src – Source 8-bit single-channel image.
        dst – Destination image of the same size and the same type as src .
        maxValue – Non-zero value assigned to the pixels for which the condition is satisfied. See the details below.
        adaptiveMethod – Adaptive thresholding algorithm to use,  or ADAPTIVE_THRESH_GAUSSIAN_C . See the details below.
        thresholdType – Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV .
        blockSize – Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
        C – Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
        """
        #thresholdImg = cv2.adaptiveThreshold(maskedImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #                                  cv2.THRESH_BINARY_INV, 135, 22)    #85
        
        thresholdImg = self.bilateral_adaptive_threshold(maskedImg)

        #  Clear the Horizon that caused by masking image
        thresholdImg[190:220] = 0
        thresholdImg[thresholdImg.shape[0]-20:thresholdImg.shape[0]] = 0
        
        

        morph = cv2.morphologyEx(thresholdImg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (13,13)))
        
        #   ###
        #   7 Conncected Components (Filter by area and width/height ratio of components)
        connectedImg = self.connected_components(morph, 499)

        """
        comb1 = np.concatenate((grayImg, grayImg), axis=1)
        comb2 = np.concatenate((smoothed, maskedImg), axis=1)
        comb3 = np.concatenate((thresholdImg[200:], morph[200:]), axis=1)
        comb4 = np.concatenate((dilateImg[200:], connectedImg[200:]), axis=1)
        comb = np.concatenate((comb1, comb2, comb3, comb4), axis=0)
        
        #cv2.imshow("out", comb)
        #cv2.waitKey(0)

        cv2.namedWindow("output2", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
                             # Read image
        imS = cv2.resize(comb, (960, 540))                    # Resize image
        cv2.imshow("output2", imS)                            # Show image
        cv2.waitKey(1)                                      # Display the image infinitely until any keypress
        """
        


        self.pr_img_pub.publish(CvBridge().cv2_to_imgmsg(connectedImg))


        return connectedImg




rospy.init_node("pre")
pre_process = PreProcessImg()

rospy.spin()

        

        
       



        
        





        

                
        
