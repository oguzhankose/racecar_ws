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


class PreProcessImg():

    def __init__(self, img):
        

        #   ###
        #   1 Take Image and Seq as Input
        # img[0] for img img[1] for img name
        self.orig_img = img[0]
        self.img_name = img[1]
        self.imshape = img[0].shape


        #cv2.imshow("1 orig_img", self.orig_img)
        #cv2.waitKey(0)

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

        for i in range(1, nb_components):
            if (w[i-1] / h[i-1]) < 8 and area[i-1] > threshold:
                out_img[labels == i] = 255
        
        return out_img 

      

    def process_image(self):


        #   ###
        #   2 Grayscale one color channel
        grayImg = cv2.cvtColor(self.orig_img.copy(), cv2.COLOR_BGR2GRAY)


        #   ###
        #   3 Gaussian Blur
        """
        Use low pass filter to remove noise. Will remove high freq stuff like noise and edges
        kernel_size specifies width/height of kernel, should be positive and odd
        Also specify stand dev in X and Y direction, give zero to calculate from kernel size
        Can also use average, median, and bilarteral blurring techniques
        """
        
        smoothed = cv2.bilateralFilter(grayImg.copy(), 11, 75, 75)


        #   ###
        #   4 Create mask to only keep area defined by four coners
        # Black out every area outside area
        vertices = np.array([[(0,self.imshape[0]), (0, 230), (self.imshape[1], 230), (self.imshape[1],self.imshape[0])]], dtype=np.int32)
        # defining a blank mask to start with, 0s with same shape of edgesImg
        # fill pixels inside the polygon defined by vertices"with the fill color
        mask = np.zeros_like(smoothed)  
        cv2.fillPoly(mask, vertices, 255)
        # create image only where mask and edge Detection image are the same
        maskedImg = cv2.bitwise_and(smoothed.copy(), mask)



        #   ###
        #   5 Adaptive Threshold for Filtering
        """
        src – Source 8-bit single-channel image.
        dst – Destination image of the same size and the same type as src .
        maxValue – Non-zero value assigned to the pixels for which the condition is satisfied. See the details below.
        adaptiveMethod – Adaptive thresholding algorithm to use, ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C . See the details below.
        thresholdType – Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV .
        blockSize – Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
        C – Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
        """
        thresholdImg = cv2.adaptiveThreshold(maskedImg.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 85, 22)

        #  Clear the Horizon that caused by masking image
        thresholdImg[200:230] = 0


        #   ###
        #   6 Dilate Image
        kernel = np.array([[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]],np.uint8)
        dilateImg = cv2.dilate(thresholdImg.copy(), kernel, iterations=2)


        # 7 Removing Noise
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype="uint8")
        cleanImg = cv2.morphologyEx(dilateImg.copy(), cv2.MORPH_OPEN, kernel, iterations = 3)  # gurultuden kurtul      


        #   ###
        #   8 Conncected Components (Filter by area and width/height ratio of components)
        connectedImg = self.connected_components(dilateImg, 499)


        #   ###
        #   9 Edge Detection
        """
        finds gradient in x,y direction, gradient direction is perpendicular to edges
        checks pixels in gradient directions to see if they are local maximums, meaning on an edge
        hysteresis thresholding has min and max value, edges with gradient intensity big enough are edges
        edges that lie in bewteen are check to see if they connect to edges with intensity greater than max value, then it is considered edge
        also assumes edges are long lines (not small pixels regions)
        """

        v = np.median(connectedImg.copy())
        sigma = 0.33
        #---- apply optimal Canny edge detection using the computed median----
        lower_thresh = int(max(0, (1.0 - sigma) * v))
        upper_thresh = int(min(255, (1.0 + sigma) * v))
        edgesImg = cv2.Canny(connectedImg.copy(), lower_thresh, upper_thresh)

        


        comb1 = np.concatenate((maskedImg[200:], thresholdImg[200:]), axis=1)
        comb2 = np.concatenate((dilateImg[200:], cleanImg[200:]), axis=1)
        comb3 = np.concatenate((connectedImg[200:], edgesImg[200:]), axis=1)
        comb = np.concatenate((comb1, comb2, comb3), axis=0)
        
        


        #cv2.imshow("out", comb)

        return edgesImg

        

        
       



        
        





        

                
        
