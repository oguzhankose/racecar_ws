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

        counter = 0
        for i in range(1, nb_components):

            if (w[i-1] / h[i-1]) < 8 and area[i-1] > threshold:
                out_img[labels == i] = 255
                counter += 1

        try:
            contours, _ = cv2.findContours(out_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        except:
            _, contours, _ = cv2.findContours(out_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)



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
                out_img = cv2.drawContours(out_img.copy(), [cnt], -1, 0, -1)
            else:
                pass

        return out_img 


    def nothing(self, x):
        pass


    def sss(self, thresholdImg):

        image = self.orig_img.copy()

        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        # Create a window
        cv2.namedWindow('image')

        # create trackbars for color change
        cv2.createTrackbar('HMin','image',0,179,self.nothing) # Hue is from 0-179 for Opencv
        cv2.createTrackbar('SMin','image',0,255,self.nothing)
        cv2.createTrackbar('VMin','image',0,255,self.nothing)
        cv2.createTrackbar('HMax','image',0,179,self.nothing)
        cv2.createTrackbar('SMax','image',0,255,self.nothing)
        cv2.createTrackbar('VMax','image',0,255,self.nothing)

        # Set default value for MAX HSV trackbars.
        cv2.setTrackbarPos('HMax', 'image', 179)
        cv2.setTrackbarPos('SMax', 'image', 255)
        cv2.setTrackbarPos('VMax', 'image', 255)

        # Initialize to check if HSV min/max value changes
        hMin = sMin = vMin = hMax = sMax = vMax = 0
        phMin = psMin = pvMin = phMax = psMax = pvMax = 0

        
        cv2.setTrackbarPos('HMin','image', 0)
        cv2.setTrackbarPos('SMin','image', 50)
        cv2.setTrackbarPos('VMin','image', 70)

        cv2.setTrackbarPos('HMax','image', 9)
        cv2.setTrackbarPos('SMax','image', 255)
        cv2.setTrackbarPos('VMax','image', 255)

        
        wait_time = 33

        while(1):

            output = np.zeros_like(thresholdImg)

            # get current positions of all trackbars
            hMin = cv2.getTrackbarPos('HMin','image')
            sMin = cv2.getTrackbarPos('SMin','image')
            vMin = cv2.getTrackbarPos('VMin','image')

            hMax = cv2.getTrackbarPos('HMax','image')
            sMax = cv2.getTrackbarPos('SMax','image')
            vMax = cv2.getTrackbarPos('VMax','image')

            # Set minimum and max HSV values to display
            lower = np.array([hMin, sMin, vMin])
            upper = np.array([hMax, sMax, vMax])

            # Create HSV Image and threshold into a range.
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            
            #output = cv2.bitwise_and(image,image, mask= mask)
            output = cv2.bitwise_not(output.copy(), output.copy(), mask=mask)
            cv2.imshow("orig", self.orig_img)

            # Print if there is a change in HSV value
            if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
                print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
                phMin = hMin
                psMin = sMin
                pvMin = vMin
                phMax = hMax
                psMax = sMax
                pvMax = vMax

            # Display output image
            cv2.imshow('image',mask)

            # Wait longer to prevent freeze for videos.
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

        #cv2.destroyAllWindows()


    def hsv_red_lane(self, thresholdImg):
        # Set minimum and max HSV values to display
        lower1 = np.array([0, 10, 20])
        upper1 = np.array([5, 100, 200])

        lower2 = np.array([174, 10, 20])
        upper2 = np.array([179, 100, 200])
        mid_line_cnt = []

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(self.orig_img.copy(), cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)

        mask = cv2.bitwise_or(mask1, mask2)


        #  Clear the Horizon that caused by masking image
        mask[0:220] = 0
        mask[mask.shape[0]-20:mask.shape[0]] = 0

        
        mask = cv2.morphologyEx(mask.copy(), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (56,56)))
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype="uint8")
        mask = cv2.morphologyEx(mask.copy(), cv2.MORPH_OPEN, kernel, iterations = 1)  # gurultuden kurtul
        
    
        try:
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        except:
            _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        hsv_list = [cv2.contourArea(cnt) for cnt in contours]
        max_hsv_index = hsv_list.index(max(hsv_list))


        if cv2.contourArea(contours[max_hsv_index]) > 400 and cv2.contourArea(contours[max_hsv_index]) < 5000:
            
            mask = cv2.drawContours(np.zeros_like(mask), contours, max_hsv_index, 255, -1)
            
            
            mid_line_cnt.append(contours[max_hsv_index])
            
        else:
            mid_line = np.zeros_like(mask)
            

        hsv_thresh = np.zeros_like(thresholdImg)
        hsv_thresh = cv2.bitwise_not(hsv_thresh.copy(), thresholdImg.copy(), mask=mask)
        
        

        return hsv_thresh, mid_line_cnt
      

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
        vertices = np.array([[(0,self.imshape[0]), (0, 210), (self.imshape[1], 210), (self.imshape[1],self.imshape[0])]], dtype=np.int32)
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
        thresholdImg[190:220] = 0
        thresholdImg[thresholdImg.shape[0]-20:thresholdImg.shape[0]] = 0
        
        """
        try:
            hsv_thresh, mid_line_cnt = self.hsv_red_lane(thresholdImg)
            
            
        except:
            print("No RED LANE !!")
            mid_line_cnt = None
            hsv_thresh = thresholdImg.copy()
        """
      
        mid_line_cnt = None
        
        hsv_thresh = thresholdImg.copy()
        #   ###
        #   6 Dilate Image
        kernel = np.array([[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]],np.uint8)
        dilateImg = cv2.dilate(hsv_thresh.copy(), kernel, iterations=1)


        #   ###
        #   7 Conncected Components (Filter by area and width/height ratio of components)
        connectedImg = self.connected_components(dilateImg, 499)

    


        return connectedImg, mid_line_cnt

        

        
       



        
        





        

                
        
