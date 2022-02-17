#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 06:24:33 2017

@author: oguzhankose
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import sys
import math
import time
from numpy.polynomial.polynomial import polyfit, polyval, polyder
import scipy.optimize

import Tkinter as tk # use tkinter 



class LaneDetection():

    def __init__(self, img):
        
        # img[0] for img img[1] for img name
        self.orig_img = img[0]
        self.img_name = img[1]

        self.imshape = img[0].shape

        self.st_point = 200
        self.md_point = 125
        self.fi_point = 50
       

    def draw_lines(self, img, lines):

        lines_img = img.copy()
        # Check if we got more than 1 line
        if lines is not None and len(lines) > 2:
            # Draw all lines onto image
            for i in range(len(lines)):
                for x1,y1,x2,y2 in lines[i]:
                    cv2.line(lines_img,(x1,y1),(x2,y2),(255,255,0),2) # plot line   

        return lines_img
    
        
    def hough_lines(self, img):

        rho = 2 # distance resolution in pixels of the Hough grid
        theta = np.pi / 180 # angular resolution in radians of the Hough grid
        threshold = 45     # minimum number of votes (intersections in Hough grid cell)
        min_line_len = 20  #40 minimum number of pixels making up a line
        max_line_gap = 20    #100 maximum gap in pixels between connectable line segments
        lines = cv2.HoughLinesP(img.copy(), rho, theta, threshold, np.array([]), 
                                    minLineLength=min_line_len, maxLineGap=max_line_gap)

        return lines

    
    def process_lines(self, lines):
        #-----------------------Separate Lines Intro Positive/Negative Slope--------------------------
        # Separate line segments by their slope to decide left line vs. the right line
        slopePositiveLines = [] # x1 y1 x2 y2 slope
        slopeNegativeLines = []

        # Loop through all lines
        addedPos = False
        addedNeg = False

        # Convert lines from Int to Float
        lines = np.array(lines, dtype=float)
        for currentLine in lines:   
            # Get points of current Line
            for x1,y1,x2,y2 in currentLine:

                lineLength = ((x2-x1)**2 + (y2-y1)**2)**.5 # get line length

                if lineLength > 25: # if line is long enough

                    if x2 != x1: # dont divide by zero

                        slope = (y2-y1)/(x2-x1) # get slope line

                        # Check angle of line w/ xaxis. dont want vertical/horizontal lines
                        theta = math.degrees(math.atan((abs(y2-y1))/(abs(x2-x1)))) # 

                        if slope > 0 and abs(theta) < 88 and abs(theta) > 18: 

                            slopePositiveLines.append([x1,y1,x2,y2,-slope, lineLength]) # add positive slope line
                            addedPos = True # note that we added a positive slope line

                        if slope < 0 and abs(theta) < 88 and abs(theta) > 18:
                            
                            slopeNegativeLines.append([x1,y1,x2,y2,-slope, lineLength]) # add negative slope line
                            addedNeg = True # note that we added a negative slope line
        
        
        return addedPos, addedNeg, slopePositiveLines, slopeNegativeLines



    def plot_lane_lines(self, lines):
        
        # make new black image
        colorLines = self.orig_img.copy()
        #laneLines = np.array(np.zeros(self.maskedImg.shape), dtype='int')
        laneLines = np.zeros_like(self.orig_img.copy())*255

        # Draw lines onto image
        line_list = np.array(lines, dtype='int')

        for x1,y1,x2,y2,sl,length in line_list:
            cv2.line(laneLines,(x1,y1),(x2,y2),(255,255,255),2) # plot line   
            cv2.line(colorLines,(x1,y1),(x2,y2),(255,0,0),3) # plot line

        laneLines = cv2.cvtColor(laneLines, cv2.COLOR_BGR2GRAY)

        return laneLines, colorLines

    

    def clear_by_contours(self, c_img):

        contoursImg = c_img.astype(np.uint8)

        contours, _ = cv2.findContours(contoursImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)


        cnt = contours[max_index]
        

        laneContours = c_img

        for i in range(len(contours)):

            x,y,w,h = cv2.boundingRect(contours[i])
            cv2.rectangle(contoursImg,(x,y),(x+w,y+h),255,2)

            if(i != max_index):
                x,y,w,h = cv2.boundingRect(contours[i])
                laneContours = cv2.rectangle(laneContours,(x,y),(x+w,y+h),0,-1)

            else:
                x,y,w,h = cv2.boundingRect(contours[i])
                cv2.rectangle(contoursImg,(x,y),(x+w,y+h),255,4)



        return contoursImg, laneContours
        


    def get_c_points(self, c_img):

        contours, _ = cv2.findContours(c_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        px = []
        py = []
        for point in max(contours):
            px.append(point[0][1])
            py.append(point[0][0])

        return px, py



    def quad(self, xdata, a, b, c):
    
        return a * xdata**2 + b * xdata + c
    
    

    def get_poly(self, c_img, px, py, bnd=None):
        
        
        if(bnd == 1):
            x0 = [-0.001,1,10]            
            popt, pcov = scipy.optimize.curve_fit(self.quad, px, py, bounds=((0.0, -np.inf, -np.inf), (0.003, np.inf, np.inf)))
            poly = np.flip(popt, axis=0)

        elif(bnd == -1):
            x0 = [-0.001,1,10]            
            popt, pcov = scipy.optimize.curve_fit(self.quad, px, py, bounds=((-0.003, -np.inf, -np.inf), (0.0, np.inf, np.inf)))
            poly = np.flip(popt, axis=0)

        else:
            x0 = [-0.001,1,10]            
            popt, pcov = scipy.optimize.curve_fit(self.quad, px, py, bounds=((-0.003, -np.inf, -np.inf), (0.003, np.inf, np.inf)))
            poly = np.flip(popt, axis=0)
        

        draw_x = np.array(range(self.fi_point, self.st_point), dtype=np.int32)
        draw_y = np.array(polyval(draw_x, poly), dtype=np.int32)   # evaluate the polynomial
        
        draw_points = np.array(zip(draw_y, draw_x), dtype=np.int32)



        return poly, draw_points


    
    def tf_image(self, img, mode):

        src = np.array([[355, 204], [253, 205], [34, 335], [522, 326]], dtype=np.float32)
        dst = np.array([[70+122,0], [0+122, 0], [0+122,200], [70+122,200]], dtype=np.float32)

        if(mode == 'birdeye'):
            persp_tf = cv2.getPerspectiveTransform(src, dst)

        elif(mode == 'reverse'):
            persp_tf = cv2.getPerspectiveTransform(dst, src)

        else:
            print('Wrong Image Transform Mode')
            return False


        img = cv2.warpPerspective(img, persp_tf, (314, 250))       ### kalibrasyonndegerleri degistir class dan

        return img


    
    def get_target_points(self, src, poly_list, draw_points_list):

        pointsImg = cv2.cvtColor(src.copy(), cv2.COLOR_GRAY2BGR)

        start = [None, None]
        median = [None, None]
        finish = [None, None]
        for i in range(2):

            draw_points = draw_points_list[i]

            t_p_x = draw_points[:,0]
            t_p_y = draw_points[:,1]


            start[i] = (int(polyval(self.st_point, poly_list[i])), self.st_point)
            finish[i] = (int(polyval(self.fi_point, poly_list[i])), self.fi_point)
            median[i] = (int(polyval(self.md_point, poly_list[i])), self.md_point)

            cv2.circle(pointsImg, start[i], 5, (255,0,0), -1)
            cv2.circle(pointsImg, median[i], 5, (0,255,0), -1)
            cv2.circle(pointsImg, finish[i], 5, (0,0,255), -1)


        goal_list = [None, None, None]
        goal_list[0] = tuple(map(int, tuple(np.array(start).mean(axis=0))))
        goal_list[2] = tuple(map(int, tuple(np.array(finish).mean(axis=0))))
        goal_list[1] = tuple(map(int, tuple(np.array(median).mean(axis=0))))
        
        """
        cv2.circle(pointsImg, goal_list[0], 5, (255, 0, 0), -1)
        cv2.circle(pointsImg, goal_list[1], 5, (0, 255, 0), -1)
        cv2.circle(pointsImg, goal_list[2], 5, (0, 0, 255), -1)
        """
        goal_poly = polyfit(goal_list[:][0], goal_list[:][1], 2)

        deriv = polyder(goal_poly)


        path = []
        for goal in goal_list:        
            sl = polyval(goal[0], deriv)
            path.append([goal[0], goal[1], sl])

        
        """
        cv2.line(pointsImg, goal_list[0], goal_list[1], (255,0,0), 2) # plot line
        cv2.line(pointsImg, goal_list[1], goal_list[2], (100,100,0), 2) # plot line
        """

        return pointsImg, path





    def find_lane(self, lanes_raw, mode):

        # Process lines   
        success_p, success_n, pos_lines, neg_lines = self.process_lines(lanes_raw)


        if(mode == "left"):
            laneLines_img, blendedIm = self.plot_lane_lines(neg_lines)
        if(mode == "right"):
            laneLines_img, blendedIm = self.plot_lane_lines(pos_lines)
        else:
            pass
        
        

        kernel = np.ones((3,3),np.uint8)
        dilateImg = cv2.dilate(laneLines_img.copy(), kernel, iterations=2)
        

        contoursImg, laneContours = self.clear_by_contours(dilateImg.copy())

        
        tf_laneContours = self.tf_image(img=laneContours.copy(), mode='birdeye')


        return laneContours, tf_laneContours




    def process_image(self):

        # Grayscale one color channel
        grayIm = cv2.cvtColor(self.orig_img.copy(), cv2.COLOR_BGR2GRAY)

        # Create mask to only keep area defined by four coners
        # Black out every area outside area
        vertices = np.array([[(0,self.imshape[0]), (0, 230), (self.imshape[1], 230), (self.imshape[1],self.imshape[0])]], dtype=np.int32)
        # defining a blank mask to start with, 0s with same shape of edgesImg
        mask = np.zeros_like(grayIm.copy())   
        # fill pixels inside the polygon defined by vertices"with the fill color  
        color = 255
        mask = cv2.fillPoly(mask, vertices, color)
        # create image only where mask and edge Detection image are the same
        maskedImg = cv2.bitwise_and(grayIm.copy(), mask)

        # Use low pass filter to remove noise. Will remove high freq stuff like noise and edges
        # kernel_size specifies width/height of kernel, should be positive and odd
        # Also specify stand dev in X and Y direction, give zero to calculate from kernel size
        # Can also use average, median, and bilarteral blurring techniques


        #self.smoothedIm = cv2.GaussianBlur(self.maskedImg.copy(), (1, 1), 0)
        smoothedIm = maskedImg.copy()

        # Color Filter
        smoothedIm[np.where((smoothedIm>[120]))] = [120]

        # finds gradient in x,y direction, gradient direction is perpendicular to edges
        # checks pixels in gradient directions to see if they are local maximums, meaning on an edge
        # hysteresis thresholding has min and max value, edges with gradient intensity big enough are edges
        # edges that lie in bewteen are check to see if they connect to edges with intensity greater than max value, then it is considered edge
        # also assumes edges are long lines (not small pixels regions)

        high_thresh, thresh_im = cv2.threshold(smoothedIm.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = 0.5*high_thresh

        minVal = low_thresh # 60
        maxVal = high_thresh # 450


        edgesImg = cv2.Canny(smoothedIm.copy(), low_thresh, high_thresh)

        edgesImg[225:235] = 0


        smoothedEdge = cv2.GaussianBlur(edgesImg.copy(), (3, 3), 0)


        height, width = smoothedEdge.shape
        #   Left Mask
        center_t, center_b = ((width/2, 0), (width/2, height))

        l_vertices = np.array([[(0,0), (0, height), center_b, center_t]], dtype=np.int32)
        # create image only where mask and edge Detection image are the same
        left_img = cv2.bitwise_and(smoothedEdge.copy(), cv2.fillPoly(np.zeros_like(smoothedEdge.copy()) , l_vertices, 255))
        
        #   Right Mask
        r_vertices = np.array([[center_t, center_b, (width, height), (width, 0)]], dtype=np.int32)
        # create image only where mask and edge Detection image are the same
        right_img = cv2.bitwise_and(smoothedEdge.copy(), cv2.fillPoly(np.zeros_like(smoothedEdge.copy()) , r_vertices, 255))


        left_lines = self.hough_lines(left_img.copy())

        right_lines = self.hough_lines(right_img.copy())

        linesImg = self.draw_lines(self.orig_img.copy(), np.concatenate((left_lines, right_lines), axis=0))
        bw_linesImg = self.draw_lines(np.zeros_like(self.orig_img), np.concatenate((left_lines, right_lines), axis=0))


        left_laneContours, left_tf_img = self.find_lane(left_lines, "left")
        right_laneContours, right_tf_img = self.find_lane(right_lines, "right")
        




        tfimg_l = np.split(left_tf_img, 2, axis=1)[0]
        tfimg_r = np.split(right_tf_img, 2, axis=1)[1]
        tfimg = np.concatenate((tfimg_l, tfimg_r), axis=1)

        blendedImL = np.split(left_laneContours, 2, axis=1)
        blendedImR = np.split(right_laneContours, 2, axis=1)
        blendedIm = np.concatenate((blendedImL[0], blendedImR[1]), axis=1)






        l_px, l_py = self.get_c_points(left_tf_img.copy())
        r_px, r_py = self.get_c_points(right_tf_img.copy())

        # Sign and Confidence Check
        if(l_px > r_px):
            
            # Calculate Left
            left_poly, left_draw_points = self.get_poly(left_tf_img.copy(), l_px, l_py, bnd=None)

            right_poly, right_draw_points = self.get_poly(right_tf_img.copy(), r_px, r_py, bnd=np.sign(left_poly[-1]))

        elif(r_px > l_px):

            # Calculate Right
            right_poly, right_draw_points = self.get_poly(right_tf_img.copy(), r_px, r_py, bnd=None)

            left_poly, left_draw_points = self.get_poly(left_tf_img.copy(), l_px, l_py, bnd=np.sign(right_poly[-1]))

        else:
            raise Exception



        

        

                
        output_tf, path = self.get_target_points(tfimg.copy(), [right_poly, left_poly], [right_draw_points, left_draw_points])


        # plot output lines
        cv2.polylines(output_tf, [right_draw_points], False, color=(255, 0, 0), thickness = 2)  # args: image, points, closed, color
        cv2.polylines(output_tf, [left_draw_points], False, color=(0, 0, 255), thickness = 2)  # args: image, points, closed, color

        # plot blended img
        blendedImg_out = self.orig_img.copy()
        ind = np.where(blendedIm==[255])
        blendedImg_out[ind] = [0,0,255]
        
        
        mirror = False
        return blendedImg_out, output_tf, mirror, path

