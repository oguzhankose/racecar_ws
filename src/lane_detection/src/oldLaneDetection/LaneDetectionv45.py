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

import Tkinter as tk # use tkinter 

class LaneDetection():

    def __init__(self, img):
        
        # img[0] for img img[1] for img name
        self.orig_img = img[0]
        self.img_name = img[1]

        self.imshape = img[0].shape
       

    

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
        

    
    def get_poly(self, src_img):


        c_img = src_img.copy()


        contours, _ = cv2.findContours(c_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:

            px = []
            py = []

            for point in cnt:
                
                px.append(point[0][1])
                py.append(point[0][0])


            poly = polyfit(px, py, 2) if(len(cnt) > 60) else polyfit(px, py, 1)            

            draw_x = px
            
            draw_y = polyval(draw_x, poly)   # evaluate the polynomial


            x_hit = False
            counter = 1
            
            while not x_hit:

                extr = min(draw_x)-counter

                if polyval(extr, poly) >= c_img.shape[0] or extr < 0 or extr > c_img.shape[1] or counter > 100:
                    x_hit = True
                    continue

                else:
                    
                    draw_x = np.append(draw_x, extr)
                    draw_y = np.append(draw_y, polyval(extr, poly))   # evaluate the polynomial


                counter += 5
            

            draw_x = np.array(draw_x, dtype=np.int32)
            draw_y = np.array(draw_y, dtype=np.int32)

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

        start = []
        median = []
        finish = []


        for i in range(len(poly_list)):

            draw_points = draw_points_list[i]
            poly = poly_list[i]

            t_p_x = draw_points[:,0]
            t_p_y = draw_points[:,1]


            s = (int(np.mean(t_p_x[np.where(t_p_y==max(t_p_y))])), max(t_p_y))
            f = (int(np.mean(t_p_x[np.where(t_p_y==min(t_p_y))])), min(t_p_y))
            m = ((s[0]+f[0])/2, (s[1]+f[1])/2)

            cv2.circle(pointsImg, s, 5, (255,0,0), -1)
            cv2.circle(pointsImg, m, 5, (0,255,0), -1)
            cv2.circle(pointsImg, f, 5, (0,0,255), -1)

            start.append(s)
            median.append(m)
            finish.append(f)

            
        st = tuple(map(int, tuple(np.array(start).mean(axis=0))))
        fi = tuple(map(int, tuple(np.array(finish).mean(axis=0))))
        me = tuple(map(int, tuple(np.array(median).mean(axis=0))))
        
        cv2.circle(pointsImg, st, 5, (255, 0, 0), -1)
        cv2.circle(pointsImg, me, 5, (0, 255, 0), -1)
        cv2.circle(pointsImg, fi, 5, (0, 0, 255), -1)

        poly2 = polyfit([st[0], me[0], fi[0]], [st[1], me[1], fi[1]], 2)

        der = polyder(poly2)

        sl_st = polyval(st[0], der)
        sl_me = polyval(me[0], der)
        sl_fi = polyval(fi[0], der)


        path = []
        path.append([st[0], st[1], sl_st])
        path.append([me[0], me[1], sl_me])
        path.append([fi[0], fi[1], sl_fi])

        cv2.line(pointsImg, st, me, (255,0,0), 2) # plot line
        cv2.line(pointsImg, me, fi, (100,100,0), 2) # plot line


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

        
        ## ????
        """
        blendedIm2 = self.orig_img.copy()
        ind = np.where(self.dilateImg==[255])
        blendedIm2[ind] = [255, 0, 0]
        self.blendedIm2 = blendedIm2
        """
        

        contoursImg, laneContours = self.clear_by_contours(dilateImg.copy())

        ## ????
        """
        blendedIm3 = self.orig_img.copy()
        ind = np.where(self.laneContours==[255])
        blendedIm3[ind] = [255, 0, 0]
        """

        
        #self.img1 = self.tf_image(img=self.grayIm.copy(), mode='birdeye')


        tf_laneContours = self.tf_image(img=laneContours.copy(), mode='birdeye')

        lane_poly, lane_draw_points = self.get_poly(tf_laneContours.copy())


        return lane_poly, lane_draw_points, laneContours, tf_laneContours




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


        left_poly, left_draw_points, left_laneContours, left_tf_img = self.find_lane(left_lines, "left")
        right_poly, right_draw_points, right_laneContours, right_tf_img = self.find_lane(right_lines, "right")



        if(len(right_draw_points) > len(left_draw_points)*2.5):
            left_draw_points = right_draw_points.copy()
            left_draw_points[:,0] = np.add(314*np.ones_like(right_draw_points[:,0]), -right_draw_points[:,0])
            left_draw_points[:,1] = right_draw_points[:,1]

        elif(len(left_draw_points) > len(right_draw_points)*2.5):
            right_draw_points = left_draw_points.copy()
            right_draw_points[:,1] = np.add(314*np.ones_like(left_draw_points[:,0]), -left_draw_points[:,0])
            right_draw_points[:,0] = left_draw_points[:,1]

        else:
            pass


        tfimg_l = np.split(left_tf_img, 2, axis=1)[0]
        tfimg_r = np.split(right_tf_img, 2, axis=1)[1]
        tfimg = np.concatenate((tfimg_l, tfimg_r), axis=1)

        blendedImL = np.split(left_laneContours, 2, axis=1)
        blendedImR = np.split(right_laneContours, 2, axis=1)
        blendedIm = np.concatenate((blendedImL[0], blendedImR[1]), axis=1)

                
        output_tf, path = self.get_target_points(tfimg.copy(), [right_poly, left_poly], [right_draw_points, left_draw_points])

        # plot output lines
        cv2.polylines(output_tf, [right_draw_points], False, color=(255, 0, 0), thickness = 2)  # args: image, points, closed, color
        cv2.polylines(output_tf, [left_draw_points], False, color=(0, 0, 255), thickness = 2)  # args: image, points, closed, color

        # plot blended img
        blendedImg_out = self.orig_img.copy()
        ind = np.where(blendedIm==[255])
        blendedImg_out[ind] = [0,0,255]
        
        

        return blendedImg_out, output_tf, path

