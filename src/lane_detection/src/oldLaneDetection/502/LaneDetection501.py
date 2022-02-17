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
import time
from numpy.polynomial.polynomial import polyfit, polyval, polyder
import scipy.optimize
import traceback





class LaneDetection():

    def __init__(self, img):
        
        # img[0] for img img[1] for img name
        self.orig_img = img[0]
        self.img_name = img[1]

        self.tf_img = np.zeros((250, 314), dtype="uint8")

        self.imshape = img[0].shape

        self.st_point = 200
        self.md_point = 125
        self.fi_point = 50
       

        
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

    



    def get_c_points(self, c_img):

        contours, _ = cv2.findContours(c_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        px = []
        py = []

        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)


        cnt = contours[max_index]
        
        for point in cnt:
            px.append(point[0][1])
            py.append(point[0][0])

        return px, py



    def quad(self, xdata, a, b, c):
    
        return a * xdata**2 + b * xdata + c



    def get_poly2(self, px1, py1, px2, py2):

        poly1, pcov = scipy.optimize.curve_fit(self.quad, px1, py1, bounds=((-0.003, -np.inf, -np.inf), (0.003, np.inf, np.inf)))
        poly1 = np.flip(poly1, axis=0)

        poly2, pcov = scipy.optimize.curve_fit(self.quad, px2, py2, bounds=((-0.003, -np.inf, -np.inf), (0.003, np.inf, np.inf)))
        poly2 = np.flip(poly2, axis=0)

        

        draw_x1 = np.array(range(self.fi_point, self.st_point), dtype=np.int32)
        draw_y1 = np.array(polyval(draw_x1, poly1), dtype=np.int32)   # evaluate the polynomial
        
        draw_points1 = np.array(zip(draw_y1, draw_x1), dtype=np.int32)

        
        draw_x2 = np.array(range(self.fi_point, self.st_point), dtype=np.int32)
        draw_y2 = np.array(polyval(draw_x2, poly2), dtype=np.int32)   # evaluate the polynomial
        
        draw_points2 = np.array(zip(draw_y2, draw_x2), dtype=np.int32)



        return poly1, draw_points1, poly2, draw_points2


    def get_poly1(self, px, py):

        poly, pcov = scipy.optimize.curve_fit(self.quad, px, py, bounds=((-0.003, -np.inf, -np.inf), (0.003, np.inf, np.inf)))
        poly = np.flip(poly, axis=0)

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

        start = [None] * len(poly_list)
        median = [None] * len(poly_list)
        finish = [None] * len(poly_list)
        for i in range(len(poly_list)):

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








    def process_image(self, pr_img, left_img, right_img):

        
        left_found = False
        right_found = False

        tfimg_l = np.split(np.zeros_like(self.tf_img), 2, axis=1)[0]
        tfimg_r = np.split(np.zeros_like(self.tf_img), 2, axis=1)[1]

        blendedImL = np.split(np.zeros(self.imshape), 2, axis=1)[0]
        blendedImR = np.split(np.zeros(self.imshape), 2, axis=1)[1]



        try:
            
            left_lines = self.hough_lines(left_img.copy())

            # Process lines   
            success_p, success_n, pos_lines, neg_lines = self.process_lines(left_lines)

            laneLines_img, blendedIm = self.plot_lane_lines(neg_lines)

            tf_laneContours = self.tf_image(img=laneLines_img.copy(), mode='birdeye')

            tfimg_l = np.split(tf_laneContours, 2, axis=1)[0]
            blendedImL = np.split(blendedIm, 2, axis=1)[0]

            l_px, l_py = self.get_c_points(tf_laneContours.copy())

            left_found = True
        
        
        except Exception as e:
            print("Could NOT find any LEFT LANE")
            print(traceback.format_exc())
            left_found = False

        
        
        
        try:
            
            right_lines = self.hough_lines(right_img.copy())
            
            # Process lines   
            success_p, success_n, pos_lines, neg_lines = self.process_lines(right_lines)


            laneLines_img, blendedIm = self.plot_lane_lines(pos_lines)

            tf_laneContours = self.tf_image(img=laneLines_img.copy(), mode='birdeye')


            tfimg_r = np.split(tf_laneContours, 2, axis=1)[1]
            blendedImR = np.split(blendedIm, 2, axis=1)[1]

            r_px, r_py = self.get_c_points(tf_laneContours.copy())

            right_found = True
        

        except Exception as e:
            print("Could NOT find any RIGHT LANE")
            print(traceback.format_exc())
            right_found = False
        

        # blend the right and left images

        tfimg = np.concatenate((tfimg_l, tfimg_r), axis=1)

        blendedIm = np.concatenate((blendedImL, blendedImR), axis=1)


        # BOTH LANES FOUND
        if(right_found and left_found):

            right_poly, right_draw_points, left_poly, left_draw_points = self.get_poly2(r_px, r_py, l_px, l_py)

            output_tf, path = self.get_target_points(tfimg.copy(), [right_poly, left_poly], [right_draw_points, left_draw_points])


            # plot output lines
            cv2.polylines(output_tf, [right_draw_points], False, color=(255, 0, 0), thickness = 2)  # args: image, points, closed, color
            cv2.polylines(output_tf, [left_draw_points], False, color=(0, 0, 255), thickness = 2)  # args: image, points, closed, color


        # ONE LANE FOUND
        elif(right_found):
            right_poly, right_draw_points = self.get_poly1(r_px, r_py)

            output_tf, path = self.get_target_points(tfimg.copy(), [right_poly], [right_draw_points])

            cv2.polylines(output_tf, [right_draw_points], False, color=(255, 0, 0), thickness = 2)  # args: image, points, closed, color

        elif(left_found):
            left_poly, left_draw_points = self.get_poly1(l_px, l_py)

            output_tf, path = self.get_target_points(tfimg.copy(), [left_poly], [left_draw_points])

            cv2.polylines(output_tf, [left_draw_points], False, color=(0, 0, 255), thickness = 2)  # args: image, points, closed, color

            
        

        # NO LANE FOUND
        else:

            print("None of the Lanes could be found!!")
            return blendedImg_out, output_tf, False, path


        
        # plot blended img
        blendedImg_out = self.orig_img.copy()
        ind = np.where(blendedIm==[255])
        blendedImg_out[ind] = 255


        right_lines = np.array(right_lines, dtype=np.int32)
        left_lines = np.array(left_lines, dtype=np.int32)

        ii = cv2.cvtColor(pr_img.copy(), cv2.COLOR_GRAY2BGR)

        for x in range(0, len(right_lines)):
            for x1,y1,x2,y2 in right_lines[x]:
                cv2.line(ii,(x1,y1),(x2,y2),(0,255,0),2)

        for x in range(0, len(left_lines)):
            for x1,y1,x2,y2 in left_lines[x]:
                cv2.line(ii,(x1,y1),(x2,y2),(0,255,0),2)
        #a
        #a = np.array(a, dtype=np.uint16)



        comb1 = np.concatenate((pr_img[200:], pr_img[200:]), axis=1)
        comb2 = np.concatenate((left_img[200:], right_img[200:]), axis=1)

        comb3 = np.concatenate((ii[200:], ii[200:]), axis=1)


        comb = np.concatenate((comb1, comb2), axis=0)


        cv2.imshow("cc", comb3)








        return blendedImg_out, output_tf, False, path

        

                
        
