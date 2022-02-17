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

import matplotlib.pyplot as plt 


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
        threshold = 20     # minimum number of votes (intersections in Hough grid cell)
        min_line_len = 20  #40 minimum number of pixels making up a line
        max_line_gap = 50    #100 maximum gap in pixels between connectable line segments
        lines = cv2.HoughLinesP(img.copy(), rho, theta, threshold, np.array([]), 
                                    minLineLength=min_line_len, maxLineGap=max_line_gap)

        return lines

    

    def plot_lane_lines(self, lines):
        
        # make new black image
        colorLines = self.orig_img.copy()
        #laneLines = np.array(np.zeros(self.maskedImg.shape), dtype='int')
        laneLines = np.zeros_like(self.orig_img.copy())*255

        # Draw lines onto image
        line_list = np.array(lines, dtype='int')


        for line in line_list:
            for x1,y1,x2,y2 in line:
                cv2.line(laneLines,(x1,y1),(x2,y2),(255,255,255),2) # plot line   
                cv2.line(colorLines,(x1,y1),(x2,y2),(255,0,0),3) # plot line

        laneLines = cv2.cvtColor(laneLines, cv2.COLOR_BGR2GRAY)

        return laneLines, colorLines




    def get_c_points(self, c_img):

        contours, _ = cv2.findContours(c_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        px = []
        py = []

        
        for cnt in contours:
            for point in cnt:
                px.append(point[0][1])
                py.append(point[0][0])

        return px, py



    def quad(self, xdata, a, b, c):
    
        return a * xdata**2 + b * xdata + c



    def get_poly(self, px, py):

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

        pointsImg = src

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
        
        
        cv2.circle(pointsImg, goal_list[0], 5, (255, 0, 0), -1)
        cv2.circle(pointsImg, goal_list[1], 5, (0, 255, 0), -1)
        cv2.circle(pointsImg, goal_list[2], 5, (0, 0, 255), -1)


        cv2.line(pointsImg, goal_list[0], goal_list[1], (255,0,0), 2) # plot line
        cv2.line(pointsImg, goal_list[1], goal_list[2], (100,100,0), 2) # plot line
        
        goal_poly = polyfit(goal_list[:][0], goal_list[:][1], 2)

        deriv = polyder(goal_poly)


        xmax = self.tf_img.shape[0]
        ymax = self.tf_img.shape[1]

        path = []
        for goal in goal_list: 

            x = xmax - goal[1]
            y = goal[0] - (ymax/2)
            
            sl = polyval(goal[0], deriv)
            
            path.append([x, y, sl])


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

            laneLines_img, blendedIm = self.plot_lane_lines(left_lines)

            tf_laneContours = self.tf_image(img=laneLines_img.copy(), mode='birdeye')

            tfimg_l = np.split(tf_laneContours, 2, axis=1)[0]
            blendedImL = np.split(blendedIm, 2, axis=1)[0]

            l_px, l_py = self.get_c_points(tf_laneContours.copy())

            left_poly, left_draw_points = self.get_poly(l_px, l_py)

            left_found = True
        
        
        except Exception as e:
            print("Could NOT find any LEFT LANE")
            print(traceback.format_exc())
            left_found = False

        
        
        try:
            
            right_lines = self.hough_lines(right_img.copy())

            laneLines_img, blendedIm = self.plot_lane_lines(right_lines)

            tf_laneContours = self.tf_image(img=laneLines_img.copy(), mode='birdeye')

            tfimg_r = np.split(tf_laneContours, 2, axis=1)[1]
            blendedImR = np.split(blendedIm, 2, axis=1)[1]

            r_px, r_py = self.get_c_points(tf_laneContours.copy())

            right_poly, right_draw_points = self.get_poly(r_px, r_py)
            
            right_found = True
        

        except Exception as e:
            print("Could NOT find any RIGHT LANE")
            print(traceback.format_exc())
            right_found = False
        

        # blend the right and left images

        tfimg = np.concatenate((tfimg_l, tfimg_r), axis=1)
        tfimg = cv2.cvtColor(tfimg, cv2.COLOR_GRAY2BGR)

        blendedIm = np.concatenate((blendedImL, blendedImR), axis=1)



        # BOTH LANES FOUND
        if(right_found and left_found):
            
            output_tf, path = self.get_target_points(tfimg.copy(), [right_poly, left_poly], [right_draw_points, left_draw_points])

            cv2.polylines(tfimg, [right_draw_points], False, color=(255, 0, 0), thickness = 2)  # args: image, points, closed, color
            cv2.polylines(tfimg, [left_draw_points], False, color=(0, 0, 255), thickness = 2)  # args: image, points, closed, color




            ii = cv2.cvtColor(pr_img.copy(), cv2.COLOR_GRAY2BGR)
            ###############
            right_lines = np.array(right_lines, dtype=np.int32)
            for x in range(0, len(right_lines)):
                for x1,y1,x2,y2 in right_lines[x]:
                    cv2.line(ii,(x1,y1),(x2,y2),(x*5,x*75-200,x*55-50),2)
            ###############
            left_lines = np.array(left_lines, dtype=np.int32)
            for x in range(0, len(left_lines)):
                for x1,y1,x2,y2 in left_lines[x]:
                    color = list(np.random.random(size=3) * 256)
                    cv2.line(ii,(x1,y1),(x2,y2),color,2)
            


        # ONE LANE FOUND
        elif(left_found):
            output_tf, path = self.get_target_points(tfimg.copy(), [left_poly], [left_draw_points])

            cv2.polylines(tfimg, [left_draw_points], False, color=(0, 0, 255), thickness = 2)  # args: image, points, closed, color


            ii = cv2.cvtColor(pr_img.copy(), cv2.COLOR_GRAY2BGR)
            ###############
            left_lines = np.array(left_lines, dtype=np.int32)
            for x in range(0, len(left_lines)):
                for x1,y1,x2,y2 in left_lines[x]:
                    color = list(np.random.random(size=3) * 256)
                    cv2.line(ii,(x1,y1),(x2,y2),color,2)


        elif(right_found):
            
            output_tf, path = self.get_target_points(tfimg.copy(), [right_poly], [right_draw_points])

            cv2.polylines(tfimg, [right_draw_points], False, color=(255, 0, 0), thickness = 2)  # args: image, points, closed, color


            ii = cv2.cvtColor(pr_img.copy(), cv2.COLOR_GRAY2BGR)
            ###############
            right_lines = np.array(right_lines, dtype=np.int32)
            for x in range(0, len(right_lines)):
                for x1,y1,x2,y2 in right_lines[x]:
                    color = list(np.random.random(size=3) * 256)
                    cv2.line(ii,(x1,y1),(x2,y2),color,2)
            
            
        

        # NO LANE FOUND
        else:

            print("None of the Lanes could be found!!")
            return blendedIm, tfimg, False, []


        # plot blended img
        blendedImg_out = self.orig_img.copy()
        ind = np.where(blendedIm==[255])
        blendedImg_out[ind] = 255



        
        



        
        

        return blendedImg_out, output_tf, False, path

        

                
        
