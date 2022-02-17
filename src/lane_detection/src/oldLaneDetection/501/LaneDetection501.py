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



    def quad3(self, xdata, a, b, c):
    
        #b = -125*2*a
        return a * xdata**2 + b * xdata + c


    def quad2(self, params):

        t = time.time()

        a1, a2, b1, b2, c1, c2 = params

        npoly1 = [c1, b1, a1]
        npoly2 = [c2, b2, a2]

        x1 = [self.x1[i] for i in range(0, len(self.x1), 5)]
        x2 = [self.x2[i] for i in range(0, len(self.x2), 5)]
        y1 = [self.y1[i] for i in range(0, len(self.y1), 5)]
        y2 = [self.y2[i] for i in range(0, len(self.y2), 5)]

        y1p = polyval(x1, npoly1) 
        y2p = polyval(x2, npoly2)

        ny2p = [-x for x in y2p]
        mdist = np.mean([sum(dst) for dst in zip(y1p, ny2p)])

        err = []
        
        rng = list(range(50, 200, 10))
        y1pn = polyval(rng, npoly1) 
        y2pn = polyval(rng, npoly2) 
        
        aa = list(range(len(rng)+1, 1, -1))
        err = np.sum(np.abs(np.abs(y1pn-y2pn) - mdist)*aa)


        e_p1 = sum(abs(y1p-y1)/y1)*len(self.x1)
        e_p2 = sum(abs(y2p-y2)/y2)*len(self.x2)
        
        e_std = err / len(rng) 


        return  e_p1 + e_p2 + e_std
       

    def get_poly2(self, px1, py1, px2, py2):
        
        
        self.y1 = py1
        self.y2 = py2
        self.x1= px1
        self.x2= px2

        self.cc = 0

        poly1, pcov = scipy.optimize.curve_fit(self.quad3, px1, py1, bounds=((-0.003, -np.inf, -np.inf), (0.003, np.inf, np.inf)))
        poly1 = np.flip(poly1, axis=0)

        poly2, pcov = scipy.optimize.curve_fit(self.quad3, px2, py2, bounds=((-0.003, -np.inf, -np.inf), (0.003, np.inf, np.inf)))
        poly2 = np.flip(poly2, axis=0)


        self.poly1 = poly1
        self.poly2 = poly2

        minb = [poly1[-1]*0.5, poly2[-1]*0.5, poly1[-2]*0.5, poly2[-2]*0.5, poly1[-3]*0.01, poly2[-3]*0.01]
        maxb = [poly1[-1]*1.5, poly2[-1]*1.5, poly1[-2]*1.5, poly2[-2]*1.5, poly1[-3]*1.01, poly2[-3]*1.01]

        bnd = np.array([minb, maxb]).T
        bnd.sort(axis=1)

        
        x_0 = [poly1[-1], poly2[-1], poly1[-2], poly2[-2], poly1[-3], poly2[-3]]

        #options = {'maxfun' : max(100, 10*len(x_0))}
        options = {'maxiter' : 40}
        res = scipy.optimize.minimize(self.quad2, x0=x_0, method='BFGS', bounds=bnd)

        #res = ModuleTNC.minimize(self.quad2, x0=x_0, method='TNC', bounds=bnd, maxfun = max(100, 10*len(x_0)))

        print(res)
        
        print("\nright")
        print(poly1)
        poly1[-1] = res.x[0]
        poly1[-2] = res.x[2]
        poly1[-3] = res.x[4]
        

       
        print(poly1)
        
        print("\nleft")
        print(poly2)

        poly2[-1] = res.x[1]
        poly2[-2] = res.x[3]
        poly2[-3] = res.x[5]

   
        print(poly2)
        
        
        
        
        

        draw_x1 = np.array(range(self.fi_point, self.st_point), dtype=np.int32)
        draw_y1 = np.array(polyval(draw_x1, poly1), dtype=np.int32)   # evaluate the polynomial
        
        draw_points1 = np.array(zip(draw_y1, draw_x1), dtype=np.int32)

        
        draw_x2 = np.array(range(self.fi_point, self.st_point), dtype=np.int32)
        draw_y2 = np.array(polyval(draw_x2, poly2), dtype=np.int32)   # evaluate the polynomial
        
        draw_points2 = np.array(zip(draw_y2, draw_x2), dtype=np.int32)



        return poly1, draw_points1, poly2, draw_points2


    
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



        #2 Grayscale one color channel
        grayIm = cv2.cvtColor(self.orig_img.copy(), cv2.COLOR_BGR2GRAY)
        #3 Gaussian Blur
        smoothed = cv2.GaussianBlur(grayIm.copy(), (3, 3), 0)

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





        left_found = False
        right_found = False

        tfimg_l = np.split(np.zeros_like(self.tf_img), 2, axis=1)[0]
        blendedImL = np.split(np.zeros((self.imshape[0], self.imshape[1])), 2, axis=1)[0]


        tf_img_r = np.split(np.zeros_like(self.tf_img), 2, axis=1)[1]
        blendedImR = np.split(np.zeros((self.imshape[0], self.imshape[1])), 2, axis=1)[1]



        try:
            height, width = smoothedEdge.shape
            center_t, center_b = ((width/2, 0), (width/2, height))
            
            #   Left Mask
            l_vertices = np.array([[(0,0), (0, height), center_b, center_t]], dtype=np.int32)
            # create image only where mask and edge Detection image are the same
            left_img = cv2.bitwise_and(smoothedEdge.copy(), cv2.fillPoly(np.zeros_like(smoothedEdge.copy()) , l_vertices, 255))
            
            left_lines = self.hough_lines(left_img.copy())
            left_laneContours, left_tf_img = self.find_lane(left_lines, "left")

            tfimg_l = np.split(left_tf_img, 2, axis=1)[0]
            blendedImL = np.split(left_laneContours, 2, axis=1)[0]

            l_px, l_py = self.get_c_points(left_tf_img.copy())

            left_found = True
        
        except:
            left_found = False

            print("Could NOT find any LEFT LANE")
        
        
        
        try:
            #   Right Mask
            r_vertices = np.array([[center_t, center_b, (width, height), (width, 0)]], dtype=np.int32)
            # create image only where mask and edge Detection image are the same
            right_img = cv2.bitwise_and(smoothedEdge.copy(), cv2.fillPoly(np.zeros_like(smoothedEdge.copy()) , r_vertices, 255))

            right_lines = self.hough_lines(right_img.copy())
            right_laneContours, right_tf_img = self.find_lane(right_lines, "right")

            tfimg_r = np.split(right_tf_img, 2, axis=1)[1]
            blendedImR = np.split(right_laneContours, 2, axis=1)[1]

            r_px, r_py = self.get_c_points(right_tf_img.copy())

            right_found = True
        
        except:
            right_found = False

            print("Could NOT find any RIGHT LANE")

        

        # blend the right and left images

        tfimg = np.concatenate((tfimg_l, tfimg_r), axis=1)
        

        blendedIm = np.concatenate((blendedImL, blendedImR), axis=1)





        # BOTH LANES FOUND
        if(right_found and left_found):

            right_poly2, right_draw_points2, left_poly2, left_draw_points2 = self.get_poly2(r_px, r_py, l_px, l_py)

            """
            # Sign and Confidence Check
            if(l_px > r_px):
                
                # Calculate Left
                left_poly, left_draw_points = self.get_poly(l_px, l_py)

                right_poly, right_draw_points = self.get_poly(r_px, r_py)

            elif(r_px > l_px):

                # Calculate Right
                right_poly, right_draw_points = self.get_poly(r_px, r_py)

                left_poly, left_draw_points = self.get_poly(l_px, l_py)

            else:
                raise Exception
            

            output_tf, path = self.get_target_points(tfimg.copy(), [right_poly, left_poly], [right_draw_points, left_draw_points])


            # plot output lines
            cv2.polylines(output_tf, [right_draw_points], False, color=(255, 0, 0), thickness = 2)  # args: image, points, closed, color
            cv2.polylines(output_tf, [left_draw_points], False, color=(0, 0, 255), thickness = 2)  # args: image, points, closed, color

            """







            output_tf, path = self.get_target_points(tfimg.copy(), [right_poly2, left_poly2], [right_draw_points2, left_draw_points2])


            # plot output lines
            cv2.polylines(output_tf, [right_draw_points2], False, color=(255, 0, 0), thickness = 2)  # args: image, points, closed, color
            cv2.polylines(output_tf, [left_draw_points2], False, color=(0, 0, 255), thickness = 2)  # args: image, points, closed, color

            



        # ONE LANE FOUND
        elif(right_found or left_found):
            
            px = l_px if left_found else r_px
            py = l_py if left_found else r_py

            poly, draw_points = self.get_poly(tfimg.copy(), px, py, bnd=None)


            output_tf, path = self.get_target_points(tfimg.copy(), [poly], [draw_points])


            # plot output lines
            cv2.polylines(output_tf, [draw_points], False, color=(255, 0, 0), thickness = 2)  # args: image, points, closed, color


            

        

        # NO LANE FOUND
        else:

            print("None of the Lanes could be found!!")
            return blendedImg_out, output_tf, False, path


        
        # plot blended img
        blendedImg_out = self.orig_img.copy()
        ind = np.where(blendedIm==[255])
        blendedImg_out[ind] = [0,0,255]


        return blendedImg_out, output_tf, False, path

        

                
        
