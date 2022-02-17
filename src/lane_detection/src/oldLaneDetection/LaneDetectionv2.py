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

        self.grayIm = np.ones_like(img[0])*255
        self.smoothedIm = np.ones_like(img[0])*255
        self.edgesImg = np.ones_like(img[0])*255
        self.mask = np.ones_like(img[0])*255
        self.maskedImg = np.ones_like(img[0])*255
        self.allLines = np.ones_like(img[0])*255
        self.laneLines = np.ones_like(img[0])*255
        self.blendedIm = np.ones_like(img[0])*255
        self.perspImg = np.ones_like(img[0])*255
        self.dilateImg = np.ones_like(img[0])*255
        self.blendedIm2 = np.ones_like(img[0])*255
        self.contoursImg = np.ones_like(img[0])*255
        self.laneContours = np.ones_like(img[0])*255
        self.blendedIm3 = np.ones_like(img[0])*255
        

        self.img1 = np.ones_like(img[0])*255
        self.img2 = np.ones_like(img[0])*255
        self.img3 = np.ones_like(img[0])*255
        self.img4 = np.ones_like(img[0])*255
        self.img5 = np.ones_like(img[0])*255


        self.verbose = True


        #root = tk.Tk()
        #self.s_width = root.winfo_screenwidth()
        #self.s_height = root.winfo_screenheight()


    
    def view_images(self):

        #fig, ax = plt.subplots(nrows=5, ncols=4, figsize=(self.s_width/100., self.s_height/100.), dpi=100)
        fig, ax = plt.subplots(nrows=5, ncols=4, figsize=(10., 11.), dpi=100)

        plt.figure(1)

        #----------------ORIGINAL IMAGE-----------------
        ax[0,0].imshow(self.orig_img)
        ax[0,0].title.set_text("Original Image [ {} ]".format(self.img_name))
        ax[0,0].axis('off')

        # -------------GREYSCALE IMAGE-----------------
        ax[0,1].imshow(self.grayIm,cmap='gray')
        ax[0,1].title.set_text('Greyscaled image')
        ax[0,1].scatter([0],[self.imshape[0]])
        ax[0,1].scatter([0],[230])
        ax[0,1].scatter([self.imshape[1]], [230])
        ax[0,1].scatter([self.imshape[1]],[self.imshape[0]])
        ax[0,1].axis('off')

        #---------------------MASK---------------------
        #ax[0,2].imshow(self.mask,cmap='gray')
        #ax[0,2].title.set_text('Mask')

        #-------------------MASKED IMAGE---------------
        ax[0,2].imshow(self.maskedImg,cmap='gray')
        ax[0,2].title.set_text('Masked Image')
        ax[0,2].axis('off')

        #--------------GAUSSIAN SMOOTHING--------------
        ax[1,0].imshow(self.smoothedIm,cmap='gray')
        ax[1,0].title.set_text('Smoothed image')
        ax[1,0].axis('off')

        #----------------EDGE DETECTION----------------
        ax[1,1].imshow(self.edgesImg,cmap='gray')            
        ax[1,1].title.set_text('Edge Detection')
        ax[1,1].axis('off')

        #--------------GAUSSIAN SMOOTHING--------------
        ax[1,2].imshow(self.smoothedEdge,cmap='gray')
        ax[1,2].title.set_text('Smoothed Edge image')
        ax[1,2].axis('off')

        #-----------------ALL LINES FOUND--------------
        ax[2,0].imshow(self.allLines,cmap='gray')
        ax[2,0].title.set_text('All Hough Lines Found')
        ax[2,0].axis('off')

        #------------------LANE LINES------------------
        ax[2,1].imshow(self.laneLines,cmap='gray')
        ax[2,1].title.set_text('Lane Lines')
        ax[2,1].axis('off')

        #----------------FINAL OUTPUT------------------
        ax[2,2].imshow(self.blendedIm)
        ax[2,2].title.set_text('Final Output')
        ax[2,2].axis('off')

        #-------------------DILATE IMG-----------------
        ax[3,0].imshow(self.dilateImg,cmap='gray')
        ax[3,0].title.set_text('Dilate Image')
        ax[3,0].axis('off')

        #------------------LANE LINES------------------
        ax[3,1].imshow(self.perspImg,cmap='gray')
        ax[3,1].title.set_text('Perspective Image')
        ax[3,1].axis('off')

        #----------------FINAL OUTPUT------------------
        ax[3,2].imshow(self.blendedIm2)
        ax[3,2].title.set_text('Final Output')
        ax[3,2].axis('off')

        #----------------CONTOURS IMG------------------
        ax[4,0].imshow(self.contoursImg,cmap='gray')
        ax[4,0].title.set_text('Contours Image')
        ax[4,0].axis('off')

        #----------------Lane Contours------------------
        ax[4,1].imshow(self.laneContours,cmap='gray')
        ax[4,1].title.set_text('Lane Contours')
        ax[4,1].axis('off')

        #----------------FINAL OUTPUT-------------------
        ax[4,2].imshow(self.blendedIm3)
        ax[4,2].title.set_text('Final Output')
        ax[4,2].axis('off')





        ax[0,3].imshow(self.img1, cmap='gray')
        ax[0,3].title.set_text('Polynomial Image')
        ax[0,3].axis('off')

        ax[1,3].imshow(self.img2, cmap='gray')
        ax[1,3].title.set_text('Polynomial Image')
        ax[1,3].axis('off')

        ax[2,3].imshow(self.img3, cmap='gray')
        ax[2,3].title.set_text('Polynomial Image')
        ax[2,3].axis('off')

        ax[3,3].imshow(self.img4, cmap='gray')
        ax[3,3].title.set_text('Polynomial Image')
        ax[3,3].axis('off')

        ax[4,3].imshow(self.img5, cmap='gray')
        ax[4,3].title.set_text('Polynomial Image')
        ax[4,3].axis('off')


        

        
        plot_backend = plt.get_backend()     # toggle fullscreen mode
        mng = plt.get_current_fig_manager() 
        mng.resize(*mng.window.maxsize())

        if self.verbose:
            plt.show()
            

        else:
            fig.savefig("outputs/{0}".format(self.img_name), bbox_inches='tight')
            plt.close('all')
            

    
    def apply_mask(self, img):

        # Black out every area outside area
        vertices = np.array([[(0,self.imshape[0]), (0, 230), (self.imshape[1], 230), (self.imshape[1],self.imshape[0])]], dtype=np.int32)

        # defining a blank mask to start with, 0s with same shape of edgesImg
        mask = np.zeros_like(img.copy())   
                
        # fill pixels inside the polygon defined by vertices"with the fill color  
        color = 255
        mask = cv2.fillPoly(mask, vertices, color)

        # create image only where mask and edge Detection image are the same
        maskedImg = cv2.bitwise_and(img.copy(), mask)

        return mask, maskedImg

    
        
    def hough_lines(self, img):

        rho = 2 # distance resolution in pixels of the Hough grid
        theta = np.pi / 180 # angular resolution in radians of the Hough grid
        threshold = 45     # minimum number of votes (intersections in Hough grid cell)
        min_line_len = 20  #40 minimum number of pixels making up a line
        max_line_gap = 10    #100 maximum gap in pixels between connectable line segments
        lines = cv2.HoughLinesP(img.copy(), rho, theta, threshold, np.array([]), 
                                    minLineLength=min_line_len, maxLineGap=max_line_gap)


        # Check if we got more than 1 line
        if lines is not None and len(lines) > 2:
            # Draw all lines onto image
            allLines = np.zeros_like(img)
            for i in range(len(lines)):
                for x1,y1,x2,y2 in lines[i]:
                    cv2.line(allLines,(x1,y1),(x2,y2),(255,255,0),2) # plot line   

        return lines, allLines

    
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

                        if slope > 0 and abs(theta) < 88 and abs(theta) > 10: 

                            slopePositiveLines.append([x1,y1,x2,y2,-slope, lineLength]) # add positive slope line
                            addedPos = True # note that we added a positive slope line

                        if slope < 0 and abs(theta) < 88 and abs(theta) > 10:
                            
                            slopeNegativeLines.append([x1,y1,x2,y2,-slope, lineLength]) # add negative slope line
                            addedNeg = True # note that we added a negative slope line
        
        
        return addedPos, addedNeg, slopePositiveLines, slopeNegativeLines



    def plot_lane_lines(self, lines):
        
        # make new black image
        colorLines = self.orig_img.copy()
        #laneLines = np.array(np.zeros(self.maskedImg.shape), dtype='int')
        laneLines = np.zeros_like(self.laneLines.copy())*255

        # Draw lines onto image
        line_list = np.array(lines, dtype='int')

        for x1,y1,x2,y2,sl,length in line_list:
            cv2.line(laneLines,(x1,y1),(x2,y2),(255,255,255),2) # plot line   
            cv2.line(colorLines,(x1,y1),(x2,y2),(255,0,0),3) # plot line

        laneLines = cv2.cvtColor(laneLines, cv2.COLOR_BGR2GRAY)

        return laneLines, colorLines

    

    def clear_by_contours(self, c_img):

        contoursImg = c_img.astype(np.uint8)

        _, contours, _ = cv2.findContours(contoursImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index1 = np.argmax(areas)

        max_index2 = np.argmax([0 if i==max_index1 else areas[i] for i in range(len(areas))])

        cnt1 = contours[max_index1]
        cnt2 = contours[max_index2]

        laneContours = c_img

        for i in range(len(contours)):

            x,y,w,h = cv2.boundingRect(contours[i])
            cv2.rectangle(contoursImg,(x,y),(x+w,y+h),255,2)

            if(i != max_index1 and i != max_index2):
                x,y,w,h = cv2.boundingRect(contours[i])
                laneContours = cv2.rectangle(laneContours,(x,y),(x+w,y+h),0,-1)

            else:
                x,y,w,h = cv2.boundingRect(contours[i])
                cv2.rectangle(contoursImg,(x,y),(x+w,y+h),255,4)



        return contoursImg, laneContours
        

    
    def get_poly(self, src_img, dst_img):

        polyImg = dst_img.copy() if(len(dst_img.shape) == 3) else cv2.cvtColor(dst_img.copy(), cv2.COLOR_GRAY2BGR) 

        c_img = src_img.copy() if(len(dst_img.shape) == 2) else cv2.cvtColor(src_img.copy(), cv2.COLOR_BGR2GRAY) 


        _, contours, _ = cv2.findContours(c_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        poly_list = []
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

                if polyval(extr, poly) >= polyImg.shape[0] or extr < 0 or extr > polyImg.shape[1] or counter > 100:
                    x_hit = True
                    continue

                else:
                    
                    draw_x = np.append(draw_x, extr)
                    draw_y = np.append(draw_y, polyval(extr, poly))   # evaluate the polynomial


                counter += 5
            

            draw_x = np.array(draw_x, dtype=np.int32)
            draw_y = np.array(draw_y, dtype=np.int32)



            draw_points = np.array(zip(draw_y, draw_x), dtype=np.int32)

            poly_list.append([poly, draw_points])

            polyImg = cv2.polylines(polyImg.copy(), [draw_points], False, color=(255,0,0), thickness = 2)  # args: image, points, closed, color


        return poly_list, polyImg


    
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

    


    def get_target_points(self, src, poly_list):


        pointsImg = cv2.cvtColor(src.copy(), cv2.COLOR_GRAY2BGR)

        start = []
        median = []
        finish = []
        for poly in poly_list:

            draw_points = poly[1]
            poly = poly[0]

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





    def process_image(self):

        # Grayscale one color channel
        self.grayIm = cv2.cvtColor(self.orig_img.copy(), cv2.COLOR_BGR2GRAY)

        # Create mask to only keep area defined by four coners
        self.mask, self.maskedImg = self.apply_mask(self.grayIm.copy())

        # Use low pass filter to remove noise. Will remove high freq stuff like noise and edges
        # kernel_size specifies width/height of kernel, should be positive and odd
        # Also specify stand dev in X and Y direction, give zero to calculate from kernel size
        # Can also use average, median, and bilarteral blurring techniques

        #self.smoothedIm = cv2.GaussianBlur(self.maskedImg.copy(), (1, 1), 0)

        self.smoothedIm = self.maskedImg.copy()

        self.smoothedIm[np.where((self.smoothedIm>[120]))] = [120]

        # finds gradient in x,y direction, gradient direction is perpendicular to edges
        # checks pixels in gradient directions to see if they are local maximums, meaning on an edge
        # hysteresis thresholding has min and max value, edges with gradient intensity big enough are edges
        # edges that lie in bewteen are check to see if they connect to edges with intensity greater than max value, then it is considered edge
        # also assumes edges are long lines (not small pixels regions)

        high_thresh, thresh_im = cv2.threshold(self.smoothedIm.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = 0.5*high_thresh

        minVal = low_thresh # 60
        maxVal = high_thresh # 450

        self.edgesImg = cv2.Canny(self.smoothedIm.copy(), low_thresh, high_thresh)

        self.edgesImg[225:235] = 0


        self.smoothedEdge = cv2.GaussianBlur(self.edgesImg.copy(), (3, 3), 0)


        # Apply Hough Lines Method
        
        lines, self.allLines = self.hough_lines(self.smoothedEdge.copy())


        if lines is not None:
            # Process lines   
            success_p, success_n, pos_lines, neg_lines = self.process_lines(lines)

            if success_p or success_n:
                
                if success_p and not success_n:
                    # Plot processed lines and blended image
                    line_list = pos_lines
                elif success_n and not success_p:
                    # Plot processed lines and blended image
                    line_list = neg_lines
                elif success_p and success_n:
                    line_list = np.concatenate((pos_lines, neg_lines), axis=0)

                self.laneLines, self.blendedIm = self.plot_lane_lines(line_list)

                kernel = np.ones((3,3),np.uint8)
                self.dilateImg = cv2.dilate(self.laneLines.copy(), kernel, iterations=2)


                blendedIm2 = self.orig_img.copy()
                ind = np.where(self.dilateImg==[255])
                blendedIm2[ind] = [255, 0, 0]
                self.blendedIm2 = blendedIm2

                self.contoursImg, self.laneContours = self.clear_by_contours(self.dilateImg.copy())

                blendedIm3 = self.orig_img.copy()
                ind = np.where(self.laneContours==[255])
                blendedIm3[ind] = [255, 0, 0]
                self.blendedIm3 = blendedIm3


                self.img1 = self.tf_image(img=self.grayIm.copy(), mode='birdeye')

                self.img2 = self.tf_image(img=self.laneContours.copy(), mode='birdeye')


                poly_list, self.img3 = self.get_poly(src_img=self.img2.copy(), dst_img=self.img2.copy())


                
                self.img4, path = self.get_target_points(self.img2.copy(), poly_list)


            else:
                print('Couldn`t find any lane')

            
        self.view_images()

        return self.img4, path

