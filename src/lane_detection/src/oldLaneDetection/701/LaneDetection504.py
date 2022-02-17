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
import rospy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class LaneDetection():

    def __init__(self):

        self.blended_img_pub = rospy.Publisher("/lane_detection/blended", Image, queue_size=1)
        self.tf_img_pub = rospy.Publisher("/lane_detection/tfimg", Image, queue_size=1)
        rospy.Subscriber("/lane_detection/processed", Image, self.img_callback, queue_size=1, buff_size=52428800)
        
        
        self.tf_img = np.zeros((250, 314), dtype="uint8")


        self.st_point = 200
        self.md_point = 165
        self.fi_point = 100



    def img_callback(self, msg):

        start = time.time()

        img = CvBridge().imgmsg_to_cv2(msg).copy()

        seq = msg.header.seq

        self.process_image(img.copy())

        t = time.time() - start
        f = 1/t
        print("LaneDection  time : {0}, frequency: {1}".format(t, f))
        


    def quad(self, xdata, a, b, c):
    
        return a * xdata**2 + b * xdata + c



    def get_poly(self, contours):

        poly_list = []
        draw_list = []
        
        contours = sorted(contours, key = cv2.contourArea, reverse = True)

        for cnt in contours[:3]:
            

            if cv2.contourArea(cnt) < 100:
                continue
            
            py = cnt[:,0,0]
            px = cnt[:,0,1]

            poly, pcov = scipy.optimize.curve_fit(self.quad, px, py, bounds=((-0.003, -np.inf, -np.inf), (0.003, np.inf, np.inf)))
            
            poly = np.flip(poly, axis=0)

            draw_x = np.array(range(self.fi_point, self.st_point), dtype=np.int32)
            draw_y = np.array(polyval(draw_x, poly), dtype=np.int32)   # evaluate the polynomial

            draw_points = np.array(zip(draw_y, draw_x), dtype=np.int32)

            poly_list.append(poly)
            draw_list.append(draw_points)
            perimeter = cv2.arcLength(cnt,True)

            #print(poly)
            #print(perimeter)
            
        if len(poly_list) >= 1:    
            return poly_list, draw_list, True
        else:
            return [], [], False


    
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
        orgin_dist = [None] * len(poly_list)
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

            orgin_dist[i] = abs((pointsImg.shape[1]/2) - int(polyval(self.st_point, poly_list[i])))



        # If the number of fitting poly is greater than 2 
        # Remove the point which have max origin dist from the list
        if(len(start) > 2):
            start.pop(orgin_dist.index(max(orgin_dist)))
            median.pop(orgin_dist.index(max(orgin_dist)))
            finish.pop(orgin_dist.index(max(orgin_dist)))
            
            goal_list = [None, None, None]
            goal_list[0] = tuple(map(int, tuple(np.array(start).mean(axis=0))))
            goal_list[2] = tuple(map(int, tuple(np.array(finish).mean(axis=0))))
            goal_list[1] = tuple(map(int, tuple(np.array(median).mean(axis=0))))

        # If the number of fitting poly is one 
        # Shift yazilmasi gerekiyor 
        elif(len(start) == 1):

            goal_list = [None, None, None]
            goal_list[0] = tuple(map(int, tuple(np.array(start).mean(axis=0))))
            goal_list[2] = tuple(map(int, tuple(np.array(finish).mean(axis=0))))
            goal_list[1] = tuple(map(int, tuple(np.array(median).mean(axis=0))))

        else:
            goal_list = [None, None, None]
            goal_list[0] = tuple(map(int, tuple(np.array(start).mean(axis=0))))
            goal_list[2] = tuple(map(int, tuple(np.array(finish).mean(axis=0))))
            goal_list[1] = tuple(map(int, tuple(np.array(median).mean(axis=0))))
    
        
        cv2.circle(pointsImg, goal_list[0], 5, (255, 0, 0), -1)
        cv2.circle(pointsImg, goal_list[1], 5, (0, 255, 0), -1)
        cv2.circle(pointsImg, goal_list[2], 5, (0, 0, 255), -1)


        cv2.line(pointsImg, goal_list[0], goal_list[1], (255,0,0), 8) # plot line
        cv2.line(pointsImg, goal_list[1], goal_list[2], (100,100,0), 8) # plot line
    
        goal_poly = polyfit(goal_list[:][0], goal_list[:][1], 1)

        deriv = polyder(goal_poly)


        xmax = self.tf_img.shape[0]
        ymax = self.tf_img.shape[1]

        path = []
        for goal in goal_list: 

            x = xmax - goal[1]
            y = goal[0] - (ymax/2)
            
            sl = polyval(goal[0], deriv)
            
            path.append([x, y, sl])


        return pointsImg, path, goal_list



    def process_image(self, pr_img):


        lane_found = False

        tf_laneContours = self.tf_image(img=pr_img.copy(), mode='birdeye')

        try:
            contours, _ = cv2.findContours(tf_laneContours.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        except:
            _, contours, _ = cv2.findContours(tf_laneContours.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        #ccc = cv2.drawContours(cv2.cvtColor(tf_laneContours.copy(), cv2.COLOR_GRAY2BGR), contours, -1, (255,0,0), 3) 
        
        poly_list, draw_list, lane_found = self.get_poly(contours)

        
        # LANE FOUND
        if(lane_found):
            
            tfimg = cv2.cvtColor(tf_laneContours.copy(), cv2.COLOR_GRAY2BGR)
            output_tf, path, goal_list = self.get_target_points(tfimg.copy(), poly_list, draw_list)
 
            for draw in draw_list:
                color = list(np.random.random(size=3) * 256)
                cv2.polylines(output_tf, [draw], False, color=color, thickness = 4)  # args: image, points, closed, color


            # plot blended im
            blendedImg_out = pr_img.copy()
            
            #ind = np.where(pr_img==[255])
            #blendedImg_out[ind] = (255,0,0)

            self.blended_img_pub.publish(CvBridge().cv2_to_imgmsg(blendedImg_out))
            self.tf_img_pub.publish(CvBridge().cv2_to_imgmsg(output_tf, "bgr8"))

            return blendedImg_out, output_tf, len(poly_list), path


        # NO LANE FOUND
        else:
            tfimg = cv2.cvtColor(tf_laneContours.copy(), cv2.COLOR_GRAY2BGR)
            pr_img = cv2.cvtColor(pr_img.copy(), cv2.COLOR_GRAY2BGR)

            self.blended_img_pub.publish(CvBridge().cv2_to_imgmsg(pr_img))
            self.tf_img_pub.publish(CvBridge().cv2_to_imgmsg(tfimg, "bgr8"))
            
            print("None of the Lanes could be found!!")
            return pr_img, tfimg, 0, []


        
rospy.init_node("lanede")
detector = LaneDetection()


rospy.spin()
        

                
        
