#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_py as tf2

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, Image
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

import ros_numpy
from ros_numpy import point_cloud2 as pc2
from cv_bridge import CvBridge

import cv2
import numpy as np

import time

class TransformPointCloud():
    def __init__(self, name, topic):
        try:
            self.source_name = name
            self.offset_lookup_time = 0
            self.timeout = 2
            self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(5))
            self.tl = tf2_ros.TransformListener(self.tf_buffer)

            self.source_frame = rospy.wait_for_message(topic, PointCloud2, self.timeout).header.frame_id
            self.target_frame = "base_link"
            self.pub = rospy.Publisher("{}/output".format(self.source_name), PointCloud2, queue_size=10)
            self.sub = rospy.Subscriber(topic, PointCloud2,
                                        self.point_cloud_callback, queue_size=10)
            
            self.pcl_arr = np.zeros([0, 0], np.uint8)

        except Exception as e:
            print(e)
            

    def point_cloud_callback(self, msg):

        lookup_time = msg.header.stamp + rospy.Duration(self.offset_lookup_time)
        target_frame = msg.header.frame_id if self.target_frame == "" else self.target_frame
        source_frame = msg.header.frame_id if self.source_frame == "" else self.source_frame
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, source_frame, lookup_time,
                                                    rospy.Duration(self.timeout))
        except tf2.LookupException as ex:
            rospy.logwarn(str(lookup_time.to_sec()))
            rospy.logwarn(ex)
            return
        except tf2.ExtrapolationException as ex:
            rospy.logwarn(str(lookup_time.to_sec()))
            rospy.logwarn(ex)
            return
        cloud_out = do_transform_cloud(msg, trans)

        self.pub.publish(cloud_out)
        self.pcl_arr = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(cloud_out, remove_nans=True)    # Remove Nans



class GlobalMap():
    def __init__(self, width=400, height=400, max_dist=19.99):
        self.map_width = width
        self.map_height = height
        self.max_distance = max_dist
        self.scale_factor = height / (2*int(max_dist))      # pixel per meter
        self.gmap = np.zeros([self.map_height, self.map_width, 3], np.uint8)
        self.gmap_zed = np.zeros([self.map_height, self.map_width, 3], np.uint8)       #########################

        self.gmap_pub = rospy.Publisher("/global_map/gmap", Image, queue_size=10)

        #self.new_gmap = np.zeros([self.map_height, self.map_width, 3], np.uint8)        ##########################33



    def update_gmap(self, pcl_list):
        #start = time.time()
        self.gmap = np.zeros([self.map_height, self.map_width, 3], np.uint8)
        self.gmap.fill(255)
        self.gmap_zed = np.zeros([self.map_height, self.map_width, 3], np.uint8)                   #######################
        self.gmap_zed.fill(255)
        #print("Time elapsed: {0}\nCODE: {1}\n-----------".format(time.time()-start, "SLM1"))
        #start = time.time()
        for pcl in pcl_list:
            #print(pcl.source_name, len(pcl.pcl_arr))
            for point in pcl.pcl_arr:

                point[0] = max(min(point[0], self.max_distance), -self.max_distance)
                point[1] = max(min(point[1], self.max_distance), -self.max_distance)

                if(point[0] > 0.15):
                    if(point[2] > 0.15):     # Height 
                        continue
                    #elif(point[2] >= 0 and point[2] <= 0.3):
                    else:
                        pix_x = int(self.map_height/2 - point[0]*self.scale_factor)    
                        pix_y = int(self.map_width/2 - point[1]*self.scale_factor)
                        self.gmap[pix_x, pix_y] = (255,0,0) if (pcl.source_name == "zed") else (0,0,255)

                        if(pcl.source_name == "zed"):
                            self.gmap_zed[pix_x, pix_y] = (0,0,0)
                """
                elif(point[2] > 0.01 and point[2] <= 0.1):
                    c2 += 1
                    pix_x = int(self.map_height/2 - point[0]*self.scale_factor)    
                    pix_y = int(self.map_width/2 - point[1]*self.scale_factor)
                    self.gmap[pix_x, pix_y] = (0,0,255) if (pcl.source_name == "zed") else (0,0,255)
                elif(point[2] < 0.01):
                    c3 += 1
                    pix_x = int(self.map_height/2 - point[0]*self.scale_factor)    
                    pix_y = int(self.map_width/2 - point[1]*self.scale_factor)
                    self.gmap[pix_x, pix_y] = (0,255,0) if (pcl.source_name == "zed") else (0,0,255)
                """
            cv2.circle(self.gmap, center=(200,200), radius=3, color=(0,0,0), thickness=-1, lineType=8, shift=0)

        #print("Time elapsed: {0}\nCODE: {1}\n-----------".format(time.time()-start, "SLM2"))
        #start = time.time()
        
        #cv2.imshow("gmap", cv2.resize(self.gmap, (800,800)))
        #cv2.waitKey(1)
        
        #cv2.imshow("gmap_zed", cv2.resize(self.gmap_zed, (800,800)))
        #cv2.waitKey(1)
        self.gmap_pub.publish(CvBridge().cv2_to_imgmsg(self.gmap, "bgr8"))
        
        """
        print("Time elapsed: {0}\nCODE: {1}\n-----------".format(time.time()-start, "SLM3"))
        start = time.time()
        """
        #self.manipulate_map(self.gmap_zed)
        """
        print("Time elapsed: {0}\nCODE: {1}\n-----------".format(time.time()-start, "SLM4"))
        start = time.time()
        """
        #self.clear_map()
        #self.deneme2()
        
    def deneme2(self):

        points = np.transpose(np.nonzero(np.all(self.gmap_zed == (0,0,0), axis=2)))
        
        for point in points:
            
            new = (point[0]+1, point[1]+1)
            #print(self.gmap_zed[new[0], new[1]])
            if([0,0,0] == new):
                print("asjknd")


    
    def clear_map(self):

        p = (255,0,0)
        
        points = np.transpose(np.nonzero(np.all(self.gmap == p, axis=2)))
        #print(len(points))

        new_map = self.gmap.copy()
        new_map.fill(255)
        slope_list = []
        for point in points:

            new_sl = self.slope_of_point(point)
            
            if not (new_sl in slope_list):
                slope_list.append(new_sl)
                new_map[point[0], point[1]] = (255,0,0)

        cv2.imshow("denee", new_map)
        cv2.waitKey(1)
                

                
    def slope_of_point(self, pixel):

        if(abs(self.map_width / 2 - pixel[1]) == 0):
            return -1
        else:
            return abs(self.map_height / 2 - pixel[0]) / abs(self.map_width / 2 - pixel[1])

     


    def manipulate_map(self, frame):

        gray_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        #cv2.imshow('gray_frame', gray_frame)
        #cv2.waitKey(1)

        ret, threshold = cv2.threshold(gray_frame.copy(), 250, 255, cv2.THRESH_BINARY)
        cv2.bitwise_not(threshold, threshold)

        #cv2.imshow('threshold', threshold)
        #cv2.waitKey(1)

        _, contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   
        c_frame = cv2.drawContours(frame.copy(), contours, -1, (0,0,0), 1)

        #self.draw_grid(c_frame)

        resized_map = cv2.resize(c_frame, (800,800)) 
        cv2.imshow('c_frame', resized_map)
        cv2.waitKey(1)

    """
    def draw_grid(self, img, line_color=(30, 40, 50), thickness=1, type_=cv2.LINE_AA):
        
        pxstep = self.scale_factor*5

        x = pxstep
        y = pxstep
        while x < img.shape[1]:
            cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
            x += pxstep

        while y < img.shape[0]:
            cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
            y += pxstep
    """






if __name__ == '__main__':

    rospy.init_node('pointcloud_to_map')

    zed_tf_pcl = TransformPointCloud(name="zed", topic="/pcl/zed_voxel_grid/output")
    lidar_tf_pcl = TransformPointCloud(name="lidar", topic="/pcl/lidar_voxel_grid/output")
    
    rospy.spin()
    """
    _map = GlobalMap()

    rate = rospy.Rate(20)
    
    while not rospy.is_shutdown():
        
        start = time.time()

        _map.update_gmap(pcl_list=[zed_tf_pcl, lidar_tf_pcl])
        #_map.update_gmap(pcl_list=[zed_tf_pcl])

        rate.sleep()

        print("Time elapsed: {0}\nLoop Freq: {1}\n-----------".format(time.time()-start, 1/(time.time()-start)))
    """
    