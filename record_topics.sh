#!/bin/bash

cd /home/nvidia/oguz-marc/src/racecar-nav/records

rosbag record -o teee --split --size=1024 \
/ackermann_cmd_mux/active \
/ackermann_cmd_mux/input/default \
/ackermann_cmd_mux/input/navigation \
/ackermann_cmd_mux/input/teleop \
/ackermann_cmd_mux/output \
/cmd_vel \
/imu/data \
/imu/mag \
/zed/zed_node/right/image_rect_color \
/lidar/output \
/move_base/TebLocalPlannerROS/global_plan \
/move_base/TebLocalPlannerROS/local_plan \
/move_base/TebLocalPlannerROS/teb_markers \
/move_base/TebLocalPlannerROS/teb_poses \
/move_base/global_costmap/costmap \
/move_base/local_costmap/costmap \
/move_base/result \
/move_base/status




