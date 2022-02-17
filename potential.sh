#!/bin/bash


rosparam delete /local_costmap
rosparam delete /global_costmap

roslaunch racecar-potential potential.launch
