#!/bin/bash

cd oguz-marc
source devel/setup.bash

rosparam delete /move_base
roslaunch racecar-nav move_base.launch
