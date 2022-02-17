#!/bin/bash

LIDAR=/dev/ttyUSB0
VESC=/dev/ttyACM0
IMU=/dev/ttyACM1

cd oguz-marc

echo 1 | sudo -S echo

if [ ! -c "$LIDAR" ]; then
	echo "Device not found: LIDAR -> $LIDAR"
	sudo chmod 777 $LIDAR
else
	sudo chmod 777 $LIDAR
fi


if [ ! -c "$VESC" ]; then
	echo "Device not found:VESC not found -> $VESC"
	sudo chmod 777 $VESC
else 
	sudo chmod 777 $VESC
fi


if [ ! -c "$IMU" ]; then
	echo "Device not found: IMU -> $IMU"
	sudo chmod 777 $IMU
else
	sudo chmod 777 $IMU
fi


source devel/setup.bash

roslaunch racecar-nav racecar.launch




