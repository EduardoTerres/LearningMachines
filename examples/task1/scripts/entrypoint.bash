#!/usr/bin/env bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
source /root/catkin_ws/setup.bash

# Fix OpenCV TLS issue on ARM architectures
export OPENCV_IO_ENABLE_OPENEXR=0
export OPENCV_IO_ENABLE_OPENEXR=0

rosrun learning_machines train.py "$@"