#!/bin/bash
# 启动 Unitree Go2 运动控制节点

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit

echo "🔧 正在加载 ROS 2 环境..."
source /opt/ros/humble/setup.bash
source /home/unitree/ww/docker_ubuntu22/nav2/install/setup.sh
source install/setup.bash

echo "🚀 启动 mid360 ..."
ros2 launch livox_ros_driver2 msg_MID360_launch.py
