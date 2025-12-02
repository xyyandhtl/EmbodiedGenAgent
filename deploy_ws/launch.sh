#!/bin/bash
# 运行 Unitree Go2 运动示例节点脚本
# Author: wei wang

# 获取当前脚本所在目录，防止路径错乱
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 切换到工作空间根目录
cd "$SCRIPT_DIR" || exit

# 检查 install 目录是否存在
if [ ! -d "install" ]; then
    echo "❌ 未找到 install 目录，请先执行: colcon build --symlink-install"
    exit 1
fi

# 加载 ROS 2 环境和工作空间
echo "🔧 正在加载 ROS 2 环境..."
source /opt/ros/humble/setup.bash
source install/setup.bash

# 启动节点
# echo "🚀 启动 go2_vel_sport 节点..."
# ros2 run unitree_ros2_example go2_vel_sport
echo "🚀 启动 camera_vel_node 节点..."
ros2 run unitree_ros2_example camera_vel_node \
    --ros-args \
    -p enable_speed_control:=true \
    -p camera_image_raw_topic:=/camera/rgb/image_raw \
    -p cmd_vel_topic:=/cmd_vel \
    -p max_linear_x:=1.0 \
    -p max_angular_z:=0.5 \
    -p enable_resize:=false \
    -p publish_compressed:=true

# 标定节点
# echo "🚀 启动 bag_to_pcd 节点..."
# ros2 launch livox_camera_calib bag_to_pcd.launch.py
# ros2 launch livox_camera_calib calib.launch.py
