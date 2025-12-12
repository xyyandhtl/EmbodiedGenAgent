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

# LIO节点
# echo "🚀 启动 LIO 节点..."
ros2 launch lio_ros_wrapper lio_mid360.launch.py
# ros2 launch fast_lio mapping.launch.py
