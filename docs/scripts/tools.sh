convert Egent.png -quality 80 Agent.jpg

ros2 bag record /camera/rgb/image_raw /camera/depth/image_raw /camera/pose /camera_info /lidar/pointcloud /lidar/pose -o isaaclab_city

ros2 run image_view extract_images --ros-args -r image:=/udp_cam/image
