cmake_minimum_required(VERSION 3.5)
project(unitree_ros2_example)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


include_directories(include include/common)
link_directories(src)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(unitree_go REQUIRED)
find_package(unitree_hg REQUIRED)
find_package(unitree_api REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)

set (
  DEPENDENCY_LIST
  unitree_go
  unitree_hg
  unitree_api
  rclcpp
  std_msgs
  rosbag2_cpp
)

#add_executable(record_bag src/record_bag.cpp)
add_executable(go2_sport_client src/go2/go2_sport_client.cpp src/common/ros2_sport_client.cpp)
target_compile_features(go2_sport_client PRIVATE cxx_std_20)

add_executable(go2_stand_example src/go2/go2_stand_example.cpp src/common/motor_crc.cpp)
target_compile_features(go2_stand_example PRIVATE cxx_std_20)

add_executable(go2_robot_state_client src/go2/go2_robot_state_client.cpp src/common/motor_crc.cpp) 
target_compile_features(go2_robot_state_client PRIVATE cxx_std_20)

add_executable(go2_vel_sport src/go2/go2_vel_sport.cpp src/common/ros2_sport_client.cpp)
target_compile_features(go2_vel_sport PRIVATE cxx_std_20)

add_executable(camera_vel_node src/camera_vel_node.cpp src/common/ros2_sport_client.cpp)
target_compile_features(camera_vel_node PRIVATE cxx_std_20)
target_link_libraries(camera_vel_node ${OpenCV_LIBS})

# Non-ROS
add_executable(camera_test src/camera.cpp)
target_compile_features(camera_test PRIVATE cxx_std_20)
target_link_libraries(camera_test ${OpenCV_LIBS})

#ament_target_dependencies(record_bag ${DEPENDENCY_LIST})
ament_target_dependencies(go2_sport_client ${DEPENDENCY_LIST})
ament_target_dependencies(go2_stand_example ${DEPENDENCY_LIST})
ament_target_dependencies(go2_robot_state_client ${DEPENDENCY_LIST})
ament_target_dependencies(go2_vel_sport ${DEPENDENCY_LIST})
ament_target_dependencies(camera_vel_node ${DEPENDENCY_LIST} sensor_msgs)

install(TARGETS 
  go2_vel_sport
  go2_sport_client
  go2_stand_example
  go2_robot_state_client
  camera_vel_node
  DESTINATION lib/${PROJECT_NAME})




if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
