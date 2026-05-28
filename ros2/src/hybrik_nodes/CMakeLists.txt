cmake_minimum_required(VERSION 3.8)
project(hybrik_nodes)

get_filename_component(HYBRIK_ROS2_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)

add_executable(human_skeleton_tf_publisher src/human_skeleton_tf_publisher.cpp)
ament_target_dependencies(
  human_skeleton_tf_publisher
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  tf2
)

install(
  TARGETS human_skeleton_tf_publisher
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/esp32_imu_reader.py
    scripts/hybrik_smpl_tf_publisher.py
    scripts/hybrik_udp_bridge.py
    scripts/live_gru_smpl_infer.py
    scripts/sensor_udp_streamer.py
    scripts/latest_camera_publisher.py
    scripts/kaipu_mujoco_sim.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
    launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY
    ${HYBRIK_ROS2_ROOT}/kaipu_model
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
