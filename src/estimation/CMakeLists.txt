project(tracker)

cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-Wall -Wextra -Wpedantic)

message("#####################")
message("##### MY CONFIG #####")
message("#####################")

if (DEFINED ENV{DEBUG})
    message("Debug mode")
    set(CMAKE_BUILD_TYPE Debug)
    add_compile_options(-g -O0)
else()
    message("Release mode")
    set(CMAKE_BUILD_TYPE Release)
endif()

# enable/disable cassert NDEBUG
if (DEFINED ENV{ASSERT})
    message("Assert enabled")
    add_compile_options(-UNDEBUG)
else()
    message("Assert disabled")
    add_compile_options(-DNDEBUG)
endif()

message("#####################")

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(estimation_node src/main.cpp)
target_include_directories(estimation_node PUBLIC include ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(estimation_node rclcpp sensor_msgs cv_bridge vision_msgs Eigen3 tf2_ros geometry_msgs)

add_executable(filter_exe src/filter.cpp src/vo.cpp)
target_include_directories(filter_exe PUBLIC kalman/include ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(filter_exe rclcpp sensor_msgs cv_bridge Eigen3 tf2_ros geometry_msgs)
target_link_libraries(filter_exe ${OpenCV_LIBS})

if (TEST)
    find_package(GTest REQUIRED)
endif()
