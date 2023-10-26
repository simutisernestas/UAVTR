cmake_minimum_required(VERSION 3.22)
project(tracker)

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

option (FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." TRUE)
if (${FORCE_COLORED_OUTPUT})
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
       add_compile_options (-fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
       add_compile_options (-fcolor-diagnostics)
    endif ()
endif ()

message("#####################")

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(px4_msgs REQUIRED)

include_directories(src/estimation/include)

add_library(kalman_lib SHARED src/estimation/src/kalman.cpp)
target_include_directories(kalman_lib PUBLIC src/estimation/include ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(kalman_lib Eigen3)

add_library(estimator_lib SHARED src/estimation/src/estimator.cpp)
target_include_directories(estimator_lib PUBLIC src/estimation/include ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(estimator_lib Eigen3 OpenCV)

add_executable(estimation_node src/estimation/src/ros_node.cpp)
target_include_directories(estimation_node PUBLIC src/estimation/include ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(estimation_node rclcpp sensor_msgs cv_bridge vision_msgs Eigen3 tf2_ros geometry_msgs px4_msgs)
target_link_libraries(estimation_node estimator_lib kalman_lib)

add_executable(imu_repub_node src/estimation/src/imu_repub.cpp)
ament_target_dependencies(imu_repub_node rclcpp sensor_msgs px4_msgs)

include_directories(src/detection/onnxruntime/include)

add_library(tracker_lib src/detection/src/tracker.cpp)
target_link_directories(tracker_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/detection/onnxruntime/lib)
target_link_libraries(tracker_lib ${OpenCV_LIBS} onnxruntime)

add_executable(tracker_app src/detection/src/main.cpp)
target_link_libraries(tracker_app ${OpenCV_LIBS} tracker_lib)

add_executable(yolo src/detection/src/yolo.cpp)
target_link_directories(yolo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/detection/onnxruntime/lib)
target_link_libraries(yolo ${OpenCV_LIBS} onnxruntime)

add_executable(tracking_ros_node src/detection/src/ros_node.cpp)
target_link_libraries(tracking_ros_node ${OpenCV_LIBS} tracker_lib)
ament_target_dependencies(tracking_ros_node rclcpp sensor_msgs cv_bridge vision_msgs)

if (TEST)
    find_package(GTest REQUIRED)

#    find_package(GTest REQUIRED)
#    add_executable(test_tracker test/test_tracker.cpp)
#    target_link_libraries(test_tracker ${OpenCV_LIBS} ${GTEST_BOTH_LIBRARIES} tracker_lib)
endif()
