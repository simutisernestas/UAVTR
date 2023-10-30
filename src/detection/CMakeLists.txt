cmake_minimum_required(VERSION 3.22)
project(tracker)

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-Wall -Wextra -Wpedantic)

message("#####################")
message("##### MY CONFIG #####")
message("#####################")

#if (DEFINED ENV{DEBUG})
#    message("Debug mode")
#    set(CMAKE_BUILD_TYPE Debug)
#    add_compile_options(-g -O0)
#else()
#    message("Release mode")
#    set(CMAKE_BUILD_TYPE Release)
#endif()

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

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(onnxruntime/include)
link_directories(onnxruntime/lib)

add_library(tracker_lib src/tracker.cpp)
target_link_libraries(tracker_lib ${OpenCV_LIBS} onnxruntime)

add_executable(tracker_app src/main.cpp)
target_link_libraries(tracker_app ${OpenCV_LIBS} tracker_lib)

add_executable(yolo src/yolo.cpp)
target_link_libraries(yolo ${OpenCV_LIBS} onnxruntime)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(vision_msgs REQUIRED)

add_executable(tracking_ros_node src/ros_node.cpp)
target_link_libraries(tracking_ros_node ${OpenCV_LIBS} tracker_lib)
ament_target_dependencies(tracking_ros_node rclcpp sensor_msgs cv_bridge vision_msgs)

if (TEST)
    find_package(GTest REQUIRED)
    add_executable(test_tracker test/test_tracker.cpp)
    target_link_libraries(test_tracker ${OpenCV_LIBS} ${GTEST_BOTH_LIBRARIES} tracker_lib)
endif()
