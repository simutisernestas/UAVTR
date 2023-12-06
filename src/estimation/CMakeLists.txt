cmake_minimum_required(VERSION 3.22)
project(estimator)

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-Wall -Wextra -Wpedantic)

message("#####################")
message("##### MY CONFIG #####")
message("#####################")

if (DEFINED ENV{DEBUG})
    message("Debug mode")
    set(CMAKE_BUILD_TYPE Debug)
    add_compile_options(-g -O0)
    add_compile_definitions("DEBUG=1")
endif ()

# enable/disable cassert NDEBUG
if (DEFINED ENV{DISABLE_ASSERT})
    message("Assert disabled")
    add_compile_options(-DNDEBUG)
else ()
    message("Assert enabled")
    add_compile_options(-UNDEBUG)
endif ()

option(FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." TRUE)
if (${FORCE_COLORED_OUTPUT})
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        add_compile_options(-fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options(-fcolor-diagnostics)
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
find_package(visualization_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(image_geometry REQUIRED)

### Estimator library
add_library(estimator_lib SHARED
        src/estimator.cpp
        src/kalman.cpp
)
target_include_directories(estimator_lib PRIVATE
        include
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(estimator_lib Eigen3 OpenCV)
target_precompile_headers(estimator_lib PRIVATE
        include/estimator.hpp include/kalman.hpp)

### Estimator node executable
add_executable(estimation_node
        src/estimator_ros_node.cpp
        src/estimator_ros.cpp
)
target_include_directories(estimation_node PRIVATE
        include
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)
ament_target_dependencies(estimation_node
        rclcpp
        sensor_msgs
        cv_bridge
        vision_msgs
        Eigen3
        tf2_ros
        geometry_msgs
        px4_msgs
        visualization_msgs
        image_geometry
        OpenCV
)
target_link_libraries(estimation_node estimator_lib)
target_precompile_headers(estimation_node PRIVATE
        include/estimator_ros.hpp)

get_filename_component(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
message("PROJECT_ROOT_DIR: ${PROJECT_ROOT_DIR}")

add_executable(orientation_filter src/orient_filter.cpp)
ament_target_dependencies(orientation_filter rclcpp sensor_msgs px4_msgs tf2_ros geometry_msgs Eigen3)
target_link_libraries(orientation_filter ${PROJECT_ROOT_DIR}/Fusion/build/Fusion/libFusion.a)
target_include_directories(orientation_filter PUBLIC
        ${PROJECT_ROOT_DIR}/Fusion/Fusion ${EIGEN3_INCLUDE_DIR})

# test directory
get_filename_component(TEST_SOURCE test/test_flow_velocity.cpp ABSOLUTE)
get_filename_component(TEST_PATH ${TEST_SOURCE} DIRECTORY)

find_package(GTest REQUIRED)
add_executable(flow_velocity_test test/test_flow_velocity.cpp)
include_directories(flow_velocity_test PUBLIC ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR} include ${EIGEN3_INCLUDE_DIR})
target_link_libraries(flow_velocity_test gtest gtest_main estimator_lib ${EIGEN3_LIBRARIES})
ament_target_dependencies(flow_velocity_test OpenCV)
target_compile_definitions(flow_velocity_test PRIVATE TEST_DIR="${TEST_PATH}")
