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

add_executable(estimation_node src/main.cpp)
target_include_directories(estimation_node PUBLIC include ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(estimation_node rclcpp sensor_msgs cv_bridge vision_msgs Eigen3 tf2_ros geometry_msgs px4_msgs)

if (TEST)
    find_package(GTest REQUIRED)
endif()
