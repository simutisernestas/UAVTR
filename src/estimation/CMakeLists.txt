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
endif()

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

add_library(kalman_lib SHARED src/kalman.cpp)
target_include_directories(kalman_lib PUBLIC include ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(kalman_lib Eigen3)

add_library(estimator_lib SHARED src/estimator.cpp)
target_include_directories(estimator_lib PUBLIC include ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(estimator_lib Eigen3 OpenCV)

add_executable(estimation_node src/estimator_ros_node.cpp)
target_include_directories(estimation_node PUBLIC include ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(estimation_node rclcpp sensor_msgs cv_bridge vision_msgs Eigen3 tf2_ros geometry_msgs px4_msgs visualization_msgs)
target_link_libraries(estimation_node estimator_lib kalman_lib)

add_executable(orientation_filter src/orient_filter.cpp)
ament_target_dependencies(orientation_filter rclcpp sensor_msgs px4_msgs tf2_ros geometry_msgs Eigen3)
target_link_libraries(orientation_filter /home/ernie/thesis/track/Fusion/build/Fusion/libFusion.a)
target_include_directories(orientation_filter PUBLIC /home/ernie/thesis/track/Fusion/Fusion ${EIGEN3_INCLUDE_DIR})

if (TEST)
    find_package(GTest REQUIRED)
endif ()
