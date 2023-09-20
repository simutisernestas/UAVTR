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
