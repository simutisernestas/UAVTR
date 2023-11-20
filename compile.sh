#!/bin/bash

set -e

# DEBUG=1

mkdir -p src/detection/build
cd src/detection/build
# cmake -G Ninja .. -D CMAKE_C_COMPILER=gcc-12 -D CMAKE_CXX_COMPILER=g++-12 -D CMAKE_BUILD_TYPE=Release
ninja &

mkdir -p ../../estimation/build
cd ../../estimation/build
# DEBUG=1 
# cmake -G Ninja .. -D CMAKE_C_COMPILER=gcc-12 -D CMAKE_CXX_COMPILER=g++-12 -D CMAKE_BUILD_TYPE=Release
ninja