#!/bin/bash

set -e

mkdir -p src/detection/build
cd src/detection/build
# DEBUG=1 cmake ..
# cmake -G Ninja .. 
ninja &

mkdir -p ../../estimation/build
cd ../../estimation/build
# cmake -G Ninja ..
# DEBUG=1 cmake -G Ninja ..
ninja