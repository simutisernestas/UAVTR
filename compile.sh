#!/bin/bash

set -e

# DEBUG=1

mkdir -p src/detection/build
cd src/detection/build
# cmake -G Ninja .. 
ninja &

mkdir -p ../../estimation/build
cd ../../estimation/build
# cmake -G Ninja ..
ninja