#!/bin/bash

set -e

cd src/detection/build
make -j8 &

cd ../../estimation/build
# DEBUG=1 cmake ..
ninja