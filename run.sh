#!/bin/bash

set -e

# build
BUILD_TYPE="Release"

rm -rf build 
mkdir -p build 
cmake -B build -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
cmake --build build -j 4

# run
# ./build/bin/sgemm_test