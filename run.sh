#!/bin/bash

set -e

# Default build type
BUILD_TYPE="Release"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -b, --build-type TYPE    Set build type (Release, Debug, or Profiling). Default: Release"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate build type
if [[ "$BUILD_TYPE" != "Release" && "$BUILD_TYPE" != "Debug" && "$BUILD_TYPE" != "Profiling" ]]; then
    echo "Error: BUILD_TYPE must be 'Release', 'Debug', or 'Profiling', got: $BUILD_TYPE"
    exit 1
fi


# build
echo "Build type: $BUILD_TYPE"
rm -rf build 
mkdir -p build 
cmake -B build -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
cmake --build build -j 8

# # run matmul
# touch matmul.log && ./build/bin/matmul_test
# if [[ "$BUILD_TYPE" == "Profiling" ]]; then
#     ncu -f -o matmul \
#         --set full \
#         --section SpeedOfLight \
#         --section SpeedOfLight_RooflineChart \
#         --section SpeedOfLight_HierarchicalSingleRooflineChart \
#         ./build/bin/matmul_test
# fi

# run rmsnorm 
touch rmsnorm.log && ./build/bin/rmsnorm_test
if [[ "$BUILD_TYPE" == "Profiling" ]]; then
    ncu -f -o rmsnorm \
        --set full \
        --section SpeedOfLight \
        --section SpeedOfLight_RooflineChart \
        --section SpeedOfLight_HierarchicalSingleRooflineChart \
        ./build/bin/rmsnorm_test
fi