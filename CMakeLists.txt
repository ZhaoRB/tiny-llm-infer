cmake_minimum_required(VERSION 3.22)
project(tiny-llm-infer LANGUAGES CXX CUDA)

# C++/CUDA Version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# cuda toolkits
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# cuda architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# 设置默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# 根据构建类型设置编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Debug 模式：启用调试信息，禁用优化，添加调试标志
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -DDEBUG")
    message(STATUS "Debug mode: Debugging enabled, optimizations disabled")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Release 模式：最大优化，禁用调试信息
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
    message(STATUS "Release mode: Maximum optimizations enabled")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    # Release with Debug Info：优化 + 调试信息 + 行号信息
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -lineinfo -DNDEBUG")
    message(STATUS "RelWithDebInfo mode: Optimizations + debug info + line info enabled")
endif()

# include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

function(add_cuda_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(${target_name} 
        CUDA::cudart 
        CUDA::cublas
        CUDA::cublasLt
        CUDA::curand
    )
endfunction()

# test targets (target_name:source_file)
set(TEST_TARGETS
    "sgemm_test:src/operators/sgemm/test.cu"
    "rmsnorm_test:src/operators/rmsnorm/test.cu"
    "layernorm_test:src/operators/layernorm/layernorm.cu"
    "reduce_test:src/operators/reduce/reduce.cu"
    # "sgemv_test:src/operators/sgemv/sgemv.cu"
    # "sgemv_cub_test:src/operators/sgemv/sgemv_cub.cu"
    "softmax_test:src/operators/softmax/softmax.cu"
    "transpose_test:src/operators/transpose/transpose.cu"
)

set(ALL_TARGETS)
foreach(TEST_TARGET ${TEST_TARGETS})
    string(REPLACE ":" ";" TARGET_INFO ${TEST_TARGET})
    list(GET TARGET_INFO 0 TARGET_NAME)
    list(GET TARGET_INFO 1 SOURCE_FILE)
    
    add_cuda_executable(${TARGET_NAME} ${SOURCE_FILE})
    list(APPEND ALL_TARGETS ${TARGET_NAME})
    
    message(STATUS "Added target: ${TARGET_NAME} -> ${SOURCE_FILE}")
endforeach()

set_target_properties(${ALL_TARGETS}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 打印构建信息
message(STATUS "=== Build Configuration ===")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "C++ Flags: ${CMAKE_CXX_FLAGS_DEBUG}")
    message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "C++ Flags: ${CMAKE_CXX_FLAGS_RELEASE}")
    message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "C++ Flags: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")
endif()
message(STATUS "Output Directory: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "=============================")
