cmake_minimum_required(VERSION 3.10)
project(AickTensorrt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8-a+crc+crypto")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -arch=sm_87")
set(CMAKE_CUDA_ARCHITECTURES 87)

#查找依赖
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

find_library(LIB_NVINFER NAMES nvinfer)
find_library(LIB_NVONNX NAMES nvonnxparser)
find_library(LIB_NVPLUGIN NAMES nvinfer_plugin)

#头文件路径
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)
include_directories(/usr/include/aarch64-linux-gnu) 

file(GLOB SOURCES "src/*.cpp")

add_library(AickTensorrt SHARED 
    src/AickTensorrt.cpp 
    src/DetModel.cpp 
    src/dbscan.cpp
    src/MemoryState.cpp
)

target_link_libraries(AickTensorrt 
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${LIB_NVINFER}
    ${LIB_NVONNX} 
    ${LIB_NVPLUGIN}
    cudart
    m 
    pthread
    dl
)

set_target_properties(AickTensorrt PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)