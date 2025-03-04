cmake_minimum_required(VERSION 3.22)
#project(CTC_decoder)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
    message(STATUS "CMAKE_BUILD_TYPE = Debug")
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
    message(STATUS "CMAKE_BUILD_TYPE = Release")
endif ()

#set(CMAKE_C_COMPILER "/usr/bin/x86_64-linux-gnu-gcc-11")
#set(CMAKE_CXX_COMPILER "/usr/bin/x86_64-linux-gnu-g++-11")

set(CMAKE_CXX_STANDARD 17)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.4)
set(CUDNN_INCLUDE_DIR /usr/local/cuda-12.4/include)
set(CUDNN_LIBRARY /usr/local/cuda-12.4/lib64)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)
set(CMAKE_PREFIX_PATH /home/zyserver/miniconda3/envs/Torch2.0/lib/python3.10/site-packages/torch/share/cmake)
find_package(Torch REQUIRED)
# find_package(ZLIB REQUIRED)
# find_package(spdlog REQUIRED)


add_library(ctc_decode SHARED ctc_decode.cpp)

target_link_libraries(
        ctc_decode
        ${TORCH_LIBRARIES}
)