cmake_minimum_required(VERSION 3.15)
project(main CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-g -O3")
set(CMAKE_CUDA_FLAGS "--extended-lambda")


add_executable(main main.cpp kernel.cuh kernel.cu)
