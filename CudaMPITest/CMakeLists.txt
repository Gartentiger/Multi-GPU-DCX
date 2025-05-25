cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(Test LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 60)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(Test CudaMPI.cu)

target_link_libraries(Test PRIVATE CUDA::cudart)
target_link_libraries(Test INTERFACE MPI::MPI_CXX)

target_compile_options(kamping INTERFACE
   $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-std=gnu++17>)