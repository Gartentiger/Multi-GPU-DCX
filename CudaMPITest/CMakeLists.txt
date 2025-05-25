cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(Test LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 60)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_language(CUDA)

add_executable(Test CudaMPI.cu)

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

find_package(MPI REQUIRED)
target_link_libraries(Test INTERFACE MPI::MPI_CXX)
target_include_directories(Test PRIVATE ${MPI_CXX_INCLUDE_PATH})

find_package(CUDAToolkit REQUIRED)
target_link_libraries(Test PRIVATE CUDA::cudart)