cmake_minimum_required(VERSION 3.10)
project(Libcubwt LANGUAGES CXX CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 70)

add_executable(BenchmarkLibcub LibcubBenchmark.cu)

# include(FetchContent)
# FetchContent_Declare(
#   kamping
#   GIT_REPOSITORY https://github.com/kamping-site/kamping.git
#   GIT_TAG v0.1.2
# )
# 
# FetchContent_MakeAvailable(kamping)
# target_link_libraries(BenchmarkLibcub PRIVATE kamping)

add_subdirectory(external/kamping)
target_link_libraries(BenchmarkLibcub PRIVATE kamping)
target_compile_options(kamping INTERFACE
   $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-std=gnu++17>)

add_subdirectory(external/libcubwt)
target_link_libraries(BenchmarkLibcub PRIVATE libcubwt)


add_library(io STATIC io.cu)
target_link_libraries(BenchmarkLibcub PRIVATE io)