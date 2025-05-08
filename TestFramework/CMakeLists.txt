cmake_minimum_required(VERSION 3.10)
project(Libcubwt LANGUAGES CXX CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(BenchmarkLibcub LibcubBenchmark.cu)

include(FetchContent)
FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping.git
  GIT_TAG v0.1.2
)

FetchContent_MakeAvailable(kamping)
target_link_libraries(BenchmarkLibcub PRIVATE kamping)

#target_compile_options(BenchmarkLibcub PRIVATE "--DKAMPING_ASSERTION_LEVEL=0")
#include(FetchContent)
#FetchContent_Declare(
#  libcubwt
#  GIT_REPOSITORY https://github.com/IlyaGrebnov/libcubwt.git
#  GIT_TAG v1.0.0
#)

#FetchContent_MakeAvailable(libcubwt)
#target_link_libraries(BenchmarkLibcub PRIVATE libcubwt)


add_subdirectory(external/libcubwt-1.0.0)
target_link_libraries(BenchmarkLibcub PRIVATE libcubwt)


add_library(io STATIC io.cu)
target_link_libraries(BenchmarkLibcub PRIVATE io)