cmake_minimum_required(VERSION 3.18)
project(Libcubwt LANGUAGES CXX CUDA)

set(CUDA_LIBS
    io.cu
    libcubwt.cu
)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_library(cubwtIo STATIC ${CUDA_LIBS})

add_executable(BenchmarkLibcub LibcubBenchmark.cu)

target_link_libraries(BenchmarkLibcub PRIVATE cubwtIo)

target_compile_features(BenchmarkLibcub PRIVATE cxx_std_17)