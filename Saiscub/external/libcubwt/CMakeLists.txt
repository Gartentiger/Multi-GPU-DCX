cmake_minimum_required(VERSION 3.10)
project(Libcubwt LANGUAGES CXX CUDA)

add_library(libcubwt STATIC libcubwt.cu)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)