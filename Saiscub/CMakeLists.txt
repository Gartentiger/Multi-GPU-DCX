cmake_minimum_required(VERSION 3.10)
project(Saiscub LANGUAGES CXX CUDA C)

option(LIBSAIS "Libsais" OFF)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 70)

add_executable(Saiscub Saiscub.cu)

include(FetchContent)
FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping.git
  GIT_TAG v0.1.2
)

FetchContent_MakeAvailable(kamping)
target_link_libraries(Saiscub PRIVATE kamping)

if(LIBSAIS)
	include(FetchContent)
	FetchContent_Declare(
	  libsais
	  GIT_REPOSITORY https://github.com/IlyaGrebnov/libsais.git
	  GIT_TAG v2.10.1
	)

	FetchContent_MakeAvailable(libsais)
	target_link_libraries(Saiscub PRIVATE libsais)
else()
	add_compile_definitions(libcub)
	add_subdirectory(external/libcubwt)
	target_link_libraries(Saiscub PRIVATE libcubwt)
endif()

target_compile_options(kamping INTERFACE
   $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-std=gnu++17>)

add_library(io STATIC io.cu)
target_link_libraries(Saiscub PRIVATE io)