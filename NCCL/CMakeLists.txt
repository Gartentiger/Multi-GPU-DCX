cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(nccl LANGUAGES CXX CUDA)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
option(USE_NCCL "Use NCCL" ON)

set(CMAKE_CUDA_ARCHITECTURES 60)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

#set(NCCL_LIBRARY "/usr/lib/x86_64-linux-gnu/libnccl.so")

add_executable(nccl testNccl.cu)

#target_link_libraries(nccl PRIVATE ${NCCL_LIBRARY})


include(FetchContent)
FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping.git
  GIT_TAG v0.1.2
)

FetchContent_MakeAvailable(kamping)
target_link_libraries(nccl PRIVATE kamping)


target_compile_options(kamping INTERFACE
   $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-std=gnu++2a>)

find_package(NCCL REQUIRED)
if (NCCL_FOUND)
    add_definitions(-DUSE_NCCL)
    include_directories(SYSTEM ${NCCL_INCLUDE_DIR})
endif ()
find_package(MPI REQUIRED)

if(NOT TARGET MPI::MPI_CXX)
    add_library(MPI::MPI_CXX IMPORTED INTERFACE)

    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")
    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
endif()

target_link_libraries(nccl PUBLIC MPI::MPI_CXX)

set(GEN_CODE_OPTIONS "--relocatable-device-code=false -gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60")

set(CMAKE_CUDA_FLAGS "--expt-extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations -g -O3 ${GEN_CODE_OPTIONS}")

target_include_directories(nccl PRIVATE build/_deps/cereal-src/include build/_deps/kamping-src/include build/_deps/kassert-src/include build/_deps/pfr-src/include)

