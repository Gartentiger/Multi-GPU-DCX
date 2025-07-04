cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(suffix_array LANGUAGES CXX CUDA)

option(DGX1_TOPOLOGY "Handle DGX-1 topology" OFF)

set(CMAKE_CUDA_ARCHITECTURES 60)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)

add_executable(suffix_array 
                src/util.h
                src/util.cpp
                src/suffix_types.h
                src/suffix_array.cu
                src/prefix_doubling.hpp
#                src/test_merges.cu
#                src/distrib_merge/test_distrib_merge.cu

                src/suffix_array_kernels.cuh
                src/suffix_array_kernels.cu
                src/suffix_array_templated_kernels.cuh
                src/suffixarraymemorymanager.hpp
		src/cuda_helpers.h
                src/io.cu
                src/io.cuh
                src/merge_copy_detour_guide.hpp
                src/performancerecorder.hpp
                src/suffixarrayperformancemeasurements.hpp
                src/stages.h
                src/my_mgpu_context.hxx
                src/qdallocator.hpp

                src/gossip/multisplit.cuh
                src/gossip/all_to_all.cuh
                src/gossip/all_to_all_dgx1.cuh
                src/gossip/context.cuh
                src/gossip/auxiliary.cuh

                src/multisplit/api/wms_api.h
                src/multisplit/config/config_wms.h
                src/multisplit/kernels/wms/wms_postscan.cuh
                src/multisplit/kernels/wms/wms_postscan_pairs.cuh
                src/multisplit/kernels/wms/wms_prescan.cuh
                src/multisplit/dispatch_multisplit.cuh

                src/remerge/remergemanager.hpp
                src/remerge/merge_types.hpp
                src/remerge/mergeprocessor.hpp
                src/remerge/mergenodeutils.hpp
                src/remerge/remerge_gpu_topology_helper.hpp
                src/remerge/two_way_micromerge.hpp
                src/remerge/multi_way_partitioning_search.hpp
                src/remerge/multi_way_micromerge.hpp
                src/remerge/multi_way_micromerge_on_one_node.hpp

                src/distrib_merge/distrib_merge_array.hpp
                src/distrib_merge/distrib_merge.hpp
                src/distrib_merge/distrib_merge_topology_helper.hpp
                )

if(DGX1_TOPOLOGY)
    target_compile_definitions(suffix_array PRIVATE DGX1_TOPOLOGY)
endif(DGX1_TOPOLOGY)

include(FetchContent)
FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping.git
  GIT_TAG v0.1.2
)

FetchContent_MakeAvailable(kamping)
target_link_libraries(suffix_array PRIVATE kamping)


target_compile_options(kamping INTERFACE
   $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-std=gnu++2a>)


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

target_link_libraries(suffix_array PUBLIC MPI::MPI_CXX)

set(GEN_CODE_OPTIONS "--relocatable-device-code=false -gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60")

set(CMAKE_CUDA_FLAGS "--expt-extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations -g -O3 ${GEN_CODE_OPTIONS}")

target_include_directories(suffix_array PRIVATE deps src build/_deps/kamping-src/include build/_deps/kassert-src/include build/_deps/pfr-src/include)

