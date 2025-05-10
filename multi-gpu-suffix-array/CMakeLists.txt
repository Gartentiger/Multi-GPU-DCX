cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(suffix_array LANGUAGES CXX CUDA)

option(DGX1_TOPOLOGY "Handle DGX-1 topology" OFF)

set(CMAKE_CUDA_ARCHITECTURES 60)

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

target_compile_features(suffix_array PUBLIC cxx_std_11)

set(GEN_CODE_OPTIONS "--relocatable-device-code=false -gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60")

set(CMAKE_CUDA_FLAGS "--expt-extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations -g -O3 ${GEN_CODE_OPTIONS}")

target_include_directories(suffix_array PRIVATE deps src)

