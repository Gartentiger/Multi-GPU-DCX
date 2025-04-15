#include <iostream>
#include <array>
#include "remerge/remergemanager.hpp"
#include "remerge/remerge_gpu_topology_helper.hpp"
#include "cuda_helpers.h"
#include <cuda_runtime.h>
#include <algorithm>
#include "cub/cub.cuh"

//#define PROFILING

#ifdef PROFILING
#undef TIMERSTART
#undef TIMERSTOP
#define TIMERSTART(x)
#define TIMERSTOP(x)
#endif

#define INCLUDE_VALUES 1

#ifdef DGX1_TOPOLOGY
template <size_t NUM_GPUS, class mtypes>
using TopologyHelper = crossGPUReMerge::DGX1TopologyHelper<NUM_GPUS, mtypes>;
#else
template <size_t NUM_GPUS, class mtypes>
using TopologyHelper = crossGPUReMerge::MergeGPUAllConnectedTopologyHelper<NUM_GPUS, mtypes>;
#endif

const size_t SIZE = 80*1024*1024;
const size_t NUM_GPUS = 8;

using Context = MultiGPUContext<NUM_GPUS>;

static const constexpr size_t NUM_ARRAYS = INCLUDE_VALUES ? 6 : 3;

struct MergeTestArrays {
    std::array<sa_index_t*, NUM_ARRAYS> Data;
    size_t len;
};


template<size_t SIZE>
void generate_test_keys(sa_index_t* target) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::iota(target, target+SIZE, 1);
    std::shuffle(target, target+SIZE, g);
}

template<size_t SIZE>
bool check(const sa_index_t* data) {
    for (uint i = 0; i < SIZE; ++i) {
        if(data[i] != i+1) {
            std::cerr << "\nFailed at position " << i << "\n";
            return false;
        }
    }
    std::cerr << "\nCheck okay." << "\n";
    return true;
}

void dump_array(const sa_index_t* array, size_t size) {
    std::copy(array, array+size, std::ostream_iterator<int>(std::cout," ,")
                 );
    std::cout << "\n";
}

template <typename T>
struct LessComp : public std::binary_function<T, T,  bool> {
    __host__ __device__ __forceinline__ bool operator()(const T& a, const T& b) const {
        return a < b;
    }
};

// Expects keys in array 1
template <size_t SIZE>
inline void sort_keys_node(Context& context, const MergeTestArrays& arrays, uint gpu, size_t temp_storage_bytes,
                           uint& out_array_index) {
    cudaSetDevice(context.get_device_id(gpu));

    size_t temp_storage_bytes_req = 0;

    cub::DoubleBuffer<sa_index_t> key_dbuff(arrays.Data[1], arrays.Data[0]);

    cudaError_t err = cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes_req, key_dbuff, SIZE,
                                                     0, sizeof(sa_index_t)*8,
                                                     context.get_gpu_default_stream(gpu));
    CUERR_CHECK(err);


//        std::cout << "\nNeed " <<   temp_storage_bytes << " temporary storage bytes for sort "<< "\n";

    ASSERT(temp_storage_bytes_req <= temp_storage_bytes);

    err = cub::DeviceRadixSort::SortKeys(arrays.Data[2], temp_storage_bytes, key_dbuff, SIZE,
                                         0, sizeof(sa_index_t)*8,
                                         context.get_gpu_default_stream(gpu));
    CUERR_CHECK(err);
    out_array_index = key_dbuff.Current() == arrays.Data[1] ? 1 : 0;
}

// Expects keys in 1, values in 4. Will return index of key array after sort and value will follow.
template <size_t SIZE>
inline void sort_kv_node(Context& context, const MergeTestArrays& arrays, uint gpu, size_t temp_storage_bytes,
                         uint& out_array_index) {
    cudaSetDevice(context.get_device_id(gpu));

    cub::DoubleBuffer<sa_index_t> key_dbuff(arrays.Data[1], arrays.Data[0]);
    cub::DoubleBuffer<sa_index_t> value_dbuff(arrays.Data[4], arrays.Data[3]);

    size_t temp_storage_bytes_req = 0;

    cudaError_t err = cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes_req, key_dbuff, value_dbuff,
                                                     SIZE, 0, sizeof(sa_index_t)*8,
                                                     context.get_gpu_default_stream(gpu));
    CUERR_CHECK(err);


//        std::cout << "\nNeed " <<   temp_storage_bytes << " temporary storage bytes for sort "<< "\n";

    ASSERT(temp_storage_bytes_req <= temp_storage_bytes);

    err = cub::DeviceRadixSort::SortPairs(arrays.Data[5], temp_storage_bytes, key_dbuff, value_dbuff, SIZE,
                                         0, sizeof(sa_index_t)*8,
                                         context.get_gpu_default_stream(gpu));
    CUERR_CHECK(err);
    out_array_index = key_dbuff.Current() == arrays.Data[1] ? 1 : 0;
}


template <size_t NUM_GPUS, size_t SIZE, bool include_values>
void sort_per_node(Context& context, std::array<MergeTestArrays, NUM_GPUS>& arrays, size_t temp_storage_bytes,
                   uint& out_keys_array_index) {
    for (uint gpu = 0; gpu < NUM_GPUS; ++gpu) {
        if (include_values)
            sort_kv_node<SIZE>(context, arrays[gpu], gpu, temp_storage_bytes, out_keys_array_index);
        else
            sort_keys_node<SIZE>(context, arrays[gpu], gpu, temp_storage_bytes, out_keys_array_index);
    }

    context.sync_default_streams();
}

template <size_t NUM_GPUS, size_t SIZE>
void copy_up(Context& context, const std::array<MergeTestArrays, NUM_GPUS>& arrays,
             sa_index_t* test_data, sa_index_t* test_data2, uint array_index, uint array_index2) {
    for (uint i = 0; i < NUM_GPUS; ++i) {
        cudaSetDevice(context.get_device_id(i));
        cudaMemcpyAsync(arrays[i].Data[array_index], test_data+i*SIZE, sizeof(sa_index_t) * SIZE,
                        cudaMemcpyHostToDevice, context.get_gpu_default_stream(i));CUERR;
        if (test_data2) {
            cudaMemcpyAsync(arrays[i].Data[array_index2], test_data2+i*SIZE, sizeof(sa_index_t) * SIZE,
                            cudaMemcpyHostToDevice, context.get_streams(i)[1]);CUERR;
        }
    }
    context.sync_all_streams();
}


template <size_t NUM_GPUS, size_t SIZE>
void copy_down(Context& context, const std::array<MergeTestArrays, NUM_GPUS>& arrays,
             sa_index_t* test_data, sa_index_t* test_data2, uint array_index, uint array_index2) {
    for (uint i = 0; i < NUM_GPUS; ++i) {
        cudaSetDevice(context.get_device_id(i));
        cudaMemcpyAsync(test_data+i*SIZE, arrays[i].Data[array_index], sizeof(sa_index_t) * SIZE,
                        cudaMemcpyDeviceToHost, context.get_gpu_default_stream(i));CUERR;
        if (test_data2) {
            cudaMemcpyAsync(test_data2+i*SIZE, arrays[i].Data[array_index2], sizeof(sa_index_t) * SIZE,
                            cudaMemcpyDeviceToHost, context.get_streams(i)[1]);CUERR;
        }

    }
    context.sync_all_streams();
}

template <size_t NUM_GPUS, size_t SIZE>
void dump(Context& context, const std::array<MergeTestArrays, NUM_GPUS>& arrays,
          sa_index_t* test_data, sa_index_t* test_data2, uint array_index, uint array_index2) {
    copy_down<NUM_GPUS, SIZE>(context, arrays, test_data, test_data2, array_index, array_index2);
    dump_array(test_data, NUM_GPUS*SIZE);
}

int main (int argc, char* argv[]) {

    #ifdef DGX1_TOPOLOGY
    const std::array<uint, NUM_GPUS> gpu_ids { 0, 3, 2, 1,    5, 6, 7, 4 };
    TIMERSTART(create_context);
    Context context(&gpu_ids);
    TIMERSTOP(create_context);
    #else
    Context context;
    #endif


    TIMERSTART(malloc);

    std::array<MergeTestArrays, NUM_GPUS> arrays;
    size_t HOST_BUFFER_SIZE = 4096;
    void* host_mem_buffer;
    cudaMallocHost(&host_mem_buffer, HOST_BUFFER_SIZE);CUERR;

    sa_index_t* test_data = nullptr;
    cudaMallocHost(&test_data, sizeof(sa_index_t) * SIZE * NUM_GPUS); CUERR;

#if INCLUDE_VALUES
    sa_index_t* test_data2 = nullptr;
    cudaMallocHost(&test_data2, sizeof(sa_index_t) * SIZE * NUM_GPUS); CUERR;
    printf("\nTesting with values...\n");
#else
    printf("\nTesting without values...\n");
#endif

    sa_index_t* test_data_backup = new sa_index_t[SIZE * NUM_GPUS];

    size_t temp_size = 0, temp_size_node_0 = 0;

    using merge_types = crossGPUReMerge::mergeTypes<sa_index_t, sa_index_t>;
    using MergeNodeInfo = crossGPUReMerge::MergeNodeInfo<merge_types>;
    std::array<MergeNodeInfo, NUM_GPUS> node_info;
    for (uint i = 0; i < NUM_GPUS; ++i) {
        cudaSetDevice(context.get_device_id(i)); CUERR;
        arrays[i].len = SIZE;
        for (uint j = 0; j < NUM_ARRAYS; ++j) {
            size_t size = sizeof(sa_index_t) * SIZE;
            if (j == NUM_ARRAYS-1) {
                // need more mem for temp buffer for CUB sorting
//                size*=4;
                size = std::max(1024ul, size); // Need some min size (small test inputs).
                temp_size = size;
            }
            if (i == 0) {
                size*=NUM_GPUS;
                temp_size_node_0 = size;
            }
            cudaMalloc(&arrays[i].Data[j], size);CUERR;
        }
    }

//    context.print_connectivity_matrix();

    TIMERSTOP(malloc);

    printf("Generating %zu K keys...\n", SIZE*NUM_GPUS / 1024);

    TIMERSTART(generate_keys);
    generate_test_keys<SIZE*NUM_GPUS>(test_data);
    TIMERSTOP(generate_keys);

    memcpy(test_data_backup, test_data, NUM_GPUS*SIZE*sizeof(sa_index_t));

#if INCLUDE_VALUES
    memcpy(test_data2, test_data, NUM_GPUS*SIZE*sizeof(sa_index_t));


    TIMERSTART(copy_up);
    copy_up<NUM_GPUS, SIZE>(context, arrays, test_data, test_data2, 1, 4);
    TIMERSTOP(copy_up);
#else
    TIMERSTART(copy_up);
    copy_up<NUM_GPUS, SIZE>(context, arrays, test_data, nullptr, 1, 0);
    TIMERSTOP(copy_up);
#endif


    uint sort_out_keys_array_index;

    TIMERSTART(sorting);
    sort_per_node<NUM_GPUS, SIZE, INCLUDE_VALUES>(context, arrays, temp_size, sort_out_keys_array_index);
    TIMERSTOP(sorting);

    using MergeManager = crossGPUReMerge::ReMergeManager<NUM_GPUS, merge_types, TopologyHelper>;

    QDAllocator host_temp_allocator(host_mem_buffer, HOST_BUFFER_SIZE);

    for (uint i = 0; i < NUM_GPUS; ++i) {
        uint merge_temp = 1 , merge_in = 0;
        if (sort_out_keys_array_index == 1) {
            merge_in = 1;
            merge_temp = 0;
        }
        sort_out_keys_array_index = merge_in; // Should already be the case.
#if INCLUDE_VALUES
        node_info[i] = { SIZE, SIZE, i, arrays[i].Data[merge_in], arrays[i].Data[merge_in+3],
                              arrays[i].Data[merge_temp], arrays[i].Data[merge_temp+3],
                            arrays[i].Data[2], arrays[i].Data[5]};
        context.get_device_temp_allocator(i).init(arrays[i].Data[5], temp_size);
#else
        node_info[i] = { SIZE, SIZE, i, arrays[i].Data[merge_in], nullptr, arrays[i].Data[merge_temp], nullptr,
                                       arrays[i].Data[2], nullptr };
        context.get_device_temp_allocator(i).init(arrays[i].Data[2], temp_size);
#endif

    }

    MergeManager merge_manager(context, host_temp_allocator);
    merge_manager.set_node_info(node_info);

    std::vector<crossGPUReMerge::MergeRange> ranges;
    ranges.push_back({{0, 0 }, { NUM_GPUS-1, SIZE }});

    LessComp<sa_index_t> comp;
    TIMERSTART(merging);
    merge_manager.merge(ranges, comp);
    TIMERSTOP(merging);

#if INCLUDE_VALUES

    TIMERSTART(copy_down);
    copy_down<NUM_GPUS, SIZE>(context, arrays, test_data, test_data2, sort_out_keys_array_index,
                              sort_out_keys_array_index+3);
    TIMERSTOP(copy_down);
    context.sync_all_streams();

    TIMERSTART(check);
    check<SIZE*NUM_GPUS>(test_data);
    check<SIZE*NUM_GPUS>(test_data2);
    TIMERSTOP(check);

    memcpy(test_data, test_data_backup, NUM_GPUS*SIZE*sizeof(sa_index_t));
    memcpy(test_data2, test_data_backup, NUM_GPUS*SIZE*sizeof(sa_index_t));

    TIMERSTART(copy_up_single);
    cudaSetDevice(context.get_device_id(0));
    cudaMemcpyAsync(arrays[0].Data[1], test_data, sizeof(sa_index_t) * SIZE * NUM_GPUS,
                    cudaMemcpyHostToDevice, context.get_gpu_default_stream(0));CUERR;
    cudaMemcpyAsync(arrays[0].Data[4], test_data2, sizeof(sa_index_t) * SIZE * NUM_GPUS,
                    cudaMemcpyHostToDevice, context.get_gpu_default_stream(0));CUERR;
    TIMERSTOP(copy_up_single);

    TIMERSTART(sort_single);
    sort_kv_node<SIZE*NUM_GPUS>(context, arrays[0], 0, temp_size_node_0, sort_out_keys_array_index);
    TIMERSTOP(sort_single);

    TIMERSTART(copy_down_single);
    cudaMemcpyAsync(test_data, arrays[0].Data[sort_out_keys_array_index], sizeof(sa_index_t) * SIZE * NUM_GPUS, cudaMemcpyDeviceToHost,
                    context.get_gpu_default_stream(0));CUERR;
    cudaMemcpyAsync(test_data2, arrays[0].Data[sort_out_keys_array_index+3], sizeof(sa_index_t) * SIZE * NUM_GPUS, cudaMemcpyDeviceToHost,
                    context.get_gpu_default_stream(0));CUERR;
    TIMERSTOP(copy_down_single);

    context.sync_gpu_default_stream(0);

    TIMERSTART(check_single);
    check<SIZE*NUM_GPUS>(test_data);
    check<SIZE*NUM_GPUS>(test_data2);
    TIMERSTOP(check_single);

#else

    TIMERSTART(copy_down);
    copy_down<NUM_GPUS, SIZE>(context, arrays, test_data, nullptr, sort_out_keys_array_index,
                              sort_out_keys_array_index+3);
    TIMERSTOP(copy_down);
    context.sync_all_streams();

    TIMERSTART(check);
    check<SIZE*NUM_GPUS>(test_data);
    TIMERSTOP(check);

    memcpy(test_data, test_data_backup, NUM_GPUS*SIZE*sizeof(sa_index_t));

    TIMERSTART(copy_up_single);
    cudaSetDevice(context.get_device_id(0));
    cudaMemcpyAsync(arrays[0].Data[1], test_data, sizeof(sa_index_t) * SIZE * NUM_GPUS,
                    cudaMemcpyHostToDevice, context.get_gpu_default_stream(0));CUERR;
    TIMERSTOP(copy_up_single);

    TIMERSTART(sort_single);
    sort_keys_node<SIZE*NUM_GPUS>(context, arrays[0], 0, temp_size_node_0, sort_out_keys_array_index);
    TIMERSTOP(sort_single);

    TIMERSTART(copy_down_single);
    cudaMemcpyAsync(test_data, arrays[0].Data[sort_out_keys_array_index], sizeof(sa_index_t) * SIZE * NUM_GPUS, cudaMemcpyDeviceToHost,
                    context.get_gpu_default_stream(0));CUERR;
    TIMERSTOP(copy_down_single);

    context.sync_gpu_default_stream(0);

    TIMERSTART(check_single);
    check<SIZE*NUM_GPUS>(test_data);
    TIMERSTOP(check_single);

#endif

    TIMERSTART(free);
    for (uint i = 0; i < NUM_GPUS; ++i) {
        cudaSetDevice(context.get_device_id(i));CUERR;
        for (uint j = 0; j < NUM_ARRAYS; ++j) {
            cudaFree(arrays[i].Data[j]);
        }
    }
    cudaFreeHost(test_data); CUERR;
#if INCLUDE_VALUES
    cudaFreeHost(test_data2); CUERR;
#endif
    cudaFreeHost(host_mem_buffer);CUERR;
    delete[] test_data_backup;
    TIMERSTOP(free);
}

