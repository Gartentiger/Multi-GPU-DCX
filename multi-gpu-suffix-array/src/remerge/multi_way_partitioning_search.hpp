#ifndef MULTI_WAY_PARTITIONING_SEARCH_HPP
#define MULTI_WAY_PARTITIONING_SEARCH_HPP

#include <tuple>
#include "util.h"
#include <iostream>
#include <thrust/device_vector.h>

template<size_t SIZE, typename key_t, typename int_t>
struct ArrayDescriptor {
    const key_t* keys[SIZE];
    int_t lengths[SIZE];
};


#define UPDATE_MID_INDEX_AND_VALUE(index) \
mid_index[index] = (starts[index] + ends[index]) / 2; \
mid_values[index] = arr_descr.keys[index][mid_index[index]];

template<typename key_t, typename int_t, class comp_fun_t, size_t MAX_GPUS>
HOST_DEVICE std::tuple<uint, int_t, key_t>
multi_way_k_select(ArrayDescriptor<MAX_GPUS, key_t, int_t> arr_descr, int_t M, int_t k, comp_fun_t comp) {

    int_t mid_index[MAX_GPUS];
    key_t mid_values[MAX_GPUS];
    int_t starts[MAX_GPUS];
    int_t ends[MAX_GPUS];
    int before_mid_count = 0;
    size_t total_size = 0;
    // Initialize
    // M = ranges.size()
    for (uint i = 0; i < M; ++i) {
        starts[i] = 0;
        ends[i] = arr_descr.lengths[i];
        uint a = (starts[i] + ends[i]) / 2;

        UPDATE_MID_INDEX_AND_VALUE(i);

        before_mid_count += mid_index[i];
        total_size += arr_descr.lengths[i];
    }


    assert(k < total_size);

    bool done = false;
    key_t result_value;
    int_t result_index = 0;
    int_t result_list_index;

    while (!done) {
        key_t min_value, max_value;
        //        memset(&min_value , 0xff, sizeof(key_t));
        //        memset(&max_value , 0x00, sizeof(key_t));

        //        key_t min_value = std::numeric_limits<key_t>::max();  // Does silently fail for unknown types... :/
        //        key_t max_value = std::numeric_limits<key_t>::min();
        int_t max_index = -1, min_index = -2;

        for (int i = 0; i < M; ++i) {
            if (starts[i] < ends[i]) {
                // Pick the min index in a range of equal values.
                if (comp(mid_values[i], min_value) || min_index < 0) { // <
                    min_value = mid_values[i];
                    min_index = i;
                }
                // Pick the max index in a range of equal values.
                if (!comp(mid_values[i], max_value) || max_index < 0) { // >=
                    max_value = mid_values[i];
                    max_index = i;
                }
            }
            //            printf("Multi-way partioning search: %u from %u to %u.\n", i, starts[i], ends[i]);
            //            std::cout << i << ". From " << starts[i] << " to " << ends[i] << ", mid: " << mid_index[i]
            //                      << ", value: " << mid_values[i] << std::endl;
        }

        //        std::cout << "min index: " << min_index << ", value: " << min_value << std::endl;
        //        std::cout << "max index: " << max_index << ", value: " << max_value;

        //        std::cout << "\nbefore mid count: " << before_mid_count << ", k: " << k << std::endl;
        if (min_index == max_index && before_mid_count == k) {
            result_value = mid_values[min_index];
            result_index = mid_index[min_index];
            result_list_index = min_index;
            break;
        }
        if (before_mid_count < k) {
            //            std::cout << "Adjusting min..." << min_index << std::endl;
            int_t old_mid = mid_index[min_index];
            if (starts[min_index] == mid_index[min_index]) {
                starts[min_index] = mid_index[min_index] + 1;
            }
            else {
                starts[min_index] = mid_index[min_index];
            }
            UPDATE_MID_INDEX_AND_VALUE(min_index);
            before_mid_count += mid_index[min_index] - old_mid;
        }
        else {
            //            std::cout << "Adjusting max... " << max_index << std::endl;
            ends[max_index] = mid_index[max_index];
            int_t old_mid = mid_index[max_index];
            UPDATE_MID_INDEX_AND_VALUE(max_index);
            before_mid_count -= old_mid - mid_index[max_index];
        }
        //        std::cout << std::endl;
    }

    //    std::cout << "Needed " << count << " iterations with total size " << total_size << ".\n";

    //    printf("Multi-way partioning search exited: result list index %u, result index %u.\n", result_list_index, result_index);

    return std::make_tuple(result_list_index, result_index, result_value);
}
template<size_t SIZE, typename key_t, typename int_t>
struct SearchGPU {
    int_t startIndex;
    int_t M;
    std::tuple<size_t, size_t, key_t> ksmallest;
    int_t lengths[SIZE];

};

template<size_t MAX_GPUS, typename key_t, typename int_t, class comp_fun_t>
__global__  void find_partition_points(const key_t* keys, comp_fun_t comp, uint gpuId, int_t* results, SearchGPU<MAX_GPUS, key_t, int_t>* sgpus) {
    const uint thidx = blockDim.x * blockIdx.x + threadIdx.x;
    uint  k_list_index;
    int_t k_index;
    key_t k_value;
    int_t offsets[MAX_GPUS];
    k_list_index = std::get<0>(sgpus[thidx].ksmallest);
    k_index = std::get<1>(sgpus[thidx].ksmallest);
    k_value = std::get<2>(sgpus[thidx].ksmallest);
    // printf("[%u] thidx: %d, ksmallest %u, %ld, %u\n", gpuId, thidx, k_list_index, k_index, k_value);
    //*(sgpus[thidx].safeList) = k_list_index;


    //
    offsets[0] = 0;
    for (uint i = 1; i < sgpus[thidx].M; ++i) {
        offsets[i] = offsets[i - 1] + sgpus[thidx].lengths[i - 1];
        // printf("[%u] thidx: %u, offsets: %ld\n", gpuId, thidx, offsets[i]);
    }


    int_t result;
    if (gpuId != k_list_index) {
        const key_t* arr = keys + sgpus[thidx].startIndex;
        int_t start, end, mid, offset, k_offset;
        key_t mid_value;
        key_t _k_value = k_value;
        start = 0;
        end = sgpus[thidx].lengths[gpuId];
        offset = offsets[gpuId];
        k_offset = offsets[k_list_index] + k_index;
        while (start < end) {
            mid = (start + end) / 2;
            mid_value = arr[mid];
            if (comp(mid_value, _k_value)) {
                start = mid + 1;
            }
            else if (!comp(mid_value, _k_value) && !comp(_k_value, mid_value)) { // ==
                if (offset + mid < k_offset)
                    start = mid + 1;
                else
                    end = mid;
            }
            else {
                end = mid;
            }
        }
        result = start;
    }
    else {
        result = k_index;
    }
    results[thidx] = result;
    // printf("[%u] thidx: %u, result %ld\n", gpuId, thidx, result);
}

template<size_t MAX_GPUS, typename key_t, typename int_t, class comp_fun_t>
__global__  void multi_find_partition_points(ArrayDescriptor<MAX_GPUS, key_t, int_t> arr_descr,
    int_t M, int_t k, comp_fun_t comp, int_t* Results, uint* Safe_list, std::tuple<size_t, size_t, key_t> ksmallestHost) {

    const uint thidx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ uint  k_list_index;
    __shared__ int_t k_index;
    __shared__ key_t k_value;
    __shared__ int_t offsets[MAX_GPUS];


    if (thidx == 0) {
        std::tuple<size_t, size_t, key_t> ksmallest = ksmallestHost;//multi_way_k_select(arr_descr, M, k, comp);
        k_list_index = std::get<0>(ksmallest);
        k_index = std::get<1>(ksmallest);
        k_value = std::get<2>(ksmallest);
        *Safe_list = k_list_index;
    }

    // TODO: optimize this
    if (thidx == 0) {
        offsets[0] = 0;
        for (uint i = 1; i < M; ++i) {
            offsets[i] = offsets[i - 1] + arr_descr.lengths[i - 1];
        }
    }

    __syncthreads();
    assert(blockDim.x >= M);

    if (thidx < M) {
        int_t list = thidx;
        int_t result;
        if (list != k_list_index) {
            const key_t* arr = arr_descr.keys[list];
            int_t start, end, mid, offset, k_offset;
            key_t mid_value;
            key_t _k_value = k_value;
            start = 0;
            end = arr_descr.lengths[list];
            offset = offsets[list];
            k_offset = offsets[k_list_index] + k_index;
            while (start < end) {
                mid = (start + end) / 2;
                mid_value = arr[mid];
                if (comp(mid_value, _k_value)) {
                    start = mid + 1;
                }
                else if (!comp(mid_value, _k_value) && !comp(_k_value, mid_value)) { // ==
                    if (offset + mid < k_offset)
                        start = mid + 1;
                    else
                        end = mid;
                }
                else {
                    end = mid;
                }

            }
            result = start;
        }
        else {
            result = k_index;
        }
        Results[list] = result;
    }

}

#endif // MULTI_WAY_PARTITIONING_SEARCH_HPP
