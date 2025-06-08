#ifndef MULTI_WAY_PARTITIONING_SEARCH_HPP
#define MULTI_WAY_PARTITIONING_SEARCH_HPP

#include <tuple>
#include "util.h"
#include <iostream>

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
    //printf("1\n");
    int before_mid_count = 0;
    size_t total_size = 0;
    // Initialize
    // M = ranges.size()
    printf("M : %ld\n", M);
    for (uint i = 0; i < M; ++i) {
        starts[i] = 0;
        printf("1.0\n");
        ends[i] = arr_descr.lengths[i];
        //std::cout << "1.1 length:" << arr_descr.lengths[i] << ", M: " << M << ", i:" << i <<std::endl;
        printf("1.1 length: %ld, M: %ld, i: %d\n", arr_descr.lengths[i], M, i);
        uint a = (starts[i] + ends[i]) / 2;
        printf("1.2 keys: %u, a: %u\n", i, a);

        printf("array descritor.keys: %lu\n", arr_descr.keys[i][0]);
        UPDATE_MID_INDEX_AND_VALUE(i);

        printf("1.2 %u\n", i);
        before_mid_count += mid_index[i];
        printf("1.3 %u\n", i);
        total_size += arr_descr.lengths[i];
    }
    printf("k: %ld, total size: %lu\n", k, total_size);
    //std::cout << "k: " << k << ", total_size: " << total_size << std::endl;
    //assert(k < total_size);

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
        //printf("3\n");

        for (int i = 0; i < M; ++i) {
            if (starts[i] < ends[i]) {
                // Pick the min index in a range of equal values.
                if (comp(mid_values[i], min_value) || min_index < 0) { // <
                    //printf("3.1\n");
                    min_value = mid_values[i];
                    min_index = i;
                }
                // Pick the max index in a range of equal values.
                if (!comp(mid_values[i], max_value) || max_index < 0) { // >=
                    //printf("3.2\n");
                    max_value = mid_values[i];
                    max_index = i;
                }
                //printf("3.3\n");
            }
            //            printf("Multi-way partioning search: %u from %u to %u.\n", i, starts[i], ends[i]);
            //            std::cout << i << ". From " << starts[i] << " to " << ends[i] << ", mid: " << mid_index[i]
            //                      << ", value: " << mid_values[i] << std::endl;
        }

        //        std::cout << "min index: " << min_index << ", value: " << min_value << std::endl;
        //        std::cout << "max index: " << max_index << ", value: " << max_value;

        //        std::cout << "\nbefore mid count: " << before_mid_count << ", k: " << k << std::endl;
        //printf("4\n");
        if (min_index == max_index && before_mid_count == k) {
            result_value = mid_values[min_index];
            result_index = mid_index[min_index];
            result_list_index = min_index;
            break;
        }
        //printf("5\n");
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
            //printf("6\n");
        }
        else {
            //printf("7\n");
            //            std::cout << "Adjusting max... " << max_index << std::endl;
            ends[max_index] = mid_index[max_index];
            int_t old_mid = mid_index[max_index];
            UPDATE_MID_INDEX_AND_VALUE(max_index);
            before_mid_count -= old_mid - mid_index[max_index];
        }
        //printf("8\n");
        //        std::cout << std::endl;
    }

    //    std::cout << "Needed " << count << " iterations with total size " << total_size << ".\n";

    //    printf("Multi-way partioning search exited: result list index %u, result index %u.\n", result_list_index, result_index);

    return std::make_tuple(result_list_index, result_index, result_value);
}

template<size_t MAX_GPUS, typename key_t, typename int_t, class comp_fun_t>
__global__  void multi_find_partition_points(ArrayDescriptor<MAX_GPUS, key_t, int_t> arr_descr,
    int_t M, int_t k, comp_fun_t comp, int_t* Results, uint* Safe_list) {

    const uint thidx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ uint  k_list_index;
    __shared__ int_t k_index;
    __shared__ key_t k_value;
    __shared__ int_t offsets[MAX_GPUS];


    printf("A %d\n", thidx);
    if (thidx == 0) {
        std::tuple<size_t, size_t, key_t> ksmallest = multi_way_k_select(arr_descr, M, k, comp);
        printf("B\n");
        k_list_index = std::get<0>(ksmallest);
        k_index = std::get<1>(ksmallest);
        k_value = std::get<2>(ksmallest);
        *Safe_list = k_list_index;
    }
    printf("AA %d\n", thidx);

    // TODO: optimize this
    if (thidx == 0) {
        offsets[0] = 0;
        for (uint i = 1; i < M; ++i) {
            offsets[i] = offsets[i - 1] + arr_descr.lengths[i - 1];
        }
    }
    printf("C %d\n", thidx);

    __syncthreads();
    printf("D %d\n", thidx);
    assert(blockDim.x >= M);

    if (thidx < M) {
        int_t list = thidx;
        int_t result;
        if (list != k_list_index) {
            const key_t* arr = arr_descr.keys[list];
            printf("E %d\n", thidx);
            int_t start, end, mid, offset, k_offset;
            key_t mid_value;
            key_t _k_value = k_value;
            start = 0;
            end = arr_descr.lengths[list];
            printf("F %d\n", thidx);
            offset = offsets[list];
            printf("G %d\n", thidx);
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
                //            std::cout << "From " << start << " to " << end << ", mid: " << mid << ", mid-value: " << mid_value << std::endl;

            }
            //            std::cout << "List " << list << ", k: " << k << " looking for: " << value <<", s: " << start << ", e: " << end << "\n";
            result = start;
        }
        else {
            result = k_index;
        }
        printf("H %d\n", thidx);
        Results[list] = result;
    }

}

#endif // MULTI_WAY_PARTITIONING_SEARCH_HPP
