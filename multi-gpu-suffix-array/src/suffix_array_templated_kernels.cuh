#ifndef SUFFIX_ARRAY_TEMPLATED_KERNELS_CUH
#define SUFFIX_ARRAY_TEMPLATED_KERNELS_CUH

#include <cassert>


namespace kernels {
    // Made so many stupid mistakes with unmatched parameters, a template is just better.

    template <typename kmer_t>
    __global__ void write_ranks_diff_multi(const kmer_t* Input_data, const kmer_t* last_element_prev,
        sa_index_t base_index, sa_index_t def_value, sa_index_t* Output_rank,
        size_t N)
    {
        for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
            tidx < N; tidx += blockDim.x * gridDim.x) {
            sa_index_t out;

            if (tidx > 0) {
                if (Input_data[tidx - 1] != Input_data[tidx]) {
                    out = base_index + tidx;
                }
                else {
                    out = def_value;
                }
            }
            else {
                if (last_element_prev) {
                    if (*last_element_prev != Input_data[tidx])
                        out = base_index;
                    else
                        out = def_value;
                }
                else {
                    out = base_index;
                }
            }
            Output_rank[tidx] = out;
        }
    }

    template <size_t THREADS_PER_BLOCK, size_t ITEMS_PER_THREAD>
    __global__ void combine_S12_kv_shared(const MergeStageSuffixS12HalfKey* Keys,
        const MergeStageSuffixS12HalfValue* Values,
        MergeStageSuffix* Out, size_t N) {
        __shared__ MergeStageSuffix out_buffer[THREADS_PER_BLOCK * ITEMS_PER_THREAD];

        for (sa_index_t curr_block_offset = blockIdx.x * blockDim.x * ITEMS_PER_THREAD;
            curr_block_offset < SDIV(N, ITEMS_PER_THREAD * THREADS_PER_BLOCK) * ITEMS_PER_THREAD * THREADS_PER_BLOCK;
            curr_block_offset += ITEMS_PER_THREAD * blockDim.x * gridDim.x) {

#pragma unroll
            for (uint item_c = 0; item_c < ITEMS_PER_THREAD; ++item_c) {
                sa_index_t i = curr_block_offset + item_c * THREADS_PER_BLOCK + threadIdx.x;
                MergeStageSuffixS12HalfKey k;
                MergeStageSuffixS12HalfValue v;
                MergeStageSuffix s;
                if (i < N) {
                    k = Keys[i];
                    v = Values[i];

                    s.chars[0] = v.chars[0];
                    s.rank_p1 = v.rank_p1p2;
                    s.chars[1] = v.chars[1];
                    s._padding[0] = v._padding[0];
                    s._padding[1] = v._padding[1];
                    s.rank_p2 = v.rank_p1p2;
                    s.index = k.index;
                    sa_index_t target_shared_index = k.own_rank - curr_block_offset;

                    assert(target_shared_index < THREADS_PER_BLOCK * ITEMS_PER_THREAD);

                    out_buffer[target_shared_index] = s;
                }
            }
            __syncthreads();
#pragma unroll
            for (uint item_c = 0; item_c < ITEMS_PER_THREAD; ++item_c) {
                sa_index_t i = curr_block_offset + item_c * THREADS_PER_BLOCK + threadIdx.x;
                if (i < N)
                    Out[i] = out_buffer[item_c * THREADS_PER_BLOCK + threadIdx.x];
            }
            __syncthreads();
        }
    }

    template <size_t THREADS_PER_BLOCK, size_t ITEMS_PER_THREAD>
    __global__ void write_to_isa_2_shared_all(const sa_index_t* Input_ranks, const sa_index_t* Target_index,
        size_t ISA_N, sa_index_t* Isa, sa_index_t N)
    {
        // assert(ISA_N == N);
        __shared__ sa_index_t out_buffer[THREADS_PER_BLOCK * ITEMS_PER_THREAD];

        for (sa_index_t curr_block_offset = blockIdx.x * blockDim.x * ITEMS_PER_THREAD;
            curr_block_offset < SDIV(N, ITEMS_PER_THREAD * THREADS_PER_BLOCK) * ITEMS_PER_THREAD * THREADS_PER_BLOCK;
            curr_block_offset += ITEMS_PER_THREAD * blockDim.x * gridDim.x) {

#pragma unroll
            for (uint item_c = 0; item_c < ITEMS_PER_THREAD; ++item_c) {
                sa_index_t i = curr_block_offset + item_c * THREADS_PER_BLOCK + threadIdx.x;
                sa_index_t index;
                sa_index_t rank;
                if (i < N) {
                    index = Target_index[i];
                    rank = Input_ranks[i];

                    sa_index_t target_shared_index = index - curr_block_offset;

                    assert(target_shared_index < THREADS_PER_BLOCK * ITEMS_PER_THREAD);

                    out_buffer[target_shared_index] = rank;
                }
            }
            __syncthreads();
#pragma unroll
            for (uint item_c = 0; item_c < ITEMS_PER_THREAD; ++item_c) {
                sa_index_t i = curr_block_offset + item_c * THREADS_PER_BLOCK + threadIdx.x;
                if (i < ISA_N)
                    Isa[i] = out_buffer[item_c * THREADS_PER_BLOCK + threadIdx.x];
            }
            __syncthreads();
        }
    }



    __device__ __forceinline__ sa_index_t binary_search_for_index_high_bits_block(const sa_index_t* Indices,
        sa_index_t high_bits,
        bool lower_bounds,
        const sa_index_t mask,
        sa_index_t N) {
        sa_index_t mid, mid_value;
        sa_index_t start = 0;
        sa_index_t end = N;
        while (start < end) {
            mid = (start + end) / 2;
            mid_value = Indices[mid] & mask;
            if (mid_value < high_bits) {
                start = mid + 1;
            }
            else if (mid_value == high_bits) { // ==
                if (lower_bounds)
                    end = mid;
                else
                    start = mid + 1;
            }
            else {
                end = mid;
            }
        }
        return start;
    }

    template <size_t THREADS_PER_BLOCK, size_t ITEMS_PER_THREAD>
    __global__ void write_to_isa_2_shared_most(const sa_index_t* Input_ranks, const sa_index_t* Target_index,
        size_t ISA_N, sa_index_t* Isa, sa_index_t N)
    {
        const sa_index_t MASK = ~(sa_index_t(THREADS_PER_BLOCK * ITEMS_PER_THREAD) - 1);
        __shared__ sa_index_t out_buffer[THREADS_PER_BLOCK * ITEMS_PER_THREAD];
        __shared__ sa_index_t low, high;

        for (sa_index_t curr_block_offset = blockIdx.x * blockDim.x * ITEMS_PER_THREAD;
            curr_block_offset < SDIV(ISA_N, ITEMS_PER_THREAD * THREADS_PER_BLOCK) * ITEMS_PER_THREAD * THREADS_PER_BLOCK;
            curr_block_offset += ITEMS_PER_THREAD * blockDim.x * gridDim.x) {

            // Search for the correct indices.
            if (threadIdx.x == 0)
                low = binary_search_for_index_high_bits_block(Target_index, curr_block_offset, true, MASK, N);
            else if (threadIdx.x == THREADS_PER_BLOCK - 1)
                high = binary_search_for_index_high_bits_block(Target_index, curr_block_offset, false, MASK, N);

#pragma unroll
            for (uint item_c = 0; item_c < ITEMS_PER_THREAD; ++item_c) {
                out_buffer[item_c * THREADS_PER_BLOCK + threadIdx.x] = 0;
            }
            __syncthreads();
            //        if (threadIdx.x == 0) {
            //            printf("Block %u/%u found interval: %u to %u with high bits %u.\n", blockIdx.x, gridDim.x, low, high, curr_block_offset);
            //        }

            if (low < high) {
#pragma unroll
                for (uint item_c = 0; item_c < ITEMS_PER_THREAD; ++item_c) {
                    sa_index_t i = low + item_c * THREADS_PER_BLOCK + threadIdx.x;
                    sa_index_t index;
                    sa_index_t rank;
                    if (i < high) {
                        index = Target_index[i];
                        rank = Input_ranks[i];

                        sa_index_t target_shared_index = index - curr_block_offset;

                        assert(target_shared_index < THREADS_PER_BLOCK * ITEMS_PER_THREAD);
                        assert(rank != 0);

                        out_buffer[target_shared_index] = rank;
                    }
                }
                __syncthreads();

#pragma unroll
                for (uint item_c = 0; item_c < ITEMS_PER_THREAD; ++item_c) {
                    sa_index_t i = curr_block_offset + item_c * THREADS_PER_BLOCK + threadIdx.x;
                    sa_index_t r = out_buffer[item_c * THREADS_PER_BLOCK + threadIdx.x];
                    if (i < ISA_N && r != 0)
                        Isa[i] = r;
                }
                __syncthreads();
            }
        }
    }

    template<typename key>
    __global__ void writeSamples(size_t* sample_pos, key* data, key* out, size_t sample_size) {
        const uint thidx = blockDim.x * blockIdx.x + threadIdx.x;
        if (thidx < sample_size) {
            out[thidx] = data[sample_pos[thidx]];
        }
    }

    template<typename key>
    __global__ void selectSplitter(key* samples, size_t sample_count) {
        const uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
        samples[tidx] = samples[sample_count * (tidx + 1)];
    }

    template<typename key, typename Compare>
    __global__ void find_split_index(key* keys, size_t* split_index, key* splitter, size_t size, Compare comp) {
        const uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

        // for the send sizes
        if (tidx >= NUM_GPUS - 1) {
            split_index[tidx] = size;
            return;
        }
        size_t start = 0;
        size_t end = size;
        size_t index = 0;

        while (start < end)
        {
            index = (start + end) / 2;
            if (comp(splitter[tidx], keys[index])) {
                end = index;
            }
            else
            {
                start = index + 1;
            }
        }
        split_index[tidx] = start;
    }

    template<typename key, typename Compare>
    __global__ void bucket_keys(key* keys, key* splitter, key** buckets, Compare comp) {
        const uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

        __shared__ size_t bucketIdx[NUM_GPUS];
        __shared__ uint locks[NUM_GPUS];
        if (tidx < 4) {
            bucketIdx[tidx] = 0;
            locks[tidx] = 0;
        }
        __syncthreads();

        key cur = keys[tidx];

        // for the send sizes
        // printf("splitter: %u\n", splitter[tidx].index);
        // if (tidx >= NUM_GPUS - 1) {
        //     split_index[tidx] = size;
        //     return;
        // }
        size_t start = 0;
        size_t end = NUM_GPUS;
        size_t index = 0;
        while (start < end)
        {
            index = (start + end) / 2;
            if (comp(keys[tidx], splitter[index])) {
                end = index;
            }
            else
            {
                start = index + 1;
            }
        }
        bool leaveLoop = false;
        while (!leaveLoop) {
            if (atomicExch(&(locks[start]), 1u) == 0u) {
                buckets[start][bucketIdx[start]] = keys[tidx];
                bucketIdx[start]++;
                leaveLoop = true;
                atomicExch(&(locks[start]), 0u);
            }
        }
    }

}
#endif // SUFFIX_ARRAY_TEMPLATED_KERNELS_CUH
