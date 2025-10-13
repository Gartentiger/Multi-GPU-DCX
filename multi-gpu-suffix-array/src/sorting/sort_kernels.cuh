#pragma once

#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdint>

template<typename key>
__global__ void selectSplitter(key* samples, size_t sample_count) {
    const uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    samples[tidx] = samples[sample_count * (tidx + 1)];
}

template<typename key, typename Compare, size_t NUM_GPUS>
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

template<typename key>
__global__ void writeSamples(size_t* sample_pos, key* data, key* out, size_t sample_size) {
    const uint thidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (thidx < sample_size) {
        out[thidx] = data[sample_pos[thidx]];
    }
}
