#pragma once
#include <cstdio>
#include <cassert>
#include <array>
#include <cmath>
#include <cuda_runtime.h> 
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/binary_search.h>
#include <kamping/checking_casts.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/gather.hpp>
#include <kamping/collectives/reduce.hpp>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include "kamping/collectives/scatter.hpp"
#include <kamping/data_buffer.hpp>
#include <kamping/environment.hpp>
#include <kamping/measurements/printer.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/named_parameters.hpp>
#include <kamping/communicator.hpp>
#include <kamping/p2p/recv.hpp>
#include <kamping/p2p/send.hpp>
#include <vector>
#include <mpi.h>
#include <chrono>
#include <numeric>
#include <random>
#include <thread>
#include "cuda_helpers.h"
#include "sort_kernels.cuh"
#include "nccl.h"
#include "gossip/context.cuh"


template<typename key, typename Compare, size_t NUM_GPUS>
void SampleSort(thrust::device_vector <key>& keys_vec, size_t sample_size, Compare cmp, MultiGPUContext<NUM_GPUS>& mcontext) {
    key* keys = thrust::raw_pointer_cast(keys_vec.data());
    size_t size = keys_vec.size();
    ASSERT(sample_size < size);
    // SaGPU gpu = mgpus[world_rank()];
    // size_t count = gpu.num_elements - gpu.pd_elements;

    std::mt19937 g(7777 + world_rank());
    std::uniform_int_distribution<std::mt19937::result_type> randomDist(0, size - 1);

    auto& t = kamping::measurements::timer();
    t.synchronize_and_start("sample_sort");
    // pre sort for easy splitter index binary search 
    // mcontext.get_mgpu_default_context_for_device(world_rank()).set_device_temp_mem(temp_mem, sizeof(key) * size * 3);
    // mgpu::mergesort(keys, size, cmp, mcontext.get_mgpu_default_context_for_device(world_rank()));
    // printf("[%lu] sorting keys\n", world_rank());
    {
        // t.start("init_sort");
        // size_t temp_storage_size = 0;
        // cub::DeviceMergeSort::SortKeys(nullptr, temp_storage_size, keys, size, cmp);
        // void* temp;
        // cudaMalloc(&temp, temp_storage_size);
        // CUERR;
        // cub::DeviceMergeSort::SortKeys(temp, temp_storage_size, keys, size, cmp, mcontext.get_gpu_default_stream(world_rank()));
        // cudaFreeAsync(temp, mcontext.get_gpu_default_stream(world_rank()));
        // mcontext.sync_all_streams();
        // t.stop();
    }
    // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (keys, size, world_rank());
    // mcontext.sync_all_streams();
    // comm_world().barrier();
    // printf("[%lu] sorted keys\n", world_rank());

    // pick random sample positions
    t.start("sampling");
    thrust::device_vector<key> d_samples_vec(sample_size * NUM_GPUS);
    key* d_samples = thrust::raw_pointer_cast(d_samples_vec.data());
    // cudaMalloc(&d_samples, sizeof(key) * sample_size * NUM_GPUS);
    CUERR;
    size_t* h_samples_pos = (size_t*)malloc(sizeof(size_t) * sample_size);
    for (size_t i = 0; i < sample_size; i++)
    {
        h_samples_pos[i] = randomDist(g);
        // printf("[%lu] samples pos[%lu]: %lu\n", world_rank(), i, h_samples_pos[i]);
    }
    // printf("[%lu] picked sample positions\n", world_rank());

    size_t* d_samples_pos;
    cudaMalloc(&d_samples_pos, sizeof(size_t) * sample_size);
    CUERR;
    cudaMemcpy(d_samples_pos, h_samples_pos, sample_size * sizeof(size_t), cudaMemcpyHostToDevice);
    CUERR;
    free(h_samples_pos);
    CUERR;
    // printf("[%lu] copied sample positions to device\n", world_rank());
    // sample position to sample element
    writeSamples << <1, sample_size, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (d_samples_pos, keys, d_samples + world_rank() * sample_size, sample_size);
    cudaFreeAsync(d_samples_pos, mcontext.get_gpu_default_stream(world_rank()));
    // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (d_samples + world_rank() * sample_size, sample_size, world_rank());
    mcontext.sync_all_streams();
    // printf("[%lu] mapped sample positions to corresponding keys\n", world_rank());
    comm_world().barrier();

    // send samples
    ncclGroupStart();
    // thrust::device_vector<key> d_samples_global(sample_size * world_rank());
    // comm_world().allgather(send_buf(std::span<key>(d_samples + world_rank() * sample_size, sample_size)), send_count(sample_size), recv_buf(std::span<key>(thrust::raw_pointer_cast(d_samples_global.data()), sample_size * world_rank())));
    // thrust::host_vector<key> host_global_samples = d_samples_global;
    // for (size_t i = 0; i < host_global_samples.size(); i++)
    // {
    //     printf("[%lu] sample[%lu] %u\n", world_rank(), i, host_global_samples[i].index);
    // }

    for (size_t dst = 0; dst < NUM_GPUS; dst++)
    {
        if (dst == world_rank()) {
            continue;
        }

        NCCLCHECK(ncclSend(d_samples + world_rank() * sample_size, sizeof(key) * sample_size, ncclChar, dst, mcontext.get_nccl(), mcontext.get_streams(world_rank())[dst]));
        // comm_world().isend(send_buf(std::span<key>(d_samples + world_rank() * sample_size, sample_size)), send_count(sample_size), destination(dst));;
    }

    for (size_t src = 0; src < NUM_GPUS; src++)
    {
        if (src == world_rank()) {
            continue;
        }
        // comm_world().irecv(recv_buf(std::span<key>(d_samples + src * sample_size, sample_size)), recv_count(sample_size), source(src));;

        NCCLCHECK(ncclRecv(d_samples + src * sample_size, sizeof(key) * sample_size, ncclChar, src, mcontext.get_nccl(), mcontext.get_streams(src)[world_rank()]));
    }

    ncclGroupEnd();
    mcontext.sync_all_streams();
    comm_world().barrier();
    t.stop();
    // printf("[%lu] received all samples\n", world_rank());
    // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (d_samples, sample_size * NUM_GPUS, world_rank());

    // Sort samples
    {
        t.start("sort_samples");
        // mcontext.get_mgpu_default_context_for_device(world_rank()).reset_temp_memory();
        // mgpu::mergesort(d_samples, sample_size * NUM_GPUS, cmp, mcontext.get_mgpu_default_context_for_device(world_rank()));
        // mcontext.sync_all_streams();
        size_t temp_storage_size = 0;
        cub::DeviceMergeSort::StableSortKeys(nullptr, temp_storage_size, d_samples, sample_size * NUM_GPUS, cmp);
        void* temp;
        cudaMalloc(&temp, temp_storage_size);
        CUERR;
        cub::DeviceMergeSort::StableSortKeys(temp, temp_storage_size, d_samples, sample_size * NUM_GPUS, cmp, mcontext.get_gpu_default_stream(world_rank()));
        cudaFreeAsync(temp, mcontext.get_gpu_default_stream(world_rank()));
        mcontext.sync_all_streams();
        t.stop();
    }
    // printf("[%lu] sorted samples\n", world_rank());
    // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (d_samples, sample_size * NUM_GPUS, world_rank());
    mcontext.sync_all_streams();
    comm_world().barrier();
    t.start("select_splitter");
    selectSplitter << <1, NUM_GPUS - 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (d_samples, sample_size);
    mcontext.sync_all_streams();
    d_samples_vec.resize(NUM_GPUS - 1);
    // printf("[%lu] picked splitters\n", world_rank());
    t.stop();

    // {
    //     comm_world().barrier();
    //     thrust::host_vector<key> h_samples = d_samples_vec;
    //     for (size_t i = 0; i < h_samples.size(); i++)
    //     {
    //         printf("[%lu] splitter[%lu]: %8u\n", world_rank(), i, reinterpret_cast<MergeSuffixes*>(thrust::raw_pointer_cast(h_samples.data()))[i].index);
    //     }
    //     comm_world().barrier();

    // }

    // comm_world().bcast(send_recv_buf(std::span<key>(d_samples, NUM_GPUS - 1)), send_recv_count(NUM_GPUS - 1), root(0));

    // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (d_samples, NUM_GPUS - 1, world_rank());
    // mcontext.sync_all_streams();
    // comm_world().barrier();
    // thrust::transform(d_samples.begin(), d_samples.end(), d_samples.begin(), d_s thrust::placeholders::_1 * sample_size);

    t.start("upperbound");

    // size_t* split_index;
    // cudaMalloc(&split_index, sizeof(size_t) * NUM_GPUS);
    // kernels::find_split_index << <1, NUM_GPUS, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (keys, split_index, d_samples, size, cmp);
    // std::vector<size_t> h_split_index(NUM_GPUS, 0);
    // cudaMemcpyAsync(h_split_index.data(), split_index, sizeof(size_t) * NUM_GPUS, cudaMemcpyDeviceToHost, mcontext.get_gpu_default_stream(world_rank()));
    // cudaFreeAsync(d_samples, mcontext.get_gpu_default_stream(world_rank()));
    // cudaFreeAsync(split_index, mcontext.get_gpu_default_stream(world_rank()));
    // mcontext.sync_all_streams();
    // for (size_t i = 0; i < size; i++)
    // {
    std::vector<size_t> h_bucket_sizes(NUM_GPUS);
    {
        thrust::device_vector<size_t> bucket_sizes(NUM_GPUS);
        thrust::device_vector<uint32_t> bound(size);
        thrust::upper_bound(d_samples_vec.begin(), d_samples_vec.end(), keys_vec.begin(), keys_vec.end(), bound.begin(), cmp);
        // printf("[%lu] after upper bound\n", world_rank());
        t.stop();
        t.start("sorting_upper_bound");
        thrust::device_vector<uint32_t> sorted_upper_bounds(size);
        thrust::device_vector<key> sorted_keys(size);
        // most significant bit is exclusive -> +1
        int sortDown = std::min(int(sizeof(size_t) * 8), int(log2(NUM_GPUS) + 1));

        cub::DoubleBuffer<uint32_t> double_keys(thrust::raw_pointer_cast(bound.data()), thrust::raw_pointer_cast(sorted_upper_bounds.data()));
        cub::DoubleBuffer<key> double_values(keys, thrust::raw_pointer_cast(sorted_keys.data()));

        size_t temp_storage_size = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size,
            double_keys,
            double_values,
            size, 0, sortDown);
        size_t* num_run;
        size_t temp_storage_size2 = 0;

        cub::DeviceRunLengthEncode::Encode(
            nullptr, temp_storage_size2,
            double_keys.Current(), reinterpret_cast<uint32_t*>(double_values.Alternate()),
            thrust::raw_pointer_cast(bucket_sizes.data()), num_run, size);
        void* temp;
        temp_storage_size = std::max(temp_storage_size, temp_storage_size2);
        // printf("temp_storage: %lu\n", temp_storage_size);
        cudaMalloc(&temp, temp_storage_size);
        cudaMalloc(&num_run, sizeof(size_t));

        cub::DeviceRadixSort::SortPairs(temp, temp_storage_size,
            double_keys,
            double_values,
            size, 0, sortDown,
            mcontext.get_gpu_default_stream(world_rank()));
        mcontext.sync_all_streams();
        comm_world().barrier();

        t.stop();
        // printf("[%lu] after sorting bound\n", world_rank());

        t.start("find_lengths");
        cub::DeviceRunLengthEncode::Encode(
            temp, temp_storage_size2,
            double_keys.Current(), reinterpret_cast<uint32_t*>(double_keys.Alternate()),
            thrust::raw_pointer_cast(bucket_sizes.data()), num_run, size, mcontext.get_gpu_default_stream(world_rank()));
        cudaMemcpyAsync(h_bucket_sizes.data(), thrust::raw_pointer_cast(bucket_sizes.data()), sizeof(size_t) * NUM_GPUS, cudaMemcpyDeviceToHost, mcontext.get_gpu_default_stream(world_rank()));
        mcontext.sync_all_streams();
        // size_t test = 0;
        // for (size_t i = 0; i < h_bucket_sizes.size(); i++)
        // {
            // printf("[%lu] bucket_size[%lu]: %lu\n", world_rank(), i, h_bucket_sizes[i]);
            // test += h_bucket_sizes[i];
        // }
        // ASSERT(sorted_keys.size() == keys_vec.size());
        // ASSERT(test == keys_vec.size());
        // printf("[%lu] sorted_key.size %lu == keys %lu", world_rank(), sorted_keys.size(), keys_vec.size());

        if (double_values.Current() != keys) {
            keys_vec.swap(sorted_keys);
            // size_t prefix_sum = 0;
            // for (size_t i = 0; i < NUM_GPUS; i++) {
            //     cudaMemcpyAsync(keys + prefix_sum, thrust::raw_pointer_cast(sorted_keys.data()) + prefix_sum, sizeof(key) * h_bucket_sizes[i], cudaMemcpyDeviceToDevice, mcontext.get_gpu_default_stream(world_rank()));
            //     prefix_sum += h_bucket_sizes[i];
            // }
        }
        cudaFreeAsync(temp, mcontext.get_gpu_default_stream(world_rank()));
        cudaFreeAsync(num_run, mcontext.get_gpu_default_stream(world_rank()));
        mcontext.sync_all_streams();
        t.stop();
        // printf("[%lu] after memcpy\n", world_rank());
    }

    t.stop();

    // for (size_t i = 0; i < NUM_GPUS; i++)
    // {
    //     printf("[%lu] splitter index [%lu]: %lu\n", world_rank(), i, h_split_index[i]);
    // }
    comm_world().barrier();
    printf("[%lu] bucketing done\n", world_rank());
    t.start("alltoall_send_sizes");

    // send_sizes[0] = h_split_index[0];
    // // last split index is size
    // for (size_t i = 1; i < NUM_GPUS; i++)
    // {
    //     send_sizes[i] = h_split_index[i] - h_split_index[i - 1];
    // }
    // for (size_t i = 0; i < NUM_GPUS; i++)
    // {
    //     printf("[%lu] send size[%lu]: %lu\n", world_rank(), i, send_sizes[i]);
    // }

    // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (keys, size, world_rank());
    // mcontext.sync_all_streams();
    // comm_world().barrier();


    std::vector<size_t> recv_sizes;

    recv_sizes = comm_world().alltoall(send_buf(h_bucket_sizes));

    size_t out_size = std::accumulate(recv_sizes.begin(), recv_sizes.end(), 0);
    t.stop();
    t.start("resize_keys_out");
    {
        thrust::device_vector<key> keys_buffer_vec(out_size);
        // keys_vec.resize(out_size);
        t.stop();
        t.start("reorder");
        size_t send_sum = 0;
        size_t recv_sum = 0;
        printf("[%lu] sending\n", world_rank());
        // ALL to ALL
        ncclGroupStart();
        for (size_t dst = 0; dst < NUM_GPUS; dst++)
        {
            // comm_world().isend(send_buf(std::span<key>(keys + send_sum, send_sizes[dst])), send_count(send_sizes[dst]), destination(dst));;

            NCCLCHECK(ncclSend(thrust::raw_pointer_cast(keys_vec.data()) + send_sum, sizeof(key) * h_bucket_sizes[dst], ncclChar, dst, mcontext.get_nccl(), mcontext.get_streams(world_rank())[dst]));
            // NCCLCHECK(ncclSend(thrust::raw_pointer_cast(buckets[dst].data()), sizeof(key) * send_sizes[dst], ncclChar, dst, mcontext.get_nccl(), mcontext.get_streams(world_rank())[dst]));
            send_sum += h_bucket_sizes[dst];
        }

        for (size_t src = 0; src < NUM_GPUS; src++)
        {
            // comm_world().irecv(recv_buf(std::span<key>(keys_out + recv_sum, recv_sizes[src])), recv_count(recv_sizes[src]), source(src));;

            // NCCLCHECK(ncclRecv(keys_out + recv_sum, sizeof(key) * recv_sizes[src], ncclChar, src, mcontext.get_nccl(), mcontext.get_streams(src)[world_rank()]));
            NCCLCHECK(ncclRecv(thrust::raw_pointer_cast(keys_buffer_vec.data()) + recv_sum, sizeof(key) * recv_sizes[src], ncclChar, src, mcontext.get_nccl(), mcontext.get_streams(src)[world_rank()]));

            recv_sum += recv_sizes[src];
        }
        ncclGroupEnd();
        mcontext.sync_all_streams();
        comm_world().barrier();
        keys_vec.swap(keys_buffer_vec);
    }
    t.stop();
    printf("[%lu] reordered keys with splitter, size: %lu\n", world_rank(), out_size);
    {

        t.start("final_sort");
        size_t temp_storage_size = 0;
        cub::DeviceMergeSort::StableSortKeys(nullptr, temp_storage_size, thrust::raw_pointer_cast(keys_vec.data()), out_size, cmp);
        // keys_vec.resize(SDIV(temp_storage_size, sizeof(key)));
        void* temp;
        cudaMalloc(&temp, temp_storage_size);
        CUERR;
        cub::DeviceMergeSort::StableSortKeys(temp, temp_storage_size, thrust::raw_pointer_cast(keys_vec.data()), out_size, cmp, mcontext.get_gpu_default_stream(world_rank()));
        cudaFreeAsync(temp, mcontext.get_gpu_default_stream(world_rank()));

        mcontext.sync_all_streams();
        t.stop();
    }
    comm_world().barrier();
    t.stop();
    printf("[%lu] sorted key.size: %lu\n", world_rank(), keys_vec.size());
}

template<typename key>
void HostSampleSort(std::vector<key>& keys, std::vector<key>& keys_out, size_t NUM_GPUS, size_t size, size_t sample_size) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<std::mt19937::result_type> randomDist(0, size - 1);

    std::vector<key> samples(sample_size + 1);
    for (size_t i = 0; i < sample_size + 1; i++)
    {
        samples[i] = keys[randomDist(g)];
    }
    printf("[%lu] alloc\n", world_rank());
    std::vector<key> recv_samples((sample_size + 1) * NUM_GPUS);
    comm_world().allgather(send_buf(samples), send_count(sample_size + 1), recv_buf(recv_samples));
    std::sort(recv_samples.begin(), recv_samples.end());
    printf("[%lu] allgather\n", world_rank());

    for (size_t i = 0; i < NUM_GPUS - 1; i++)
    {
        recv_samples[i] = recv_samples[(sample_size + 1) * (i + 1)];
    }
    recv_samples.resize(NUM_GPUS - 1);

    std::vector<std::vector<key>> buckets(NUM_GPUS);
    for (auto& bucket : buckets) bucket.reserve((size / NUM_GPUS) * 2);
    for (size_t i = 0; i < size; i++)
    {
        const auto bound = std::upper_bound(recv_samples.begin(), recv_samples.end(), keys[i]);
        buckets[std::min(size_t(bound - samples.begin()), size_t(NUM_GPUS - 1))].push_back(keys[i]);
    }
    keys.clear();
    std::vector<int> sCounts;
    printf("[%lu] bucket\n", world_rank());
    for (auto& bucket : buckets) {
        keys.insert(keys.end(), bucket.begin(), bucket.end());
        sCounts.push_back(bucket.size());
    }
    printf("[%lu] send\n", world_rank());
    keys_out = comm_world().alltoallv(send_buf(keys), send_counts(sCounts));

    std::sort(keys_out.begin(), keys_out.end());
}
template <size_t NUM_GPUS, class mtypes>
using ReMergeTopology = crossGPUReMerge::MergeGPUAllConnectedTopologyHelper<NUM_GPUS, mtypes>;
template<typename key, typename Compare_Device, typename Compare_Host, size_t NUM_GPUS>
void MultiMerge(thrust::device_vector <key>& keys_vec, thrust::device_vector <key>& keys_buffer_vec, Compare_Device cmp_device, Compare_Host cmp_host, MultiGPUContext<NUM_GPUS>& mcontext) {

    using merge_types = crossGPUReMerge::mergeTypes<key, key>;
    using MergeManager = crossGPUReMerge::ReMergeManager<NUM_GPUS, merge_types, ReMergeTopology>;
    using MergeNodeInfo = crossGPUReMerge::MergeNodeInfo<merge_types>;
    size_t temp_storage_size = 0;
    void* temp;
    temp_storage_size = sizeof(key) * keys_vec.size() * 2;
    cudaMalloc(&temp, temp_storage_size);
    mcontext.get_device_temp_allocator(world_rank()).init(temp, temp_storage_size);

    key* h_temp_mem = (key*)malloc(temp_storage_size);
    memset(h_temp_mem, 0, temp_storage_size);
    QDAllocator host_pinned_allocator(h_temp_mem, temp_storage_size);
    std::array<MergeNodeInfo, NUM_GPUS> merge_nodes_info;
    auto& t = kamping::measurements::timer();
    char sf[30];
    size_t bytes = sizeof(key) * keys_vec.size();
    sprintf(sf, "sample_sort_%lu", bytes);
    t.synchronize_and_start(sf);
    t.start("init_sort");
    thrust::sort(keys_vec.begin(), keys_vec.end(), cmp_device);
    t.stop();
    printf("[%lu] initial sort done\n", world_rank());
    key* keys_ptr = thrust::raw_pointer_cast(keys_vec.data());
    key* keys_buffer_ptr = thrust::raw_pointer_cast(keys_buffer_vec.data());
    for (uint gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)
    {
        merge_nodes_info[gpu_index] = { keys_vec.size(), 0, gpu_index,
            keys_ptr, nullptr ,
            keys_buffer_ptr,  nullptr,
            nullptr, nullptr
        };
    }

    MergeManager merge_manager(mcontext, host_pinned_allocator);
    merge_manager.set_node_info(merge_nodes_info);

    std::vector<crossGPUReMerge::MergeRange> ranges;
    ranges.push_back({ 0, 0, (sa_index_t)NUM_GPUS - 1, (sa_index_t)(keys_vec.size()) });
    mcontext.sync_all_streams();


    t.start("merge");
    merge_manager.merge(ranges, cmp_device, cmp_host);
    mcontext.sync_all_streams();
    t.stop();
    comm_world().barrier();
    t.stop_and_append();
}