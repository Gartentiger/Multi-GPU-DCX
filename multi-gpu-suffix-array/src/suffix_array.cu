#include <cuda_runtime.h> // For syntax completion

#include <cstdio>
#include <cassert>
#include <array>
#include <cmath>

#include "io.cuh"

#include "stages.h"
#include "suffixarrayperformancemeasurements.hpp"

#include "suffix_array_kernels.cuh"
#include "suffixarraymemorymanager.hpp"
#include "cuda_helpers.h"
#include "remerge/remergemanager.hpp"
#include "remerge/remerge_gpu_topology_helper.hpp"

#include "gossip/all_to_all.cuh"
#include "gossip/multisplit.cuh"
#include "distrib_merge/distrib_merge.hpp"

#include <chrono>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include <mpi.h>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <kamping/checking_casts.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/gather.hpp>
#include <kamping/collectives/reduce.hpp>

#include "kamping/collectives/scatter.hpp"
#include <kamping/data_buffer.hpp>
#include <kamping/environment.hpp>
#include <kamping/measurements/printer.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/named_parameters.hpp>
#include <kamping/communicator.hpp>
#include <kamping/p2p/recv.hpp>
#include <kamping/p2p/send.hpp>
// #include <nvToolsExt.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/binary_search.h>
#include "moderngpu/kernel_mergesort.hxx"
#include "dcx_data_generation.hpp"

static const uint NUM_GPUS = 4;
static const uint NUM_GPUS_PER_NODE = 4;
static_assert(NUM_GPUS% NUM_GPUS_PER_NODE == 0, "NUM_GPUS must be a multiple of NUM_GPUS_PER_NODE");
#ifdef DGX1_TOPOLOGY
#include "gossip/all_to_all_dgx1.cuh"
static_assert(NUM_GPUS == 8, "DGX-1 topology can only be used with 8 GPUs");
template <size_t NUM_GPUS>
using All2All = gossip::All2AllDGX1<NUM_GPUS>;
template <size_t NUM_GPUS, class mtypes>
using ReMergeTopology = crossGPUReMerge::DGX1TopologyHelper<NUM_GPUS, mtypes>;
template <typename key_t, typename value_t, typename index_t, size_t NUM_GPUS>
using DistribMergeTopology = distrib_merge::DGX1TopologyHelper<key_t, value_t, index_t, NUM_GPUS>;
#else
#include "gossip/all_to_all.cuh"
// static_assert(NUM_GPUS <= 4, "At the moment, there is no node with more than 4 all-connected nodes. This is likely a configuration error.");

template <size_t NUM_GPUS>
using All2All = gossip::All2All<NUM_GPUS>;
template <size_t NUM_GPUS, class mtypes>
using ReMergeTopology = crossGPUReMerge::MergeGPUAllConnectedTopologyHelper<NUM_GPUS, mtypes>;
template <typename key_t, typename value_t, typename index_t, size_t NUM_GPUS>
using DistribMergeTopology = distrib_merge::DistribMergeAllConnectedTopologyHelper<key_t, value_t, index_t, NUM_GPUS>;
#endif

#if defined(__CUDACC__)
#define _KLC_SIMPLE_(num_elements, stream) <<<std::min(MAX_GRID_SIZE, SDIV((num_elements), BLOCK_SIZE)), BLOCK_SIZE, 0, (stream)>>>
#define _KLC_SIMPLE_ITEMS_PER_THREAD_(num_elements, items_per_thread, stream) <<<std::min(MAX_GRID_SIZE, SDIV((num_elements), BLOCK_SIZE*(items_per_thread))), BLOCK_SIZE, 0, (stream)>>>
#define _KLC_(...) <<<__VA_ARGS__>>>
#else
#define __forceinline__
#define _KLC_SIMPLE_(num_elements, stream)
#define _KLC_SIMPLE_ITEMS_PER_THREAD_(num_elements, items_per_thread, stream)
#define _KLC_(...)
#endif

struct S12PartitioningFunctor : public std::unary_function<sa_index_t, uint32_t>
{
    sa_index_t split_divisor;
    uint max_v;

    __forceinline__
        S12PartitioningFunctor(sa_index_t split_divisor_, uint max_v_)
        : split_divisor(split_divisor_), max_v(max_v_)
    {
    }

    __host__ __device__ __forceinline__ uint32_t operator()(sa_index_t x) const
    {
        return min(((x - 1) / split_divisor), max_v);
    }
};

struct S0Comparator : public std::binary_function<MergeStageSuffixS0HalfKey, MergeStageSuffixS0HalfKey, bool>
{
    __host__ __device__ __forceinline__ bool operator()(const MergeStageSuffixS0HalfKey& a, const MergeStageSuffixS0HalfKey& b) const
    {
        if (a.chars[0] == b.chars[0])
            return a.rank_p1 < b.rank_p1;
        else
            return a.chars[0] < b.chars[0];
    }
};

struct MergeCompFunctor : std::binary_function<MergeStageSuffix, MergeStageSuffix, bool>
{
    __host__ __device__ __forceinline__ bool operator()(const MergeStageSuffix& a, const MergeStageSuffix& b) const
    {
        if (a.index % 3 == 0)
        {
            // assert(b.index % 3 != 0);
            if (b.index % 3 == 1)
            {
                if (a.chars[0] == b.chars[0])
                    return a.rank_p1 < b.rank_p1;
                return a.chars[0] < b.chars[0];
            }
            else
            {
                if (a.chars[0] == b.chars[0])
                {
                    if (a.chars[1] == b.chars[1])
                    {
                        return a.rank_p2 < b.rank_p2;
                    }
                    return a.chars[1] < b.chars[1];
                }
                return a.chars[0] < b.chars[0];
            }
        }
        else
        {
            // assert(b.index % 3 == 0);
            if (a.index % 3 == 1)
            {
                if (a.chars[0] == b.chars[0])
                    return a.rank_p1 < b.rank_p1;
                return a.chars[0] < b.chars[0];
            }
            else
            {
                if (a.chars[0] == b.chars[0])
                {
                    if (a.chars[1] == b.chars[1])
                    {
                        return a.rank_p2 < b.rank_p2;
                    }
                    return a.chars[1] < b.chars[1];
                }
                return a.chars[0] < b.chars[0];
            }
        }
    }
};
__global__ void printArrayss(const unsigned char* input, size_t size, size_t rank)
{
    for (size_t i = 0; i < size; i++) {

        printf("[%lu] input: %c\n", rank, input[i]);
        // unsigned char* kmerI = reinterpret_cast<*>(kmer[i]);
    }
    printf("---------------------------------------------------------------------------\n");
}
__global__ void printArrayss(uint64_t* input, size_t size, size_t rank)
{
    for (size_t i = 0; i < size; i++) {

        printf("[%lu] data[%lu]: %lu\n", rank, i, input[i]);
    }
    printf("---------------------------------------------------------------------------\n");
}
__global__ void printArrayss(sa_index_t* isa, sa_index_t* sa_rank, size_t size, size_t rank)
{
    printf("[%lu] isa: ", rank);
    for (size_t i = 0; i < size; i++) {
        if (i + 1 < size) {

            printf("%u, ", isa[i]);
        }
        else {
            printf("%u", isa[i]);
        }

    }
    printf("\n");
    printf("[%lu]  sa: ", rank);
    for (size_t i = 0; i < size; i++) {
        if (i + 1 < size) {

            printf("%u, ", sa_rank[i]);
        }
        else {
            printf("%u", sa_rank[i]);
        }

    }
    printf("\n");
    printf("---------------------------------------------------------------------------\n");
}
__global__ void printArrayss(sa_index_t* isa, unsigned char* sa_rank, const size_t size, size_t rank)
{
    printf("[%lu] isa: ", rank);
    for (size_t i = 0; i < size; i++) {
        if (i + 1 < size) {

            printf("%u, ", isa[i]);
        }
        else {
            printf("%u", isa[i]);
        }

    }
    printf("\n");
    printf("[%lu]  sa: ", rank);
    for (size_t i = 0; i < size; i++) {
        if (i + 1 < size) {

            printf("%c, ", sa_rank[i]);
        }
        else {
            printf("%c", sa_rank[i]);
        }

    }
    printf("\n");
    printf("---------------------------------------------------------------------------\n");
}
__global__ void printArrayss(char* kmer, sa_index_t* isa, size_t size, size_t rank)
{
    for (size_t i = 0; i < size; i++) {

        printf("[%lu] isa: %u", rank, isa[i]);
        // unsigned char* kmerI = reinterpret_cast<*>(kmer[i]);
        for (int j = 0; j < 8;j++) {
            printf(", %c", kmer[i * 8 + j]);
        }
        printf("\n");
    }
    printf("---------------------------------------------------------------------------\n");
}

__global__ void printArrayss(MergeStageSuffixS12HalfValue* sk, size_t size, size_t rank)
{
    for (size_t i = 0; i < size; i++) {
        printf("[%lu] ", rank);
        printf("%c %c %u\n", sk[i].chars[0], sk[i].chars[1], sk[i].rank_p1p2);
        // printf("[%lu] sk[%lu]: %c, %c, %c, %c, %c, %u, %u, %u, %u, %u, %lu", rank, i, sk[i].xPrefix0, sk[i].xPrefix1, sk[i].xPrefix2, sk[i].xPrefix3, sk[i].xPrefix4, sk[i].ranks0, sk[i].ranks1, sk[i].ranks2, sk[i].ranks3, sk[i].ranks4, sk[i].index);
        // unsigned char* kmerI = reinterpret_cast<*>(kmer[i]);
    }
    printf("---------------------------------------------------------------------------\n");
}

__global__ void printArrayss(MergeSuffixes* sk, size_t size, size_t rank)
{
    for (size_t i = 0; i < size; i++) {
        printf("[%lu] ", rank);
        for (size_t x = 0; x < DCX::X; x++) {
            printf("%c, ", sk[i].prefix[x]);
        }
        for (size_t x = 0; x < DCX::X; x++) {
            printf("%u, ", sk[i].ranks[x]);
        }
        printf("%u\n", sk[i].index);
        // printf("[%lu] sk[%lu]: %c, %c, %c, %c, %c, %u, %u, %u, %u, %u, %lu", rank, i, sk[i].xPrefix0, sk[i].xPrefix1, sk[i].xPrefix2, sk[i].xPrefix3, sk[i].xPrefix4, sk[i].ranks0, sk[i].ranks1, sk[i].ranks2, sk[i].ranks3, sk[i].ranks4, sk[i].index);
        // unsigned char* kmerI = reinterpret_cast<*>(kmer[i]);
    }
    printf("---------------------------------------------------------------------------\n");
}

#include "prefix_doubling.hpp"
#include "suffix_types.h"

#define TIMER_START_PREPARE_FINAL_MERGE_STAGE(stage) mperf_measure.start_prepare_final_merge_stage(stage)
#define TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(stage) mperf_measure.stop_prepare_final_merge_stage(stage)
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
class SuffixSorter
{
    static const int BLOCK_SIZE = 1024;
    static const size_t MAX_GRID_SIZE = 2048;

    using MemoryManager = SuffixArrayMemoryManager<NUM_GPUS, sa_index_t>;
    using MainStages = perf_rec::MainStages;
    using FinalMergeStages = perf_rec::PrepareFinalMergeStages;
    using Context = MultiGPUContext<NUM_GPUS>;

    struct SaGPU
    {
        size_t num_elements, offset;
        size_t pd_elements, pd_offset;
        PDArrays pd_ptr;
        PrepareS12Arrays prepare_S12_ptr;
        PrepareS0Arrays prepare_S0_ptr;
        MergeS12S0Arrays merge_ptr;
    };

    Context& mcontext;
    MemoryManager mmemory_manager;
    MultiSplit<NUM_GPUS> mmulti_split;
    All2All<NUM_GPUS> mall2all;
    std::array<SaGPU, NUM_GPUS> mgpus;

    SuffixArrayPerformanceMeasurements mperf_measure;

    PrefixDoublingSuffixSorter mpd_sorter;

    char* minput;
    size_t minput_len, mreserved_len, mpd_reserved_len, ms0_reserved_len, mper_gpu, mpd_per_gpu;
    size_t mpd_per_gpu_max_bit;
    size_t mtook_pd_iterations;

public:
    SuffixSorter(Context& context, size_t len, char* input)
        : mcontext(context), mmemory_manager(context),
        mmulti_split(context), mall2all(context),
        mperf_measure(32),
        mpd_sorter(mcontext, mmemory_manager, mmulti_split, mall2all, mperf_measure),
        minput(input), minput_len(len)
    {
    }

    template<typename key, typename Compare>
    void SampleSort(thrust::device_vector <key>& keys_vec, thrust::device_vector <key>& keys_out_vec, size_t sample_size, Compare cmp) {
        key* keys = thrust::raw_pointer_cast(keys_vec.data());
        size_t size = keys_vec.size();
        ASSERT(sample_size < size);
        // SaGPU gpu = mgpus[world_rank()];
        // size_t count = gpu.num_elements - gpu.pd_elements;

        std::random_device rd;
        std::mt19937 g(rd());
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
        printf("[%lu] picked sample positions\n", world_rank());

        size_t* d_samples_pos;
        cudaMalloc(&d_samples_pos, sizeof(size_t) * sample_size);
        CUERR;
        cudaMemcpy(d_samples_pos, h_samples_pos, sample_size * sizeof(size_t), cudaMemcpyHostToDevice);
        CUERR;
        free(h_samples_pos);
        CUERR;
        printf("[%lu] copied sample positions to device\n", world_rank());
        // sample position to sample element
        kernels::writeSamples << <1, sample_size, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (d_samples_pos, keys, d_samples + world_rank() * sample_size, sample_size);
        cudaFreeAsync(d_samples_pos, mcontext.get_gpu_default_stream(world_rank()));
        // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (d_samples + world_rank() * sample_size, sample_size, world_rank());
        mcontext.sync_all_streams();
        printf("[%lu] mapped sample positions to corresponding keys\n", world_rank());
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
        printf("[%lu] received all samples\n", world_rank());
        // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (d_samples, sample_size * NUM_GPUS, world_rank());

        // Sort samples
        {
            t.start("sort_samples");
            // mcontext.get_mgpu_default_context_for_device(world_rank()).reset_temp_memory();
            // mgpu::mergesort(d_samples, sample_size * NUM_GPUS, cmp, mcontext.get_mgpu_default_context_for_device(world_rank()));
            // mcontext.sync_all_streams();
            size_t temp_storage_size = 0;
            cub::DeviceMergeSort::SortKeys(nullptr, temp_storage_size, d_samples, sample_size * NUM_GPUS, cmp);
            void* temp;
            cudaMalloc(&temp, temp_storage_size);
            CUERR;
            cub::DeviceMergeSort::SortKeys(temp, temp_storage_size, d_samples, sample_size * NUM_GPUS, cmp, mcontext.get_gpu_default_stream(world_rank()));
            cudaFreeAsync(temp, mcontext.get_gpu_default_stream(world_rank()));
            mcontext.sync_all_streams();
            t.stop();
        }
        printf("[%lu] sorted samples\n", world_rank());
        // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (d_samples, sample_size * NUM_GPUS, world_rank());
        mcontext.sync_all_streams();
        comm_world().barrier();
        t.start("select_splitter");
        kernels::selectSplitter << <1, NUM_GPUS - 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (d_samples, sample_size);
        mcontext.sync_all_streams();
        d_samples_vec.resize(NUM_GPUS - 1);
        printf("[%lu] picked splitters\n", world_rank());
        t.stop();



        // comm_world().bcast(send_recv_buf(std::span<key>(d_samples, NUM_GPUS - 1)), send_recv_count(NUM_GPUS - 1), root(0));

        // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (d_samples, NUM_GPUS - 1, world_rank());
        // mcontext.sync_all_streams();
        // comm_world().barrier();
        // thrust::transform(d_samples.begin(), d_samples.end(), d_samples.begin(), d_s thrust::placeholders::_1 * sample_size);

        t.start("find_splits");

        // size_t* split_index;
        // cudaMalloc(&split_index, sizeof(size_t) * NUM_GPUS);
        // kernels::find_split_index << <1, NUM_GPUS, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (keys, split_index, d_samples, size, cmp);
        // std::vector<size_t> h_split_index(NUM_GPUS, 0);
        // cudaMemcpyAsync(h_split_index.data(), split_index, sizeof(size_t) * NUM_GPUS, cudaMemcpyDeviceToHost, mcontext.get_gpu_default_stream(world_rank()));
        // cudaFreeAsync(d_samples, mcontext.get_gpu_default_stream(world_rank()));
        // cudaFreeAsync(split_index, mcontext.get_gpu_default_stream(world_rank()));
        // mcontext.sync_all_streams();
        thrust::host_vector<thrust::device_vector<key>> buckets(NUM_GPUS);
        for (auto& bucket : buckets) bucket.reserve((size / NUM_GPUS) * 2);
        for (size_t i = 0; i < size; i++)
        {
            const auto bound = thrust::upper_bound(d_samples_vec.begin(), d_samples_vec.end(), keys[i], cmp);
            printf("[%lu] bound\n", world_rank());
            printf("[%lu] bound: %lu\n", world_rank(), size_t(bound - d_samples_vec.begin()));
            buckets[std::min(size_t(bound - d_samples_vec.begin()), size_t(NUM_GPUS - 1))].push_back(keys[i]);
        }
        // keys_vec.clear();
        t.stop();

        // for (size_t i = 0; i < NUM_GPUS; i++)
        // {
        //     printf("[%lu] splitter index [%lu]: %lu\n", world_rank(), i, h_split_index[i]);
        // }
        comm_world().barrier();
        printf("[%lu] bucketing done\n", world_rank());
        t.start("alltoall_send_sizes");
        std::vector<size_t> send_sizes(NUM_GPUS, 0);
        for (size_t i = 0; i < NUM_GPUS; i++)
        {
            send_sizes[i] = buckets[i].size();
        }

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
        comm_world().barrier();

        // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (keys, size, world_rank());
        // mcontext.sync_all_streams();
        // comm_world().barrier();


        std::vector<size_t> recv_sizes;

        recv_sizes = comm_world().alltoall(send_buf(send_sizes));

        size_t out_size = std::accumulate(recv_sizes.begin(), recv_sizes.end(), 0);
        t.stop();
        t.synchronize_and_start("resize_keys_out");
        // keys_out_vec.resize(out_size);
        keys_vec.resize(out_size);
        key* keys_out = thrust::raw_pointer_cast(keys_vec.data());
        t.stop();
        t.synchronize_and_start("reorder");
        size_t send_sum = 0;
        size_t recv_sum = 0;
        printf("[%lu] reordering\n", world_rank());
        // ALL to ALL
        ncclGroupStart();
        for (size_t dst = 0; dst < NUM_GPUS; dst++)
        {
            // comm_world().isend(send_buf(std::span<key>(keys + send_sum, send_sizes[dst])), send_count(send_sizes[dst]), destination(dst));;

            // NCCLCHECK(ncclSend(keys + send_sum, sizeof(key) * send_sizes[dst], ncclChar, dst, mcontext.get_nccl(), mcontext.get_streams(world_rank())[dst]));
            NCCLCHECK(ncclSend(thrust::raw_pointer_cast(buckets[dst].data()), sizeof(key) * send_sizes[dst], ncclChar, dst, mcontext.get_nccl(), mcontext.get_streams(world_rank())[dst]));
            // send_sum += send_sizes[dst];
        }

        for (size_t src = 0; src < NUM_GPUS; src++)
        {
            // comm_world().irecv(recv_buf(std::span<key>(keys_out + recv_sum, recv_sizes[src])), recv_count(recv_sizes[src]), source(src));;

            // NCCLCHECK(ncclRecv(keys_out + recv_sum, sizeof(key) * recv_sizes[src], ncclChar, src, mcontext.get_nccl(), mcontext.get_streams(src)[world_rank()]));
            NCCLCHECK(ncclRecv(keys_out + recv_sum, sizeof(key) * recv_sizes[src], ncclChar, src, mcontext.get_nccl(), mcontext.get_streams(src)[world_rank()]));

            recv_sum += recv_sizes[src];
        }
        ncclGroupEnd();
        mcontext.sync_all_streams();
        comm_world().barrier();
        t.stop();
        printf("[%lu] reordered keys with splitter, size: %lu\n", world_rank(), out_size);
        {

            t.start("final_sort");
            size_t temp_storage_size = 0;
            cub::DeviceMergeSort::SortKeys(nullptr, temp_storage_size, keys_out, out_size, cmp);
            // keys_vec.resize(SDIV(temp_storage_size, sizeof(key)));
            void* temp;
            cudaMalloc(&temp, temp_storage_size);
            CUERR;
            cub::DeviceMergeSort::SortKeys(temp, temp_storage_size, keys_out, out_size, cmp, mcontext.get_gpu_default_stream(world_rank()));
            cudaFreeAsync(temp, mcontext.get_gpu_default_stream(world_rank()));

            mcontext.sync_all_streams();
            t.stop();
        }
        comm_world().barrier();
        t.stop();
        printf("[%lu] sorted key.size: %lu\n", world_rank(), keys_vec.size());
    }


    void HostSampleSort(std::vector<MergeSuffixes>& keys, std::vector<MergeSuffixes>& keys_out, size_t size, size_t sample_size) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::uniform_int_distribution<std::mt19937::result_type> randomDist(0, size - 1);

        std::vector<MergeSuffixes> samples(sample_size + 1);
        for (size_t i = 0; i < sample_size + 1; i++)
        {
            samples[i] = keys[randomDist(g)];
        }
        printf("[%lu] alloc\n", world_rank());
        std::vector<MergeSuffixes> recv_samples((sample_size + 1) * NUM_GPUS);
        comm_world().allgather(send_buf(samples), send_count(sample_size + 1), recv_buf(recv_samples));
        std::sort(recv_samples.begin(), recv_samples.end());
        printf("[%lu] allgather\n", world_rank());

        for (size_t i = 0; i < NUM_GPUS - 1; i++)
        {
            recv_samples[i] = recv_samples[(sample_size + 1) * (i + 1)];
        }
        recv_samples.resize(NUM_GPUS - 1);

        std::vector<std::vector<MergeSuffixes>> buckets(NUM_GPUS);
        for (auto& bucket : buckets) bucket.reserve((size / NUM_GPUS) * 2);
        for (size_t i = 0; i < size; i++)
        {
            const auto bound = std::upper_bound(recv_samples.begin(), recv_samples.end(), keys[i]);
            buckets[bound - samples.begin()].push_back(keys[i]);
        }
        keys.clear();
        std::vector<int> sCounts, sDispls, rCounts(NUM_GPUS), rDispls(NUM_GPUS + 1);
        printf("[%lu] bucket\n", world_rank());
        for (auto& bucket : buckets) {
            keys.insert(keys.end(), bucket.begin(), bucket.end());
            sCounts.push_back(bucket.size());
        }
        printf("[%lu] send\n", world_rank());
        keys_out = comm_world().alltoallv(send_buf(keys), send_counts(sCounts));

        std::sort(keys_out.begin(), keys_out.end());
    }


    void do_sa()
    {




        // TIMER_START_MAIN_STAGE(MainStages::Copy_Input);
        copy_input();



        //
        // mcontext.sync_all_streams();
        // printf("[%lu] Copy Input\n", world_rank());
        // comm_world().barrier();
        //

        TIMERSTART(Total);
        // TIMER_STOP_MAIN_STAGE(MainStages::Copy_Input);

        TIMER_START_MAIN_STAGE(MainStages::Produce_KMers);
        produce_kmers();
        //
        // mcontext.sync_all_streams();
        // printf("[%lu] Produce kmers\n", world_rank());
        // comm_world().barrier();
        //

        TIMER_STOP_MAIN_STAGE(MainStages::Produce_KMers);

        //            mpd_sorter.dump("After K-Mers");

        mtook_pd_iterations = mpd_sorter.sort(4);
        // comm_world().barrier();
        auto& t = kamping::measurements::timer();
        t.aggregate_and_print(
            kamping::measurements::SimpleJsonPrinter{ std::cout, {} });
        std::cout << std::endl;
        t.aggregate_and_print(kamping::measurements::FlatPrinter{});
        std::cout << std::endl;
        //            mpd_sorter.dump("done");
        TIMER_START_MAIN_STAGE(MainStages::Prepare_S12_for_Merge);
        prepare_S12_for_merge();
        return;
        //
        // mcontext.sync_all_streams();
        printf("[%lu] prepare s12 for merge done\n", world_rank());
        // comm_world().barrier();
        //

        TIMER_STOP_MAIN_STAGE(MainStages::Prepare_S12_for_Merge);
        TIMER_START_MAIN_STAGE(MainStages::Prepare_S0_for_Merge);
        prepare_S0_for_merge();
        //
        // mcontext.sync_all_streams();
        printf("[%lu] prepare s0 for merge done\n", world_rank());
        // comm_world().barrier();
        //
        TIMER_STOP_MAIN_STAGE(MainStages::Prepare_S0_for_Merge);
        TIMER_START_MAIN_STAGE(MainStages::Final_Merge);
        final_merge();
        //
        // mcontext.sync_all_streams();
        printf("[%lu] final merge done\n", world_rank());
        // comm_world().barrier();
        //
        TIMER_STOP_MAIN_STAGE(MainStages::Final_Merge);
        // TIMER_START_MAIN_STAGE(MainStages::Copy_Results);
        TIMERSTOP(Total);
        mperf_measure.done();

        copy_result_to_host();
        //
        // mcontext.sync_all_streams();
        // printf("[%lu] complete\n", world_rank());
        // comm_world().barrier();
        //
        // TIMER_STOP_MAIN_STAGE(MainStages::Copy_Results);
    }

    const sa_index_t* get_result() const
    {
        return mmemory_manager.get_h_result();
    }

    SuffixArrayPerformanceMeasurements& get_perf_measurements()
    {
        return mperf_measure;
    }

    void done()
    {
        mmemory_manager.free();
    }

    void alloc()
    {
        // mper_gpu how much data for one gpu
        mper_gpu = SDIV(minput_len, NUM_GPUS);
        ASSERT_MSG(mper_gpu >= DCX::X, "Please give me more input.");

        // Ensure each gpu has a multiple of 3 because of triplets.
        mper_gpu = SDIV(mper_gpu, DCX::X) * DCX::X;
        printf("minput_len: %lu, mper_gpu %lu\n", minput_len, mper_gpu);
        ASSERT(minput_len > (NUM_GPUS - 1) * mper_gpu + 3); // Because of merge
        size_t last_gpu_elems = minput_len - (NUM_GPUS - 1) * mper_gpu;
        ASSERT(last_gpu_elems <= mper_gpu); // Because of merge.

        mreserved_len = SDIV(std::max(last_gpu_elems, mper_gpu) + 8, 14) * 14; // Ensure there are 12 elems more space.
        mreserved_len = std::max(mreserved_len, 1024ul);                       // Min len because of temp memory for CUB.

        mpd_reserved_len = SDIV(mreserved_len, DCX::X) * DCX::C;

        ms0_reserved_len = mreserved_len - mpd_reserved_len;

        auto cub_temp_mem = get_needed_cub_temp_memory(ms0_reserved_len, mpd_reserved_len);

        // Can do it this way since CUB temp memory is limited for large inputs.
        ms0_reserved_len = std::max(ms0_reserved_len, SDIV(cub_temp_mem.first, sizeof(MergeStageSuffix)));
        mpd_reserved_len = std::max(mpd_reserved_len, SDIV(cub_temp_mem.second, sizeof(MergeStageSuffix)));

        mmemory_manager.alloc(minput_len, mreserved_len, mpd_reserved_len, ms0_reserved_len, true);

        mpd_per_gpu = mper_gpu / DCX::X * DCX::C;
        mpd_per_gpu_max_bit = std::min(sa_index_t(log2(float(mpd_per_gpu))) + 1, sa_index_t(sizeof(sa_index_t) * 8));

        size_t pd_total_len = 0, offset = 0, pd_offset = 0;
        for (uint i = 0; i < NUM_GPUS - 1; i++)
        {
            mgpus[i].num_elements = mper_gpu;
            mgpus[i].pd_elements = mpd_per_gpu;
            mgpus[i].offset = offset;
            mgpus[i].pd_offset = pd_offset;
            pd_total_len += mgpus[i].pd_elements;
            init_gpu_ptrs(i);
            offset += mper_gpu;
            pd_offset += mpd_per_gpu;
        }

        mgpus.back().num_elements = last_gpu_elems;
        size_t last_gpu_add_pd_elements = 0;
        // if last_gpu_elems = 9 elements left and X=7 then we have 3+1 sample positions, last_gpu_elems = 10 3+2, last_gpu_elems = 11 3+2...
        if (last_gpu_elems % DCX::X != 0) {
            for (size_t sample = 0; sample < DCX::C; sample++)
            {
                if ((last_gpu_elems % DCX::X) > (size_t)DCX::nextSample[sample]) {
                    last_gpu_add_pd_elements++;
                }
            }
        }

        mgpus.back().pd_elements = (last_gpu_elems / DCX::X) * DCX::C + last_gpu_add_pd_elements;
        mgpus.back().offset = offset;
        mgpus.back().pd_offset = pd_offset;

        // Because of fixup.
        ASSERT(mgpus.back().pd_elements >= 4);

        pd_total_len += mgpus.back().pd_elements;
        init_gpu_ptrs(NUM_GPUS - 1);

        printf("Every node gets %zu (%zu) elements, last node: %zu (%zu), reserved len: %zu.\n", mper_gpu,
            mpd_per_gpu, last_gpu_elems, mgpus.back().pd_elements, mreserved_len);

        mpd_sorter.init(pd_total_len, mpd_per_gpu, mgpus.back().pd_elements, mpd_reserved_len);
    }

    void print_pd_stats() const
    {
        mpd_sorter.print_stats(mtook_pd_iterations);
    }

private:
    void init_gpu_ptrs(uint i)
    {
        mgpus[i].pd_ptr = mmemory_manager.get_pd_arrays(i);
        mgpus[i].prepare_S12_ptr = mmemory_manager.get_prepare_S12_arrays(i);
        mgpus[i].prepare_S0_ptr = mmemory_manager.get_prepare_S0_arrays(i);
        mgpus[i].merge_ptr = mmemory_manager.get_merge_S12_S0_arrays(i);
    }

    std::pair<size_t, size_t> get_needed_cub_temp_memory(size_t S0_count, size_t S12_count) const
    {
        cub::DoubleBuffer<uint64_t> keys(nullptr, nullptr);
        cub::DoubleBuffer<uint64_t> values(nullptr, nullptr);

        size_t temp_storage_size_S0 = 0;
        size_t temp_storage_size_S12 = 0;
        cudaError_t err = cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size_S0,
            keys, values, S0_count, 0, 40);
        CUERR_CHECK(err);
        err = cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size_S12,
            keys, values, S12_count, 0, 40);
        CUERR_CHECK(err);

        return { temp_storage_size_S0, temp_storage_size_S12 };
    }

    void copy_input()
    {
        using kmer_t = uint64_t;
        // for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        //{
        auto gpu_index = world_rank();
        SaGPU& gpu = mgpus[gpu_index];

        // Need the halo to the right for kmers...
        size_t copy_len = std::min(gpu.num_elements + sizeof(kmer_t), minput_len - gpu.offset);

        cudaMemcpyAsync(gpu.pd_ptr.Input, minput, copy_len, cudaMemcpyHostToDevice,
            mcontext.get_gpu_default_stream(gpu_index));
        CUERR;
        if (gpu_index == NUM_GPUS - 1)
        {
            cudaMemsetAsync(gpu.pd_ptr.Input + gpu.num_elements, 0, sizeof(kmer_t),
                mcontext.get_gpu_default_stream(gpu_index));
            CUERR;
        }
        //}

        mcontext.sync_default_streams();
    }

    void produce_kmers()
    {
        // for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        //{
        auto gpu_index = world_rank();
        SaGPU& gpu = mgpus[gpu_index];

        //(mcontext.get_device_id(gpu_index));
        //                kernels::produce_index_kmer_tuples _KLC_SIMPLE_(gpu.num_elements, mcontext.get_gpu_default_stream(gpu_index))
        //                        ((char*)gpu.input, offset, gpu.pd_index, gpu.pd_kmers, gpu.num_elements); CUERR;
        kernels::produce_index_kmer_tuples_12_64_dc7 _KLC_SIMPLE_(gpu.num_elements, mcontext.get_gpu_default_stream(gpu_index))((char*)gpu.pd_ptr.Input, gpu.pd_offset, gpu.pd_ptr.Isa, reinterpret_cast<ulong1*>(gpu.pd_ptr.Sa_rank),
            SDIV(gpu.num_elements, 14) * 14);
        CUERR;
        //}
        if (gpu_index == NUM_GPUS - 1)
        {
            kernels::fixup_last_four_12_kmers_64 << <1, 4, 0, mcontext.get_gpu_default_stream(gpu_index) >> > (reinterpret_cast<ulong1*>(mgpus.back().pd_ptr.Sa_rank) + mgpus.back().pd_elements - 4);
        }
        // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (reinterpret_cast<char*>(gpu.pd_ptr.Sa_rank), gpu.pd_ptr.Isa, SDIV(gpu.num_elements, DCX::X * 2) * DCX::X * 2, world_rank());
        mcontext.sync_default_streams();
        comm_world().barrier();
    }

    void prepare_S12_for_merge()
    {
        std::array<MultiSplitNodeInfoT<sa_index_t, sa_index_t, sa_index_t>, NUM_GPUS> multi_split_node_info;
        std::array<All2AllNodeInfoT<MergeSuffixes, sa_index_t, sa_index_t>, NUM_GPUS> all2all_node_info;
        split_table_tt<sa_index_t, NUM_GPUS> split_table;
        std::array<sa_index_t, NUM_GPUS> dest_lens, src_lens;

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Multisplit);
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            if (world_rank() == gpu_index)
            {
                // printArrayss << <1, 1 >> > (gpu.prepare_S12_ptr.Isa, (sa_index_t*)gpu.prepare_S12_ptr.S12_result, gpu.pd_elements, world_rank());
                // //(0);
                kernels::write_indices _KLC_SIMPLE_(gpu.pd_elements, mcontext.get_gpu_default_stream(gpu_index))((sa_index_t*)gpu.prepare_S12_ptr.S12_result, gpu.pd_elements);
                CUERR;
                mcontext.get_device_temp_allocator(gpu_index).init(gpu.prepare_S12_ptr.S12_buffer1,
                    mpd_reserved_len * sizeof(MergeStageSuffixS12));
            }

            multi_split_node_info[gpu_index].src_keys = gpu.prepare_S12_ptr.Isa;
            multi_split_node_info[gpu_index].src_values = (sa_index_t*)gpu.prepare_S12_ptr.S12_result;
            multi_split_node_info[gpu_index].src_len = gpu.pd_elements;

            multi_split_node_info[gpu_index].dest_keys = (sa_index_t*)gpu.prepare_S12_ptr.S12_buffer2;
            multi_split_node_info[gpu_index].dest_values = (sa_index_t*)gpu.prepare_S12_ptr.S12_result_half;
            multi_split_node_info[gpu_index].dest_len = gpu.pd_elements;
        }
        S12PartitioningFunctor f(mpd_per_gpu, NUM_GPUS - 1);
        //
        mcontext.sync_default_streams();
        // comm_world().barrier();
        //
        printf("[%lu] after write indices s12\n", world_rank());
        mmulti_split.execKVAsync(multi_split_node_info, split_table, src_lens, dest_lens, f);

        mcontext.sync_default_streams();
        // comm_world().barrier();
        printf("[%lu] after execKVAsync s12\n", world_rank());
        // printArrayss << <1, 1 >> > ((sa_index_t*)mgpus[world_rank()].prepare_S12_ptr.S12_buffer2, (sa_index_t*)mgpus[world_rank()].prepare_S12_ptr.S12_result_half, mgpus[world_rank()].pd_elements, world_rank());

        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Multisplit);

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Out);
        size_t count = mgpus[world_rank()].num_elements - mgpus[world_rank()].pd_elements;


        MergeSuffixes* merge_tuple;
        cudaMalloc(&merge_tuple, sizeof(MergeSuffixes) * mgpus[world_rank()].num_elements);
        CUERR;
        MergeSuffixes* merge_tuple_out;

        uint gpu_index = world_rank();
        SaGPU& gpu = mgpus[gpu_index];

        sa_index_t* next_Isa = nullptr;      //= (gpu_index + 1 < NUM_GPUS) ? mgpus[gpu_index + 1].prepare_S12_ptr.Isa : nullptr;
        unsigned char* next_Input = nullptr; //= (gpu_index + 1 < NUM_GPUS) ? mgpus[gpu_index + 1].prepare_S12_ptr.Input : nullptr;

        ncclGroupStart();
        if (gpu_index > 0)
        {
            std::span<sa_index_t> sbIsa(gpu.prepare_S12_ptr.Isa, 1);
            NCCLCHECK(ncclSend(gpu.prepare_S12_ptr.Isa, DCX::X, ncclUint32, gpu_index - 1, mcontext.get_nccl(), mcontext.get_streams(gpu_index)[gpu_index - 1]));
            NCCLCHECK(ncclSend(gpu.prepare_S12_ptr.Input, DCX::X, ncclChar, gpu_index - 1, mcontext.get_nccl(), mcontext.get_streams(gpu_index)[gpu_index - 1]));
        }
        if (gpu_index + 1 < NUM_GPUS)
        {
            next_Isa = mcontext.get_device_temp_allocator(gpu_index).get<sa_index_t>(DCX::X);
            NCCLCHECK(ncclRecv(next_Isa, DCX::X, ncclUint32, gpu_index + 1, mcontext.get_nccl(), mcontext.get_gpu_default_stream(gpu_index)));

            next_Input = mcontext.get_device_temp_allocator(gpu_index).get<unsigned char>(DCX::X);
            NCCLCHECK(ncclRecv(next_Input, DCX::X, ncclChar, gpu_index + 1, mcontext.get_nccl(), mcontext.get_gpu_default_stream(gpu_index)));
        }
        ncclGroupEnd();

        D_DCX* dcx;
        cudaMalloc(&dcx, sizeof(D_DCX));
        cudaMemcpy(dcx->inverseSamplePosition, DCX::inverseSamplePosition, DCX::X * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dcx->nextNonSample, DCX::nextNonSample, DCX::nonSampleCount * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dcx->nextSample, DCX::nextSample, DCX::X * DCX::X * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dcx->samplePosition, DCX::samplePosition, DCX::C * sizeof(uint32_t), cudaMemcpyHostToDevice);

        kernels::prepare_SK_ind_kv _KLC_SIMPLE_(gpu.pd_elements, mcontext.get_gpu_default_stream(gpu_index))((sa_index_t*)gpu.prepare_S12_ptr.S12_result_half,
            gpu.prepare_S12_ptr.Isa, gpu.prepare_S12_ptr.Input,
            next_Isa, next_Input, gpu.offset, gpu.num_elements,
            mpd_per_gpu,
            merge_tuple, gpu.pd_elements, dcx);
        CUERR;
        // printArrayss << <1, 1 >> > (merge_tuple, mgpus[world_rank()].pd_elements, world_rank());
        mcontext.sync_all_streams();
        printf("[%lu] non samples-------------------------------------------\n", world_rank());
        comm_world().barrier();

        MergeSuffixes* nonSamples = merge_tuple + gpu.pd_elements;

        size_t noSampleCount = 0;
        for (uint32_t i = 0; i < DCX::nonSampleCount; i++) {

            size_t count2 = (count / DCX::nonSampleCount);
            if (i < count % DCX::nonSampleCount) {
                count2++;
            }
            uint l = DCX::nextSample[DCX::nextNonSample[i]][DCX::nextNonSample[i]][0];
            uint f = DCX::inverseSamplePosition[(DCX::nextNonSample[i] + l) % DCX::X];
            kernels::prepare_non_sample
                _KLC_SIMPLE_(count2, mcontext.get_gpu_default_stream(gpu_index))
                // << <1, 1, 0, mcontext.get_gpu_default_stream(gpu_index) >> >

                (gpu.prepare_S12_ptr.Isa, gpu.prepare_S12_ptr.Input, next_Isa, next_Input, gpu.offset, gpu.num_elements,
                    mpd_per_gpu,
                    nonSamples + noSampleCount, count2, DCX::nextNonSample[i], f);
            CUERR;
            noSampleCount += count2;
        }
        mcontext.sync_default_streams();
        // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(gpu_index) >> > (nonSamples, count, gpu_index);
        // mcontext.sync_default_streams();
        // comm_world().barrier();

        cudaFree(dcx);


        size_t out_num_elements;
        // SampleSort(merge_tuple, merge_tuple_out, gpu.num_elements, out_num_elements, std::min(size_t(16ULL * log(NUM_GPUS) / log(2.)), mgpus[NUM_GPUS - 1].num_elements / 2), DC7Comparator{});
        mcontext.sync_all_streams();
        printf("[%lu] sample sorted\n", world_rank());

        sa_index_t* out_sa;;
        cudaFreeAsync(merge_tuple, mcontext.get_gpu_default_stream(gpu_index));
        cudaMallocAsync(&out_sa, sizeof(sa_index_t) * out_num_elements, mcontext.get_gpu_default_stream(gpu_index));
        mcontext.sync_all_streams();
        printf("[%lu] num elements: %lu\n", world_rank(), out_num_elements);
        // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(gpu_index) >> > (merge_tuple_out, out_num_elements, gpu_index);

        kernels::write_sa _KLC_SIMPLE_(out_num_elements, mcontext.get_gpu_default_stream(gpu_index))(merge_tuple_out, reinterpret_cast<sa_index_t*>(merge_tuple_out), out_num_elements);
        mcontext.sync_all_streams();
        printf("[%lu] write sa\n", world_rank());
        sa_index_t* sa = (sa_index_t*)malloc(sizeof(sa_index_t) * out_num_elements);
        cudaMemcpyAsync(sa, reinterpret_cast<sa_index_t*>(merge_tuple_out), out_num_elements * sizeof(sa_index_t), cudaMemcpyDeviceToHost, mcontext.get_gpu_default_stream(gpu_index));
        cudaFreeAsync(merge_tuple_out, mcontext.get_gpu_default_stream(gpu_index));
        mcontext.sync_all_streams();
        comm_world().barrier();
        for (size_t i = 0; i < out_num_elements; i++)
        {
            // printf("[%lu] sa[%lu]: %u\n", world_rank(), i, sa[i]);
        }

        std::vector<size_t> recv_sizes(NUM_GPUS);
        comm_world().allgather(send_buf(std::span<size_t>(&out_num_elements, 1)), recv_buf(recv_sizes), send_count(1));
        size_t acc = std::accumulate(recv_sizes.begin(), recv_sizes.begin() + world_rank(), 0);
        printf("[%lu] acc: %lu\n", world_rank(), acc);
        int ierr;
        MPI_File outputFile;
        ierr = MPI_File_open(MPI_COMM_WORLD, "outputData",
            MPI_MODE_CREATE | MPI_MODE_WRONLY,
            MPI_INFO_NULL, &outputFile);
        if (ierr != MPI_SUCCESS) {
            fprintf(stderr, "[%lu] Error opening file\n", world_rank());
            MPI_Abort(MPI_COMM_WORLD, ierr);
        }
        MPI_Offset offset = acc * sizeof(sa_index_t);
        ierr = MPI_File_write_at_all(outputFile, offset, sa, out_num_elements, MPI_UINT32_T, MPI_STATUS_IGNORE);
        // MPI_File_write_at(outputFile, gpu.offset, h_result, gpu.num_elements, MPI_UINT32_T, MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {
            fprintf(stderr, "[%lu] Error in MPI_File_write_at_all\n", world_rank());
            MPI_Abort(MPI_COMM_WORLD, ierr);
        }
        MPI_File_close(&outputFile);

        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Out);
        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_All2All);


        //
        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Into_Place);


        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Into_Place);


    }

    void prepare_S0_for_merge()
    {
        using merge_types = crossGPUReMerge::mergeTypes<MergeStageSuffixS0HalfKey, MergeStageSuffixS0HalfValue>;
        using MergeManager = crossGPUReMerge::ReMergeManager<NUM_GPUS, merge_types, ReMergeTopology>;
        using MergeNodeInfo = crossGPUReMerge::MergeNodeInfo<merge_types>;

        auto host_temp_mem = mmemory_manager.get_host_temp_mem();

        QDAllocator host_pinned_allocator(host_temp_mem.first, host_temp_mem.second);

        std::array<MergeNodeInfo, NUM_GPUS> merge_nodes_info;

        std::array<bool, NUM_GPUS> is_buffer_2_current = { false };

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S0_Write_Out_And_Sort);

        // for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            uint gpu_index = world_rank();
            SaGPU& gpu = mgpus[gpu_index];
            // //(mcontext.get_device_id(gpu_index));
            size_t count = gpu.num_elements - gpu.pd_elements;

            kernels::prepare_S0 _KLC_SIMPLE_(count, mcontext.get_gpu_default_stream(gpu_index))(gpu.prepare_S0_ptr.Isa, gpu.prepare_S0_ptr.Input, gpu.offset,
                gpu.num_elements, gpu.pd_elements,
                gpu_index == NUM_GPUS - 1,
                reinterpret_cast<MergeStageSuffixS0HalfKey*>(gpu.prepare_S0_ptr.S0_buffer1_keys),
                gpu.prepare_S0_ptr.S0_buffer1_values,
                count);
            CUERR;
            cub::DoubleBuffer<uint64_t> keys(reinterpret_cast<uint64_t*>(gpu.prepare_S0_ptr.S0_buffer1_keys),
                reinterpret_cast<uint64_t*>(gpu.prepare_S0_ptr.S0_buffer2_keys));
            cub::DoubleBuffer<uint64_t> values(reinterpret_cast<uint64_t*>(gpu.prepare_S0_ptr.S0_buffer1_values),
                reinterpret_cast<uint64_t*>(gpu.prepare_S0_ptr.S0_buffer2_values));

            size_t temp_storage_size = 0;
            cudaError_t err = cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size, keys, values, count, 0, 40);
            CUERR_CHECK(err);
            //                printf("Needed temp storage: %zu, provided %zu.\n", temp_storage_size, ms0_reserved_len*sizeof(MergeStageSuffix));
            ASSERT(temp_storage_size <= ms0_reserved_len * sizeof(MergeStageSuffix));
            err = cub::DeviceRadixSort::SortPairs(gpu.prepare_S0_ptr.S0_result, temp_storage_size,
                keys, values, count, 0, 40, mcontext.get_gpu_default_stream(gpu_index));
            CUERR_CHECK(err);

            is_buffer_2_current[gpu_index] = keys.Current() == reinterpret_cast<uint64_t*>(gpu.prepare_S0_ptr.S0_buffer2_keys);

            // merge_nodes_info[gpu_index] = { count, ms0_reserved_len, gpu_index,
            //                                is_buffer_2_current[gpu_index] ? gpu.prepare_S0_ptr.S0_buffer2_keys
            //                                                               : gpu.prepare_S0_ptr.S0_buffer1_keys,
            //                                is_buffer_2_current[gpu_index] ? gpu.prepare_S0_ptr.S0_buffer2_values
            //                                                               : gpu.prepare_S0_ptr.S0_buffer1_values,
            //                                is_buffer_2_current[gpu_index] ? gpu.prepare_S0_ptr.S0_buffer1_keys
            //                                                               : gpu.prepare_S0_ptr.S0_buffer2_keys,
            //                                is_buffer_2_current[gpu_index] ? gpu.prepare_S0_ptr.S0_buffer1_values
            //                                                               : gpu.prepare_S0_ptr.S0_buffer2_values,
            //                                reinterpret_cast<MergeStageSuffixS0HalfKey*>(gpu.prepare_S0_ptr.S0_result),
            //                                gpu.prepare_S0_ptr.S0_result_2nd_half };

            mcontext.get_device_temp_allocator(gpu_index).init(reinterpret_cast<MergeStageSuffixS0HalfKey*>(gpu.prepare_S0_ptr.S0_result),
                ms0_reserved_len * sizeof(MergeStageSuffixS0));
        }
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            size_t count = gpu.num_elements - gpu.pd_elements;
            // send which current is used (only for in node merges)
            comm_world().bcast(send_recv_buf(std::span<bool>(&is_buffer_2_current[gpu_index], 1)), send_recv_count(1), root((size_t)gpu_index));
            merge_nodes_info[gpu_index] = { count, ms0_reserved_len, gpu_index,
                                           is_buffer_2_current[gpu_index] ? gpu.prepare_S0_ptr.S0_buffer2_keys
                                                                          : gpu.prepare_S0_ptr.S0_buffer1_keys,
                                           is_buffer_2_current[gpu_index] ? gpu.prepare_S0_ptr.S0_buffer2_values
                                                                          : gpu.prepare_S0_ptr.S0_buffer1_values,
                                           is_buffer_2_current[gpu_index] ? gpu.prepare_S0_ptr.S0_buffer1_keys
                                                                          : gpu.prepare_S0_ptr.S0_buffer2_keys,
                                           is_buffer_2_current[gpu_index] ? gpu.prepare_S0_ptr.S0_buffer1_values
                                                                          : gpu.prepare_S0_ptr.S0_buffer2_values,
                                           reinterpret_cast<MergeStageSuffixS0HalfKey*>(gpu.prepare_S0_ptr.S0_result),
                                           gpu.prepare_S0_ptr.S0_result_2nd_half };
        }

        //            dump_prepare_s0("Before S0 merge");

        MergeManager merge_manager(mcontext, host_pinned_allocator);

        merge_manager.set_node_info(merge_nodes_info);

        std::vector<crossGPUReMerge::MergeRange> ranges;
        ranges.push_back({ 0, 0, (sa_index_t)NUM_GPUS - 1, (sa_index_t)(mgpus.back().num_elements - mgpus.back().pd_elements) });

        mcontext.sync_default_stream_mpi_safe();
        printf("[%lu] after S0_Write_Out_And_Sort s0\n", world_rank());
        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S0_Write_Out_And_Sort);

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S0_Merge);
        merge_manager.merge(ranges, S0Comparator());

        mcontext.sync_all_streams_mpi_safe();
        comm_world().barrier();
        printf("[%lu] after merge s0\n", world_rank());
        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S0_Merge);

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S0_Combine);

        // for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            uint gpu_index = world_rank();
            SaGPU& gpu = mgpus[gpu_index];
            //(mcontext.get_device_id(gpu_index));

            size_t count = gpu.num_elements - gpu.pd_elements;

            const MergeStageSuffixS0HalfKey* sorted_and_merged_keys = is_buffer_2_current[gpu_index] ? gpu.prepare_S0_ptr.S0_buffer2_keys : gpu.prepare_S0_ptr.S0_buffer1_keys;

            const MergeStageSuffixS0HalfValue* sorted_and_merged_values = is_buffer_2_current[gpu_index] ? gpu.prepare_S0_ptr.S0_buffer2_values : gpu.prepare_S0_ptr.S0_buffer1_values;

            kernels::combine_S0_kv _KLC_SIMPLE_(count, mcontext.get_gpu_default_stream(gpu_index))(sorted_and_merged_keys, sorted_and_merged_values, gpu.prepare_S0_ptr.S0_result, count);
            CUERR;
        }
        mcontext.sync_default_stream_mpi_safe();
        printf("[%lu] after s0\n", world_rank());
        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S0_Combine);
        //            dump_final_merge("before final merge");
    }

    void final_merge()
    {
        distrib_merge::DistributedArray<MergeStageSuffix, int, sa_index_t, NUM_GPUS> inp_S12, inp_S0, result;

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {

            SaGPU& gpu = mgpus[gpu_index];

            const size_t S0_count = gpu.num_elements - gpu.pd_elements;
            const size_t S12_count = gpu.pd_elements;
            const size_t result_count = gpu.num_elements;
            inp_S12[gpu_index] = { gpu_index, (sa_index_t)S12_count, gpu.merge_ptr.S12_result, nullptr, nullptr, nullptr };
            inp_S0[gpu_index] = { gpu_index, (sa_index_t)S0_count, gpu.merge_ptr.S0_result, nullptr, nullptr, nullptr };
            result[gpu_index] = { gpu_index, (sa_index_t)result_count, gpu.merge_ptr.S12_result, nullptr, gpu.merge_ptr.buffer, nullptr };
            if (world_rank() == gpu_index)
            {
                mcontext.get_device_temp_allocator(gpu_index).init(gpu.merge_ptr.remaining_storage,
                    gpu.merge_ptr.remaining_storage_size);
            }
        }
        printf("[%lu] final merge\n", world_rank());
        auto h_temp_mem = mmemory_manager.get_host_temp_mem();
        QDAllocator qd_alloc_h_temp(h_temp_mem.first, h_temp_mem.second);
        distrib_merge::DistributedMerge<MergeStageSuffix, int, sa_index_t, NUM_GPUS, DistribMergeTopology>::
            merge_async(inp_S12, inp_S0, result, MergeCompFunctor(), false, mcontext, qd_alloc_h_temp);

        mcontext.sync_default_streams();
        printf("[%lu] after merge_async\n", world_rank());

        // printf("[%lu] merge async done\n", world_rank());
        // comm_world().barrier();
        //            dump_final_merge("after final merge");

        // for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            uint gpu_index = world_rank();
            SaGPU& gpu = mgpus[gpu_index];
            // //(mcontext.get_device_id(gpu_index));
            kernels::from_merge_suffix_to_index _KLC_SIMPLE_(gpu.num_elements, mcontext.get_gpu_default_stream(gpu_index))(gpu.merge_ptr.S12_result, gpu.merge_ptr.result, gpu.num_elements);
            CUERR;
        }
        mcontext.sync_default_streams();
    }

    void copy_result_to_host()
    {
        sa_index_t* h_result = mmemory_manager.get_h_result();
        // for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        //{
        uint gpu_index = world_rank();
        SaGPU& gpu = mgpus[gpu_index];
        //(mcontext.get_device_id(gpu_index));
        cudaMemcpyAsync(h_result, gpu.merge_ptr.result, gpu.num_elements * sizeof(sa_index_t),
            cudaMemcpyDeviceToHost, mcontext.get_gpu_default_stream(gpu_index));
        CUERR;
        mcontext.sync_gpu_default_stream(gpu_index);

        MPI_File outputFile;
        MPI_File_open(MPI_COMM_WORLD, "outputData",
            MPI_MODE_CREATE | MPI_MODE_WRONLY,
            MPI_INFO_NULL, &outputFile);
        MPI_File_write_at_all(outputFile, gpu.offset, h_result, gpu.num_elements, MPI_UINT32_T, MPI_STATUS_IGNORE);
        // MPI_File_write_at(outputFile, gpu.offset, h_result, gpu.num_elements, MPI_UINT32_T, MPI_STATUS_IGNORE);

        MPI_File_close(&outputFile);

        //}
        // mcontext.sync_default_streams();

        // std::vector<sa_index_t> recv;
        // recv.clear();
        // std::span<sa_index_t> sb(h_result + gpu.offset, gpu.num_elements);
        // auto [sendCounts] = comm_world().gatherv(send_buf(sb), recv_buf<resize_to_fit>(recv), recv_counts_out());
        // int sumCounts = 0;
        // int i = 0;
        // for (auto count : sendCounts) {
        //     ASSERT(count == mgpus[i].num_elements);
        //     memcpy(h_result + mgpus[i].offset, recv.data() + sumCounts, sizeof(sa_index_t) * count);
        //     sumCounts += count;
        //     i++;
        // }
        // for (int gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index) {
        //     std::span<sa_index_t> buffer(h_result + gpu.offset, gpu.num_elements);
        //     comm_world().bcast(send_recv_buf(buffer), root(gpu_index));
        // }
        // std::span<sa_index_t> rb(h_result, );
    }

#ifdef ENABLE_DUMPING
    static inline void print_merge12(sa_index_t index, const MergeStageSuffixS12HalfKey& s12k,
        const MergeStageSuffixS12HalfValue& s12v)
    {
        printf("%7u. Index: %7u, own rank: %7u, rank +1/+2: %7u, c: %2x (%c), c[i+1]: %2x (%c)\n",
            index, s12k.index, s12k.own_rank, s12v.rank_p1p2, s12v.chars[0], s12v.chars[0],
            s12v.chars[1], s12v.chars[1]);
    }

    static inline void print_merge0_half(sa_index_t index, const MergeStageSuffixS0HalfKey& s0k,
        const MergeStageSuffixS0HalfValue& s0v)
    {
        printf("%7u. Index: %7u, first char: %2x (%c), c[i+1]: %2x (%c), rank[i+1]: %7u, rank[i+2]: %7u\n",
            index, s0v.index, s0k.chars[0], s0k.chars[0], s0k.chars[1], s0k.chars[1],
            s0k.rank_p1, s0v.rank_p2);
    }

    static inline void print_final_merge_suffix(sa_index_t index, const MergeStageSuffix& suff)
    {
        printf("%7u. Index: %7u, first char: %2x (%c), c[i+1]: %2x (%c), rank[i+1]: %7u, rank[i+2]: %7u\n",
            index, suff.index, suff.chars[0], suff.chars[0], suff.chars[1], suff.chars[1],
            suff.rank_p1, suff.rank_p2);
    }

    void dump_prepare_s12(const char* caption = nullptr)
    {
        if (caption)
        {
            printf("\n%s:\n", caption);
        }
        for (uint g = 0; g < NUM_GPUS; ++g)
        {
            mmemory_manager.copy_down_for_inspection(g);
            printf("\nGPU %u:\nBuffer1:\n", g);
            size_t limit = mgpus[g].pd_elements;
            const PrepareS12Arrays& arr = mmemory_manager.get_host_prepare_S12_arrays();
            for (int i = 0; i < limit; ++i)
            {
                print_merge12(i, arr.S12_buffer1[i], arr.S12_buffer1_half[i]);
            }
            printf("Buffer2:\n");
            for (int i = 0; i < limit; ++i)
            {
                print_merge12(i, arr.S12_buffer2[i], arr.S12_buffer2_half[i]);
            }
            printf("Result-buffer:\n");
            for (int i = 0; i < limit; ++i)
            {
                print_final_merge_suffix(i, arr.S12_result[i]);
            }
        }
    }

    void dump_prepare_s0(const char* caption = nullptr)
    {
        if (caption)
        {
            printf("\n%s:\n", caption);
        }
        for (uint g = 0; g < NUM_GPUS; ++g)
        {
            mmemory_manager.copy_down_for_inspection(g);
            printf("\nGPU %u:\nBuffer1:\n", g);
            size_t limit = mgpus[g].num_elements - mgpus[g].pd_elements;
            const PrepareS0Arrays& arr = mmemory_manager.get_host_prepare_S0_arrays();
            for (int i = 0; i < limit; ++i)
            {
                print_merge0_half(i, arr.S0_buffer1_keys[i], arr.S0_buffer1_values[i]);
            }
            printf("Buffer2:\n");
            for (int i = 0; i < limit; ++i)
            {
                print_merge0_half(i, reinterpret_cast<const MergeStageSuffixS0HalfKey*>(arr.S0_buffer2_keys)[i],
                    arr.S0_buffer2_values[i]);
            }
            printf("Result-buffer:\n");
            for (int i = 0; i < limit; ++i)
            {
                print_final_merge_suffix(i, arr.S0_result[i]);
            }
        }
    }

    void dump_final_merge(const char* caption = nullptr)
    {
        if (caption)
        {
            printf("\n%s:\n", caption);
        }
        for (uint g = 0; g < NUM_GPUS; ++g)
        {
            SaGPU& gpu = mgpus[g];

            mmemory_manager.copy_down_for_inspection(g);

            printf("\nGPU %u:\nS12_result:\n", g);
            const MergeS12S0Arrays& arr = mmemory_manager.get_host_merge_S12_S0_arrays();

            for (int i = 0; i < gpu.pd_elements; ++i)
            {
                if (i == 10 && gpu.pd_elements > 20)
                    i = gpu.pd_elements - 10;
                print_final_merge_suffix(i, arr.S12_result[i]);
            }
            printf("S0_result:\n");
            for (int i = 0; i < gpu.num_elements - gpu.pd_elements; ++i)
            {
                if (i == 10 && (gpu.num_elements - gpu.pd_elements) > 20)
                    i = (gpu.num_elements - gpu.pd_elements) - 10;
                print_final_merge_suffix(i, arr.S0_result[i]);
            }
            //                printf("Buffer:\n");
            //                for (int i = 0; i < gpu.num_elements; ++i) {
            //                    if (i == 10 && gpu.num_elements > 20)
            //                        i = gpu.num_elements-10;
            //                    print_final_merge_suffix(i, arr.buffer[i]);
            //                }
        }
    }
#endif


};

void print_device_info()
{
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
            2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Major.minor: %d.%d\n",
            prop.major, prop.minor);
        printf("  Max grid size: %d, %d, %d\n",
            prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Max threads dim (per block): %d, %d, %d\n",
            prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max thread per block: %d\n",
            prop.maxThreadsPerBlock);
        printf("  Warp size: %d\n",
            prop.warpSize);
        printf("  Global mem: %zd kB\n",
            prop.totalGlobalMem / 1024);
        printf("  Const mem: %zd kB\n",
            prop.totalConstMem / 1024);
        printf("  Asynchronous engines: %d\n",
            prop.asyncEngineCount);
        printf("  Unified addressing: %d\n",
            prop.unifiedAddressing);
    }
}

void ncclMeasure(MultiGPUContext<NUM_GPUS>& context)
{
    using namespace kamping;
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<std::mt19937::result_type> randomDist(0, UINT32_MAX);
    const int rounds = 26;
    std::array<double, rounds> alg_bandwidth;
    ncclComm_t nccl_comm = context.get_nccl();
    const int start_offset = 0;
    for (int i = 0; i < rounds; i++)
    {

        size_t per_gpu = NUM_GPUS << i;
        sa_index_t* A = (sa_index_t*)malloc(per_gpu * sizeof(sa_index_t));
        for (size_t j = 0; j < per_gpu; j++)
        {
            A[j] = randomDist(g);
        }
        sa_index_t* d_A_send;
        sa_index_t* d_A_recv;
        sa_index_t send_size = per_gpu / NUM_GPUS;
        cudaMalloc(&d_A_send, sizeof(sa_index_t) * per_gpu);
        cudaMalloc(&d_A_recv, sizeof(sa_index_t) * per_gpu);
        cudaMemset(d_A_recv, 0, sizeof(sa_index_t) * per_gpu);
        cudaMemcpy(d_A_send, A, sizeof(sa_index_t) * per_gpu, cudaMemcpyHostToDevice);

        // warm up
        for (int loop = 0; loop < 1; loop++)
        {
            ncclGroupStart();
            for (size_t src_gpu = 0; src_gpu < NUM_GPUS; src_gpu++)
            {
                for (size_t dst_gpu = 0; dst_gpu < NUM_GPUS; dst_gpu++)
                {
                    if (src_gpu == world_rank())
                    {
                        ncclSend(d_A_send + dst_gpu * send_size, sizeof(sa_index_t) * send_size, ncclChar, dst_gpu, nccl_comm, context.get_streams(src_gpu)[dst_gpu]);
                    }
                    if (dst_gpu == world_rank())
                    {
                        ncclRecv(d_A_recv + src_gpu * send_size, sizeof(sa_index_t) * send_size, ncclChar, src_gpu, nccl_comm, context.get_streams(dst_gpu)[src_gpu]);
                    }
                }
            }
            ncclGroupEnd();
            context.sync_all_streams();
            comm_world().barrier();
            cudaMemset(d_A_recv, 0, sizeof(sa_index_t) * per_gpu);
        }

        const size_t loop_count = 10;
        std::array<double, loop_count> loop_time;

        for (size_t loop = 0; loop < loop_count; loop++)
        {
            double start = MPI_Wtime();
            ncclGroupStart();
            for (size_t src_gpu = 0; src_gpu < NUM_GPUS; src_gpu++)
            {
                for (size_t dst_gpu = 0; dst_gpu < NUM_GPUS; dst_gpu++)
                {
                    if (src_gpu == world_rank())
                    {
                        ncclSend(d_A_send + dst_gpu * send_size, sizeof(sa_index_t) * send_size, ncclChar, dst_gpu, nccl_comm, context.get_streams(src_gpu)[dst_gpu]);
                    }
                    if (dst_gpu == world_rank())
                    {
                        ncclRecv(d_A_recv + src_gpu * send_size, sizeof(sa_index_t) * send_size, ncclChar, src_gpu, nccl_comm, context.get_streams(dst_gpu)[src_gpu]);
                    }
                }
            }
            ncclGroupEnd();
            context.sync_all_streams();
            comm_world().barrier();
            double end = MPI_Wtime();
            loop_time[loop] = end - start;
            cudaMemset(d_A_recv, 0, sizeof(sa_index_t) * per_gpu);
        }
        std::sort(loop_time.begin(), loop_time.end(), std::less_equal<double>());
        double elapsed_time = loop_time[0];
        for (int j = 1; j < loop_count; j++)
        {
            elapsed_time += loop_time[j];
        }
        size_t num_B = sizeof(sa_index_t) * per_gpu;
        size_t B_in_GB = 1 << 30;

        double num_GB = (double)num_B / (double)B_in_GB;
        double avg_time_per_transfer = elapsed_time / ((double)loop_count);
        alg_bandwidth[i - start_offset] = num_GB / avg_time_per_transfer;
        printf("[%lu] Transfer size (B): %10li, Transfer Time Avg|Min|Max (s): %15.9f %15.9f %15.9f, Bandwidth (GB/s): %15.9f\n", world_rank(), num_B, avg_time_per_transfer, loop_time.front(), loop_time.back(), alg_bandwidth[i - start_offset]);
        comm_world().barrier();
        cudaFree(d_A_send);
        cudaFree(d_A_recv);
        free(A);
        comm_world().barrier();
    }
    if (world_rank() == 0)
    {
        std::ofstream outFile("ncclBandwidthAllToAll8", std::ios::binary);
        if (!outFile)
        {
            std::cerr << "Write Error" << std::endl;
            return;
        }

        outFile.write(reinterpret_cast<char*>(alg_bandwidth.data()), rounds * sizeof(double));
        outFile.close();
    }
}

void alltoallMeasure(MultiGPUContext<NUM_GPUS>& context)
{
    using namespace kamping;
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<std::mt19937::result_type> randomDist(0, UINT32_MAX);
    const int rounds = 28;
    const int start_offset = 0;
    std::array<double, rounds> alg_bandwidth;
    for (int iter = 0; iter < rounds; iter++)
    {
        printf("[%lu] iter: %d\n", world_rank(), iter);
        MultiSplit<NUM_GPUS> multi_split(context);
        All2All<NUM_GPUS> all2all(context);
        std::array<sa_index_t*, NUM_GPUS> d_A_send;
        std::array<void*, NUM_GPUS> temp_buffer;
        std::array<sa_index_t*, NUM_GPUS> d_A_recv;
        size_t N = (NUM_GPUS * NUM_GPUS) << iter;

        // Allocate memory for A on CPU
        sa_index_t* A = (sa_index_t*)malloc(N * sizeof(sa_index_t));
        size_t per_gpu = (N / NUM_GPUS);
        if (world_rank() == 0)
        {

            // Initialize all elements of A to random values
            for (size_t j = 0; j < N; j++)
            {
                A[j] = randomDist(g);
            }

            for (size_t gpu_index = 1; gpu_index < NUM_GPUS; gpu_index++)
            {
                comm_world().send(send_buf(std::span<sa_index_t>(A + gpu_index * per_gpu, per_gpu)), send_count(per_gpu), tag(gpu_index), destination(gpu_index));
            }
        }
        else
        {
            comm_world().recv(recv_buf(std::span<sa_index_t>(A + world_rank() * per_gpu, per_gpu)), recv_count(per_gpu), tag(world_rank()), source(0));
        }

        std::array<size_t, NUM_GPUS> temp_storages;
        {
            size_t gpu_index = world_rank();
            // cudaStream_t stream = context.get_gpu_default_stream(i);
            sa_index_t* d_A, * d_A_rec;
            cudaMalloc(&d_A, per_gpu * sizeof(sa_index_t));
            CUERR;
            d_A_send[gpu_index] = d_A;
            cudaMemcpy(d_A_send[gpu_index], A + per_gpu * gpu_index, per_gpu * sizeof(sa_index_t), cudaMemcpyHostToDevice);
            CUERR;

            cudaMalloc(&d_A_rec, per_gpu * sizeof(sa_index_t));
            CUERR;
            d_A_recv[gpu_index] = d_A_rec;
            // cub::DeviceRadixSort::SortKeys(nullptr, temp_storages[gpu_index],
            //     d_A_send[gpu_index], d_A_recv[gpu_index], per_gpu);
            // void* temp;
            // temp_storages[gpu_index] = std::max(temp_storages[gpu_index], 1024ul);
            // temp_storages[gpu_index] = std::max(temp_storages[gpu_index], ((size_t)per_gpu) * sizeof(sa_index_t)) * 4;
            // cudaMalloc(&temp, temp_storages[gpu_index]);
            // CUERR;
            // temp_buffer[gpu_index] = temp;
            // cudaMemset(temp_buffer[gpu_index], 0, temp_storages[gpu_index]);
            // CUERR;

            cudaMemset(d_A_recv[gpu_index], 0, per_gpu * sizeof(sa_index_t));
            CUERR;
        }
        comm_world().barrier();

        cudaIpcMemHandle_t handleSend;
        cudaIpcGetMemHandle(&handleSend, d_A_send[world_rank()]);
        cudaIpcMemHandle_t handleRecv;
        cudaIpcGetMemHandle(&handleRecv, d_A_recv[world_rank()]);
        for (size_t dst = 0; dst < NUM_GPUS; dst++) {
            if (context.get_peer_status(world_rank(), dst) != 1) {
                continue;
            }
            comm_world().isend(send_buf(std::span<cudaIpcMemHandle_t>(&handleSend, 1)), send_count(1), tag(0), destination(dst));
            comm_world().isend(send_buf(std::span<cudaIpcMemHandle_t>(&handleRecv, 1)), send_count(1), tag(1), destination(dst));
        }
        for (size_t src = 0; src < NUM_GPUS; src++) {
            if (context.get_peer_status(world_rank(), src) != 1) {
                continue;
            }
            cudaIpcMemHandle_t other_handleSend;
            cudaIpcMemHandle_t other_handleRecv;
            comm_world().recv(recv_buf(std::span<cudaIpcMemHandle_t>(&other_handleSend, 1)), recv_count(1), tag(0), source(src));
            comm_world().recv(recv_buf(std::span<cudaIpcMemHandle_t>(&other_handleRecv, 1)), recv_count(1), tag(1), source(src));
            void* ptrHandleSend;
            void* ptrHandleRecv;
            cudaIpcOpenMemHandle(&ptrHandleSend, other_handleSend, cudaIpcMemLazyEnablePeerAccess);
            CUERR;
            cudaIpcOpenMemHandle(&ptrHandleRecv, other_handleRecv, cudaIpcMemLazyEnablePeerAccess);
            CUERR;

            printf("[%lu] opened mem handles from %d\n", world_rank(), src);
            d_A_send[src] = reinterpret_cast<sa_index_t*>(ptrHandleSend);
            d_A_recv[src] = reinterpret_cast<sa_index_t*>(ptrHandleRecv);
        }

        context.sync_default_streams();
        comm_world().barrier();

        std::array<All2AllNodeInfoT<sa_index_t, sa_index_t, sa_index_t>, NUM_GPUS> all2all_node_info;

        split_table_tt<sa_index_t, NUM_GPUS> split_table;
        // std::array<MultiSplitNodeInfoT<sa_index_t, sa_index_t, sa_index_t>, NUM_GPUS> multi_split_node_info;
        // std::array<sa_index_t, NUM_GPUS> dest_lens, src_lens;
        // for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        // {
        //     multi_split_node_info[gpu_index].src_keys = d_A_send[gpu_index];
        //     multi_split_node_info[gpu_index].src_values = d_A_send[gpu_index];
        //     multi_split_node_info[gpu_index].src_len = per_gpu;

        //     multi_split_node_info[gpu_index].dest_keys = d_A_recv[gpu_index];
        //     multi_split_node_info[gpu_index].dest_values = d_A_recv[gpu_index];
        //     multi_split_node_info[gpu_index].dest_len = per_gpu;
        //     if (world_rank() == gpu_index)
        //     {
        //         context.get_device_temp_allocator(gpu_index).init(temp_buffer[gpu_index], temp_storages[gpu_index]);
        //     }
        // }

        // PartitioningFunctor<sa_index_t> f(per_gpu, NUM_GPUS - 1);
        // multi_split.execAsync(multi_split_node_info, split_table, src_lens, dest_lens, f);
        size_t send_size = per_gpu / NUM_GPUS;
        for (uint src = 0; src < NUM_GPUS; ++src)
        {
            for (uint dst = 0; dst < NUM_GPUS; ++dst)
            {
                split_table[src][dst] = send_size;
            }
        }

        context.sync_default_streams();

        comm_world().barrier();
        context.sync_all_streams();

        // Warm-up loop
        for (int j = 0; j < 1; j++)
        {
            for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
            {
                all2all_node_info[gpu_index].src_keys = d_A_send[gpu_index];
                all2all_node_info[gpu_index].src_values = d_A_send[gpu_index];
                all2all_node_info[gpu_index].src_len = per_gpu;

                all2all_node_info[gpu_index].dest_keys = d_A_recv[gpu_index];
                all2all_node_info[gpu_index].dest_values = d_A_recv[gpu_index];
                all2all_node_info[gpu_index].dest_len = per_gpu;
            }
            all2all.execAsync(all2all_node_info, split_table);
            context.sync_all_streams();
            cudaMemset(d_A_recv[world_rank()], 0, sizeof(sa_index_t) * per_gpu);
            comm_world().barrier();
        }
        context.sync_all_streams();
        comm_world().barrier();

        // Time ping-pong for loop_count iterations of data transfer size 8*N bytes
        const int loop_count = 10;
        std::array<double, loop_count> loop_time;
        for (int j = 0; j < loop_count; j++)
        {
            double start = MPI_Wtime();
            for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
            {
                all2all_node_info[gpu_index].src_keys = d_A_send[gpu_index];
                all2all_node_info[gpu_index].src_values = d_A_send[gpu_index];
                all2all_node_info[gpu_index].src_len = per_gpu;

                all2all_node_info[gpu_index].dest_keys = d_A_recv[gpu_index];
                all2all_node_info[gpu_index].dest_values = d_A_recv[gpu_index];
                all2all_node_info[gpu_index].dest_len = per_gpu;
            }
            all2all.execAsync(all2all_node_info, split_table);
            context.sync_all_streams();
            comm_world().barrier();
            double end = MPI_Wtime();
            cudaMemset(d_A_recv[world_rank()], 0, sizeof(sa_index_t) * per_gpu);
            loop_time[j] = end - start;
        }

        std::sort(loop_time.begin(), loop_time.end(), std::less_equal<double>());
        double elapsed_time = loop_time[0];
        for (int j = 1; j < loop_count; j++)
        {
            elapsed_time += loop_time[j];
        }
        size_t num_B = sizeof(sa_index_t) * per_gpu;
        size_t B_in_GB = 1 << 30;

        double num_GB = (double)num_B / (double)B_in_GB;
        double avg_time_per_transfer = elapsed_time / ((double)loop_count);
        alg_bandwidth[iter - start_offset] = num_GB / avg_time_per_transfer;
        printf("[%lu] Transfer size (B): %10li, Transfer Time Avg|Min|Max (s): %15.9f %15.9f %15.9f, Bandwidth (GB/s): %15.9f\n", world_rank(), num_B, avg_time_per_transfer, loop_time.front(), loop_time.back(), alg_bandwidth[iter - start_offset]);
        // comm_world().barrier();
        // cudaMemcpy(A, d_A_send[world_rank()], per_gpu * sizeof(sa_index_t), cudaMemcpyDeviceToHost);
        // CUERR;
        // std::sort(A, A + per_gpu, std::less<sa_index_t>());
        // for(sa_index_t j = 0; j < per_gpu; j++){
        //     if(A[j]-per_gpu*world_rank() != j){
        //         printf("[%lu] A[%u] %u wrong\n", world_rank(), j, A[j]);
        //         break;
        //     }
        // }
        comm_world().barrier();

        cudaFree(d_A_send[world_rank()]);
        cudaFree(d_A_recv[world_rank()]);
        // cudaFree(temp_buffer[gpu_index]);
        free(A);
        comm_world().barrier();
    }

    if (world_rank() == 0)
    {
        std::ofstream outFile("algoBandwidth8", std::ios::binary);
        if (!outFile)
        {
            std::cerr << "Write Error" << std::endl;
            return;
        }

        outFile.write(reinterpret_cast<char*>(alg_bandwidth.data()), rounds * sizeof(double));
        outFile.close();
    }
}

template<typename T>
void share_gpu_ptr(std::array<T*, NUM_GPUS>& ptrs, MultiGPUContext<NUM_GPUS>& context) {

    cudaIpcMemHandle_t handleSend;
    cudaIpcGetMemHandle(&handleSend, ptrs[world_rank()]);

    for (size_t dst = 0; dst < NUM_GPUS; dst++) {
        if (context.get_peer_status(world_rank(), dst) != 1) {
            continue;
        }
        comm_world().isend(send_buf(std::span<cudaIpcMemHandle_t>(&handleSend, 1)), send_count(1), tag(0), destination(dst));
    }
    for (size_t src = 0; src < NUM_GPUS; src++) {
        if (context.get_peer_status(world_rank(), src) != 1) {
            continue;
        }
        cudaIpcMemHandle_t other_handleRecv;
        comm_world().recv(recv_buf(std::span<cudaIpcMemHandle_t>(&other_handleRecv, 1)), recv_count(1), tag(0), source(src));
        void* ptrHandleRecv;

        cudaIpcOpenMemHandle(&ptrHandleRecv, other_handleRecv, cudaIpcMemLazyEnablePeerAccess);
        CUERR;

        // printf("[%lu] opened mem handles from %d\n", world_rank(), src);
        ptrs[src] = reinterpret_cast<T*>(ptrHandleRecv);
    }
}

void sample_sort_merge_measure(MultiGPUContext<NUM_GPUS>& mcontext) {
    using merge_types = crossGPUReMerge::mergeTypes<uint64_t, uint64_t>;
    using MergeManager = crossGPUReMerge::ReMergeManager<NUM_GPUS, merge_types, ReMergeTopology>;
    using MergeNodeInfo = crossGPUReMerge::MergeNodeInfo<merge_types>;

    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<std::mt19937::result_type> randomDistChar(0, UINT64_MAX);
    size_t rounds = 22;
    for (size_t i = 0; i < rounds; i++)
    {
        size_t data_size = (128UL << i) - 1UL;
        std::array<MergeNodeInfo, NUM_GPUS> merge_nodes_info;

        std::vector<uint64_t> h_keys(data_size);
        std::vector<uint64_t> h_values(data_size);
        for (size_t i = 0; i < data_size; i++)
        {
            h_keys[i] = randomDistChar(g);
            h_values[i] = randomDistChar(g);
        }
        uint64_t* d_keys;
        cudaMalloc(&d_keys, sizeof(uint64_t) * 2 * data_size);
        cudaMemcpy(d_keys, h_keys.data(), data_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
        uint64_t* d_values;
        cudaMalloc(&d_values, sizeof(uint64_t) * 2 * data_size);
        // cudaMemset(d_values, 0, sizeof(uint64_t) * data_size);
        cudaMemcpy(d_values, h_values.data(), data_size * sizeof(uint64_t), cudaMemcpyHostToDevice);

        std::array<uint64_t*, NUM_GPUS> d_keys_gpu;
        std::array<uint64_t*, NUM_GPUS> d_values_gpu;
        d_keys_gpu[world_rank()] = d_keys;
        d_values_gpu[world_rank()] = d_values;
        share_gpu_ptr(d_keys_gpu, mcontext);
        comm_world().barrier();
        share_gpu_ptr(d_values_gpu, mcontext);
        comm_world().barrier();

        size_t temp_storage_size = 0;
        void* temp;
        temp_storage_size = sizeof(uint64_t) * data_size * 2;
        cudaMalloc(&temp, temp_storage_size);
        mcontext.get_device_temp_allocator(world_rank()).init(temp, temp_storage_size);

        uint64_t* h_temp_mem = (uint64_t*)malloc(temp_storage_size);
        memset(h_temp_mem, 0, temp_storage_size);
        QDAllocator host_pinned_allocator(h_temp_mem, temp_storage_size);

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)
        {
            merge_nodes_info[gpu_index] = { data_size, 0, gpu_index, d_keys_gpu[gpu_index], d_values_gpu[gpu_index] , d_keys_gpu[gpu_index] + data_size,  d_values_gpu[gpu_index] + data_size,  nullptr, nullptr };
        }

        MergeManager merge_manager(mcontext, host_pinned_allocator);
        merge_manager.set_node_info(merge_nodes_info);

        std::vector<crossGPUReMerge::MergeRange> ranges;
        ranges.push_back({ 0, 0, (sa_index_t)NUM_GPUS - 1, (sa_index_t)(data_size) });
        mcontext.sync_all_streams();
        auto& t = kamping::measurements::timer();
        char sf[30];
        size_t bytes = sizeof(uint64_t) * data_size;
        sprintf(sf, "sample_sort_%lu", bytes);
        t.synchronize_and_start(sf);
        t.start("init_sort");
        mcontext.get_mgpu_default_context_for_device(world_rank()).set_device_temp_mem(temp, temp_storage_size);
        mgpu::mergesort(d_keys, data_size, std::less<uint64_t>(), mcontext.get_mgpu_default_context_for_device(world_rank()));
        // err = cub::DeviceRadixSort::SortPairs(temp, temp_storage_size,
        //     d_keys, d_keys + data_size, d_values, d_values + data_size, data_size, 0, sizeof(uint64_t) * 8, mcontext.get_gpu_default_stream(world_rank()));
        // CUERR_CHECK(err);
        mcontext.sync_all_streams();
        mcontext.get_mgpu_default_context_for_device(world_rank()).reset_temp_memory();
        t.stop();
        t.start("merge");
        merge_manager.merge(ranges, std::less<uint64_t>());
        mcontext.sync_all_streams();
        t.stop();
        comm_world().barrier();
        t.stop_and_append();
        size_t mb = 1 << 20;
        double num_mB = (double)bytes / (double)mb;
        if (world_rank() == 0)
            printf("[%lu] elements: %lu, %8.3f MB\n", world_rank(), data_size, num_mB);
        // cudaMemcpy(h_temp_mem, d_keys + data_size, sizeof(uint64_t) * data_size, cudaMemcpyDeviceToHost);
        // for (size_t i = 0; i < data_size; i++)
        // {
        //     printf("[%lu] merged key[%3lu]: %20lu\n", world_rank(), i, h_temp_mem[i]);
        // }

        free(h_temp_mem);
        cudaFree(temp);
        cudaFree(d_keys);
        cudaFree(d_values);
    }
}


int main(int argc, char** argv)
{
    using namespace kamping;
    kamping::Environment e;
    Communicator comm;
    ncclComm_t nccl_comm;
    ncclUniqueId Id;
    int devices;



    cudaGetDeviceCount(&devices);
    printf("[%lu] device count: %d\n", world_rank(), devices);
    if (devices == 0)
    {
        printf("[%lu] No GPU found\n", world_rank());
        return 0;
    }
    cudaSetDevice(world_rank() % (size_t)devices);
    CUERR;

    if (world_rank() == 0)
    {
        NCCLCHECK(ncclGetUniqueId(&Id));
        comm_world().bcast_single(send_recv_buf(Id));
    }
    else
    {
        Id = comm_world().bcast_single<ncclUniqueId>();
    }

    NCCLCHECK(ncclCommInitRank(&nccl_comm, world_size(), Id, world_rank()));
    printf("[%lu] Active nccl comm\n", world_rank());

    if (argc != 3)
    {
        error("Usage: sa-test <ofile> <ifile> !");
    }
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::uniform_int_distribution<std::mt19937::result_type> randomDist(0, 10000);
        thrust::host_vector<int> samp(4);
        for (size_t i = 0; i < samp.size(); i++)
        {
            samp[i] = randomDist(g);
        }
        thrust::sort(samp.begin(), samp.end());
        thrust::device_vector<int> splitter = samp;
        std::vector<int> sizes(splitter.size());


        thrust::host_vector<int> a(10);
        for (size_t i = 0; i < 10; i++)
        {
            a[i] = randomDist(g);
        }
        thrust::device_vector<int> d_ints = a;
        thrust::host_vector<size_t> vec(d_ints.size());
        thrust::host_vector<thrust::device_vector<int>> buckets(4);
        for (size_t i = 0; i < buckets.size(); i++) buckets[i].reserve(d_ints.size() / 2);

        for (size_t i = 0; i < d_ints.size(); i++)
        {
            // const auto da = thrust::raw_pointer_cast(d_ints.data())[i];
            const auto a = thrust::upper_bound(splitter.begin(), splitter.end(), thrust::raw_pointer_cast(d_ints.data())[i]);
            const auto idx = std::min(size_t(a - splitter.begin()), sizes.size() - 1);
            std::cout << idx << std::endl;
            buckets[idx].push_back(d_ints[i]);
            // sizes[vec[i]]++;
        }
        for (int i = 0; i < splitter.size(); i++) {
            std::cout << "splitter(" << i << ") =  " << splitter[i] << std::endl;
        }
        for (int i = 0; i < buckets.size(); i++) {
            std::cout << "bucket(" << i << ") =  " << std::endl;
            for (size_t j = 0; j < buckets[i].size();j++)
            {
                std::cout << "[" << j << "]" << buckets[i][j] << "\n";
            }
        }
    }
    // for (int i = 0; i < 2; i++)
    // {

    comm_world().barrier();
    char* input = nullptr;

    size_t realLen = 0;
    // size_t maxLength = size_t(1024 * 1024) * size_t(900 * NUM_GPUS);
    // size_t inputLen = read_file_into_host_memory(&input, argv[2], realLen, sizeof(sa_index_t), maxLength, NUM_GPUS, 0);
    comm.barrier();
    CUERR;

#ifdef DGX1_TOPOLOGY
    //    const std::array<uint, NUM_GPUS> gpu_ids { 0, 3, 2, 1, 5, 6, 7, 4 };
    //    const std::array<uint, NUM_GPUS> gpu_ids { 1, 2, 3, 0, 4, 7, 6, 5 };
    //    const std::array<uint, NUM_GPUS> gpu_ids { 3, 2, 1, 0, 4, 5, 6, 7 };
    const std::array<uint, NUM_GPUS> gpu_ids{ 3, 2, 1, 0, 4, 7, 6, 5 };

    MultiGPUContext<NUM_GPUS> context(&gpu_ids);
#else
    // std::array<uint, NUM_GPUS> gpu_ids2{ 0,0,0,0 };


    MultiGPUContext<NUM_GPUS> context(nccl_comm, nullptr, NUM_GPUS_PER_NODE);

    // sample_sort_merge_measure(context);
    auto& t = kamping::measurements::timer();
    // t.aggregate_and_print(
    //     kamping::measurements::SimpleJsonPrinter{ std::cout, {} });
    // std::cout << std::endl;
    // t.aggregate_and_print(kamping::measurements::FlatPrinter{});
    // std::cout << std::endl;
    // std::ofstream outFile(argv[1], std::ios::app);
    // t.aggregate_and_print(
    //     kamping::measurements::SimpleJsonPrinter{ outFile, {} });
    // std::cout << std::endl;
    // t.aggregate_and_print(kamping::measurements::FlatPrinter{});
    // std::cout << std::endl;
    // return;
    // alltoallMeasure(context);
    // ncclMeasure(context);
    // return 0;
#endif
    cudaMemcpyToSymbol(lookupNext, DCX::nextSample, sizeof(uint32_t) * DCX::X * DCX::X * 2, 0, cudaMemcpyHostToDevice);
    CUERR;
    SuffixSorter sorter(context, realLen, input);
    CUERR;
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<std::mt19937::result_type> randomDistChar(0, 255);
    std::uniform_int_distribution<std::mt19937::result_type> randomDistSize(0, UINT64_MAX);
    using T = size_t;

    // uint32_t randomDataSize = (1024 * 1024 * 1024);
    for (size_t round = 0; round < 5; round++)
    {
        size_t randomDataSize = 128 << round;

        // auto [text, data] = generate_data_dcx(randomDataSize, 1234 + round);
        // printf("[%lu] gen data\n", world_rank());
        // auto data_on_pe = comm_world().scatter(send_buf(data));
        // printf("[%lu] scatter\n", world_rank());
        // for (size_t i = 0; i < data_on_pe.size(); i++)
        // {
        //     printf("[%lu] data_on_pe[%lu]: %u\n", world_rank(), i, data_on_pe[i].index);
        // }
        thrust::host_vector<T> h_suffixes(randomDataSize);
        for (size_t i = 0; i < randomDataSize; i++)
        {
            h_suffixes[i] = randomDistSize(g);
        }

        thrust::device_vector<T> suffixes = h_suffixes;


        size_t out_size = 0;
        const int a = (int)(8 * log(NUM_GPUS) / log(2.));
        size_t bytes = sizeof(T) * randomDataSize;
        char sf[30];
        sprintf(sf, "sample_sort_%lu", bytes);
        thrust::device_vector<T> keys_out;

        t.synchronize_and_start(sf);
        sorter.SampleSort(suffixes, keys_out, a + 1, std::less<T>());
        context.sync_all_streams();
        t.stop_and_append();

        thrust::host_vector<T> keys_out_host = suffixes;
        std::vector<T> vec_key_out_host(keys_out_host.begin(), keys_out_host.end());
        if (!std::is_sorted(vec_key_out_host.begin(), vec_key_out_host.end())) {
            std::cerr << "GPU Samplesort does not sort input correctly locally" << std::endl;
        }
        ASSERT(keys_out_host.size() > 1);
        std::vector<T> keys_out_h(2);
        keys_out_h[0] = vec_key_out_host[0];
        keys_out_h[1] = vec_key_out_host.back();
        auto const out = comm_world().gather(send_buf(keys_out_h), root(0));
        context.sync_all_streams();
        comm_world().barrier();


        if (world_rank() == 0)
        {
            if (!std::is_sorted(out.begin(), out.end())) {
                std::cerr << "GPU Samplesort does not sort input correctly globally" << std::endl;
            }

            // const auto sorted_indices = naive_suffix_sort(randomDataSize, text);
            // bool const is_correct = std::equal(
            //     sorted_indices.begin(), sorted_indices.end(), out_keys_all.begin(),
            //     out_keys_all.end(), [](const auto& index, const auto& tuple) {
            //         return index == tuple.index;
            //     });
            // if (!is_correct) {
            //     std::cerr << "GPU Samplesort does not sort input correctly" << std::endl;
            //     // std::abort();
            // }
        }

        size_t gb = 1 << 30;
        size_t num_GB = bytes / gb;
        printf("[%lu] elements: %10u,  %5lu GB, time: %15.9f\n", world_rank(), randomDataSize, num_GB);

        // cudaFree(suffixes);
        // cudaFree(temp_storage);

    }
    // auto& t = kamping::measurements::timer();
    t.aggregate_and_print(
        kamping::measurements::SimpleJsonPrinter{ std::cout, {} });
    std::cout << std::endl;
    t.aggregate_and_print(kamping::measurements::FlatPrinter{});
    std::cout << std::endl;
    std::ofstream outFile2(argv[1], std::ios::app);
    t.aggregate_and_print(
        kamping::measurements::SimpleJsonPrinter{ outFile2, {} });
    std::cout << std::endl;
    t.aggregate_and_print(kamping::measurements::FlatPrinter{});
    std::cout << std::endl;
    return;
    sorter.alloc();
    // auto stringPath = ((std::string)argv[3]);
    // int pos = stringPath.find_last_of("/\\");
    // auto fileName = (pos == std::string::npos) ? argv[3] : stringPath.substr(pos + 1);

    // auto& t = kamping::measurements::timer();
    // t.synchronize_and_start(fileName);
    // nvtxRangePush("SuffixArray");
    sorter.do_sa();
    // nvtxRangePop();
    // t.stop();
    // if (world_rank() == 0)
    //     write_array(argv[2], sorter.get_result(), realLen);
    comm_world().barrier();
    sorter.done();

    if (world_rank() == 0)
    {
        sorter.print_pd_stats();
        sorter.get_perf_measurements().print(argv[1]);
    }

    cudaFreeHost(input);
    CUERR;
    // }
    // std::ofstream outFile(argv[1], std::ios::app);
    // t.aggregate_and_print(
    //     kamping::measurements::SimpleJsonPrinter{ outFile, {} });
    // std::cout << std::endl;
    // t.aggregate_and_print(kamping::measurements::FlatPrinter{});
    // std::cout << std::endl;
    return 0;
}
