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
#include "sorting/samplesort.cuh"

static const uint NUM_GPUS = 8;
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
__global__ void printArrayss(sa_index_t* isa, size_t size, size_t rank)
{
    for (size_t i = 0; i < size; i++) {
        printf("[%lu] [%4lu] %4u\n", rank, i, isa[i]);
    }
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

            printf("%3u, ", isa[i]);
        }
        else {
            printf("%3u", isa[i]);
        }

    }
    printf("\n");
    printf("[%lu]  sa: ", rank);
    for (size_t i = 0; i < size; i++) {
        if (i + 1 < size) {

            printf("%3u, ", sa_rank[i]);
        }
        else {
            printf("%3u", sa_rank[i]);
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
__global__ void printArrayss(kmerDCX* kmer, sa_index_t* isa, size_t size, size_t rank)
{
    for (size_t i = 0; i < size; i++) {

        printf("[%2lu] idx: %3u", rank, isa[i]);
        // unsigned char* kmerI = reinterpret_cast<*>(kmer[i]);
        for (int j = 0; j < DCX::X;j++) {
            printf(", %c", kmer[i].kmer[j]);
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
        printf(" r ");
        for (size_t x = 0; x < DCX::C; x++) {
            printf("%u, ", sk[i].ranks[x]);
        }
        printf("idx %u\n", sk[i].index);
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
        DCXArrays dcx_ptr;
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
    size_t last_gpu_extra_elements;
    size_t mtook_pd_iterations;
    thrust::host_vector<size_t> set_sizes;
    D_DCX* dcx;
public:
    SuffixSorter(Context& context, size_t len, char* input)
        : mcontext(context), mmemory_manager(context),
        mmulti_split(context), mall2all(context),
        mperf_measure(32),
        mpd_sorter(mcontext, mmemory_manager, mmulti_split, mall2all, mperf_measure),
        minput(input), minput_len(len)
    {
        cudaMalloc(&dcx, sizeof(D_DCX));
        cudaMemcpy(dcx->inverseSamplePosition, DCX::inverseSamplePosition, DCX::X * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dcx->nextNonSample, DCX::nextNonSample, DCX::nonSampleCount * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dcx->nextSample, DCX::nextSample, DCX::X * DCX::X * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dcx->samplePosition, DCX::samplePosition, DCX::C * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    void do_sa()
    {
        // TIMER_START_MAIN_STAGE(MainStages::Copy_Input);
        copy_input();

        //
        mcontext.sync_all_streams();
        // printf("[%lu] Copy Input\n", world_rank());
        comm_world().barrier();
        //

        TIMERSTART(Total);
        // TIMER_STOP_MAIN_STAGE(MainStages::Copy_Input);

        TIMER_START_MAIN_STAGE(MainStages::Produce_KMers);
        produce_kmers();
        //
        mcontext.sync_all_streams();
        // printf("[%lu] Produce kmers\n", world_rank());
        comm_world().barrier();
        //

        TIMER_STOP_MAIN_STAGE(MainStages::Produce_KMers);

        //            mpd_sorter.dump("After K-Mers");

        mtook_pd_iterations = mpd_sorter.sort(1);
        // comm_world().barrier();
        // auto& t = kamping::measurements::timer();
        // t.aggregate_and_print(
        //     kamping::measurements::SimpleJsonPrinter{ std::cout, {} });
        // std::cout << std::endl;
        // t.aggregate_and_print(kamping::measurements::FlatPrinter{});
        // std::cout << std::endl;
        //            mpd_sorter.dump("done");
        TIMER_START_MAIN_STAGE(MainStages::Prepare_S12_for_Merge);
        prepare_S12_for_merge();
        TIMER_STOP_MAIN_STAGE(MainStages::Prepare_S12_for_Merge);
        //
        // mcontext.sync_all_streams();
        // printf("[%lu] prepare s12 for merge done\n", world_rank());
        // comm_world().barrier();
        //
        TIMERSTOP(Total);
        mperf_measure.done();

        copy_result_to_host();

        return;

        TIMER_START_MAIN_STAGE(MainStages::Prepare_S0_for_Merge);

        //
        // mcontext.sync_all_streams();
        printf("[%lu] prepare s0 for merge done\n", world_rank());
        // comm_world().barrier();
        //
        TIMER_STOP_MAIN_STAGE(MainStages::Prepare_S0_for_Merge);
        TIMER_START_MAIN_STAGE(MainStages::Final_Merge);

        //
        // mcontext.sync_all_streams();
        printf("[%lu] final merge done\n", world_rank());
        // comm_world().barrier();
        //
        TIMER_STOP_MAIN_STAGE(MainStages::Final_Merge);
        // TIMER_START_MAIN_STAGE(MainStages::Copy_Results);

        //
        // mcontext.sync_all_streams();
        // printf("[%lu] complete\n", world_rank());
        // comm_world().barrier();
        //
        // TIMER_STOP_MAIN_STAGE(MainStages::Copy_Results);
    }

    auto get_result()
    {
        return mmemory_manager.get_h_result();
    }
    auto get_sa_length()
    {
        return mmemory_manager.get_sa_length();
    }
    SuffixArrayPerformanceMeasurements& get_perf_measurements()
    {
        return mperf_measure;
    }

    void done()
    {
        // mmemory_manager.free();
        mmemory_manager.free_Input_Isa();
    }

    void alloc()
    {
        // mper_gpu how much data for one gpu
        mper_gpu = SDIV(minput_len, NUM_GPUS);
        ASSERT_MSG(mper_gpu >= DCX::X, "Please give me more input.");

        // Ensure each gpu has a multiple of 3 because of triplets.
        mper_gpu = SDIV(mper_gpu, DCX::X) * DCX::X;
        printf("minput_len: %lu, mper_gpu %lu\n", minput_len, mper_gpu);
        ASSERT(minput_len > (NUM_GPUS - 1) * mper_gpu + DCX::X); // Because of merge
        size_t last_gpu_elems = minput_len - (NUM_GPUS - 1) * mper_gpu;
        ASSERT(last_gpu_elems <= mper_gpu); // Because of merge.

        mreserved_len = SDIV(std::max(last_gpu_elems, mper_gpu) + 8, DCX::X * 4) * DCX::X * 4; // Ensure there are 12 elems more space.
        mreserved_len = std::max(mreserved_len, 1024ul) + 100 * DCX::X;                       // Min len because of temp memory for CUB.

        mpd_reserved_len = SDIV(mreserved_len, DCX::X) * DCX::C;

        ms0_reserved_len = mreserved_len - mpd_reserved_len;

        mpd_per_gpu = (mper_gpu / DCX::X) * DCX::C;
        size_t last_gpu_add_pd_elements = 0;
        // if last_gpu_elems = 9 elements left and X=7 then we have 3+1 sample positions, last_gpu_elems = 10 3+2, last_gpu_elems = 11 3+2...
        if (last_gpu_elems % DCX::X != 0) {
            for (size_t sample = 0; sample < DCX::C; sample++)
            {
                if ((last_gpu_elems % DCX::X) > (size_t)DCX::samplePosition[sample]) {
                    last_gpu_add_pd_elements++;
                }
            }
        }
        // makes kmers easier because the set_sizes are now always set_sizes[0]=...=set_sizes[N-2]=set_sizes[N-1]+1=pd_elements/DCX::C
        last_gpu_extra_elements = DCX::C - last_gpu_add_pd_elements - 1;
        size_t last_gpu_pd_elements = (last_gpu_elems / DCX::X) * DCX::C + last_gpu_add_pd_elements + last_gpu_extra_elements;
        mpd_per_gpu_max_bit = sizeof(sa_index_t) * 8;//= std::min(sa_index_t(log2((NUM_GPUS - 1) * last_gpu_pd_elements)) + 1, sa_index_t(sizeof(sa_index_t) * 8));

        auto cub_temp_mem = get_needed_cub_temp_memory(ms0_reserved_len, mpd_reserved_len);
        cub::DoubleBuffer<kmer> keys(nullptr, nullptr);
        cub::DoubleBuffer<sa_index_t> values(nullptr, nullptr);
        cub::DoubleBuffer<sa_index_t> keys_sa(nullptr, nullptr);
        cub::DoubleBuffer<sa_index_t> values_sa(nullptr, nullptr);

        size_t temp_storage_size_S12 = 0;
        size_t temp_storage_size_dcx = 0;

        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size_S12,
            keys, values, mpd_reserved_len, DCXKmerDecomposer{}, 0, sizeof(kmer) * 8);
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size_dcx,
            keys_sa, values_sa, mpd_reserved_len, 0, mpd_per_gpu_max_bit);
        printf("[%lu] temp device_radix kmer: %lu, sa_radix temp: %lu\n", world_rank(), temp_storage_size_S12, temp_storage_size_dcx);
        temp_storage_size_S12 = std::max(temp_storage_size_dcx, temp_storage_size_S12);
        // Can do it this way since CUB temp memory is limited for large inputs.
        ms0_reserved_len = std::max(ms0_reserved_len, SDIV(cub_temp_mem.first, sizeof(MergeStageSuffix)));
        mpd_reserved_len = std::max(mpd_reserved_len, SDIV(cub_temp_mem.second, sizeof(MergeStageSuffix)));
        printf("mpd_reserved_len after cub temp: %lu\n", mpd_reserved_len);


        mmemory_manager.alloc(minput_len, mreserved_len, mpd_reserved_len, ms0_reserved_len, true, (mpd_per_gpu + 5 * DCX::X) * sizeof(kmerDCX), temp_storage_size_S12);

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

        mgpus.back().pd_elements = last_gpu_pd_elements;

        for (size_t i = 0; i < NUM_GPUS; i++)
        {
            printf("%lu bytes for kmer: %lu\n", i, mgpus[i].pd_elements * sizeof(kmerDCX));
        }

        set_sizes.resize(DCX::C);

        for (size_t i = 0; i < DCX::C; i++)
        {
            set_sizes[i] = 0;
            for (size_t gpu_idx = 0; gpu_idx < NUM_GPUS; gpu_idx++)
            {
                set_sizes[i] += mgpus[gpu_idx].pd_elements / DCX::C;
            }
            if ((mgpus.back().pd_elements % DCX::C) > i) {
                set_sizes[i]++;
            }
        }
        for (size_t i = 0; i < DCX::C; i++)
        {
            printf("S%lu: %lu\n", i, set_sizes[i]);
        }

        mgpus.back().offset = offset;
        mgpus.back().pd_offset = pd_offset;

        // Because of fixup.
        ASSERT(mgpus.back().pd_elements >= DCX::X);

        pd_total_len += mgpus.back().pd_elements;
        init_gpu_ptrs(NUM_GPUS - 1);

        printf("Every node gets %zu (%zu) elements, last node: %zu (%zu), reserved len: %zu.\n", mper_gpu,
            mpd_per_gpu, last_gpu_elems, mgpus.back().pd_elements, mreserved_len);
        printf("mpd_reserved_len: %lu\n", mpd_reserved_len);
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
        mgpus[i].dcx_ptr = mmemory_manager.get_dcx_arrays(i);

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
        // using kmer_t = uint64_t;
        // for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        //{
        auto gpu_index = world_rank();
        SaGPU& gpu = mgpus[gpu_index];

        // Need the halo to the right for kmers...
        size_t copy_len = std::min(gpu.num_elements + sizeof(kmer), minput_len - gpu.offset);

        cudaMemcpyAsync(gpu.pd_ptr.Input, minput, copy_len, cudaMemcpyHostToDevice,
            mcontext.get_gpu_default_stream(gpu_index));
        CUERR;
        // if (gpu_index == NUM_GPUS - 1)
        // {
        //     cudaMemsetAsync(gpu.pd_ptr.Input + gpu.num_elements, 0, 1,
        //         mcontext.get_gpu_default_stream(gpu_index));
        //     CUERR;
        // }
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
        // kernels::produce_index_kmer_tuples_12_64_dc7 _KLC_SIMPLE_(gpu.num_elements, mcontext.get_gpu_default_stream(gpu_index))((char*)gpu.pd_ptr.Input, gpu.pd_offset, gpu.pd_ptr.Isa, reinterpret_cast<ulong1*>(gpu.pd_ptr.Sa_rank),
        //     SDIV(gpu.num_elements, 14) * 14);
        thrust::device_vector<size_t> d_set_sizes = set_sizes;


        mcontext.sync_all_streams();
        kernels::produce_index_kmer_tuples_12_64_dcx _KLC_SIMPLE_(gpu.pd_elements, mcontext.get_gpu_default_stream(gpu_index))
            ((unsigned char*)gpu.pd_ptr.Input, gpu.pd_offset, gpu.pd_ptr.Isa, gpu.pd_ptr.Kmer,
                gpu.pd_elements, dcx->samplePosition, gpu_index, thrust::raw_pointer_cast(d_set_sizes.data()), mgpus[0].pd_elements / DCX::C, mreserved_len, mpd_reserved_len);
        CUERR;
        mcontext.sync_all_streams();
        if (world_rank() == NUM_GPUS - 1) {
            size_t fixups = last_gpu_extra_elements + DCX::C - 1;
            kernels::fixup_last_kmers << <1, 1 >> > (gpu.pd_ptr.Kmer + gpu.pd_elements - fixups, fixups);
        }
        // mcontext.sync_all_streams();
        // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (gpu.pd_ptr.Kmer, gpu.pd_ptr.Isa, std::min(20UL, gpu.pd_elements), world_rank());

        mcontext.sync_default_streams();
        comm_world().barrier();
    }

    void prepare_S12_for_merge()
    {
        std::array<MultiSplitNodeInfoT<sa_index_t, sa_index_t, sa_index_t>, NUM_GPUS> multi_split_node_info;
        std::array<All2AllNodeInfoT<sa_index_t, sa_index_t, sa_index_t>, NUM_GPUS> all2all_node_info;
        split_table_tt<sa_index_t, NUM_GPUS> split_table;
        std::array<sa_index_t, NUM_GPUS> dest_lens, src_lens;
        // printArrayss << <1, 1 >> > (mgpus[world_rank()].prepare_S12_ptr.Isa, mgpus[world_rank()].pd_elements, world_rank());
        mcontext.sync_all_streams();
        comm_world().barrier();
        // {
        //     std::vector<sa_index_t> isa(mgpus[world_rank()].pd_elements);

        //     cudaMemcpy(isa.data(), mgpus[world_rank()].dcx_ptr.Isa, sizeof(sa_index_t) * mgpus[world_rank()].pd_elements, cudaMemcpyDeviceToHost);
        //     std::vector<int> recv_counts_vec(NUM_GPUS);
        //     for (size_t i = 0; i < NUM_GPUS; i++)
        //     {
        //         recv_counts_vec[i] = mgpus[i].pd_elements;
        //     }

        //     auto isaglob = comm_world().gatherv(send_buf(isa), recv_counts(recv_counts_vec), root(0));
        //     if (world_rank() == 0) {
        //         std::sort(isaglob.begin(), isaglob.end());
        //         std::vector<sa_index_t> compareIsa(isaglob.size());
        //         for (size_t i = 0; i < compareIsa.size(); i++)
        //         {
        //             compareIsa[i] = i + 1;
        //         }
        //         size_t write_counter = 0;
        //         for (size_t i = 0; i < compareIsa.size(); i++)
        //         {
        //             if (isaglob[i] != compareIsa[i] && write_counter < 30) {

        //                 printf("[%lu] %u != %u\n", i, isaglob[i], compareIsa[i]);
        //                 write_counter++;
        //             }
        //         }
        //         bool ascend = std::equal(compareIsa.begin(), compareIsa.end(), isaglob.begin(), isaglob.end());
        //         bool containsDuplicates = (std::unique(isaglob.begin(), isaglob.end()) != isaglob.end());
        //         printf("isa before contains_dup: %s, is_ascending: %s\n", containsDuplicates ? "true" : "false", ascend ? "true" : "false");
        //         printf("isa before mpd_per_gpu: %lu\n", mpd_per_gpu);
        //     }
        // }

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Multisplit);
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            if (world_rank() == gpu_index) {

                kernels::write_indices_opt _KLC_SIMPLE_(gpu.pd_elements, mcontext.get_gpu_default_stream(gpu_index))(gpu.dcx_ptr.Temp1, gpu.pd_elements, set_sizes[0], mpd_per_gpu, gpu_index);
                CUERR;
                mcontext.get_device_temp_allocator(gpu_index).init(gpu.dcx_ptr.Temp4,
                    mpd_reserved_len * sizeof(sa_index_t));
            }
            multi_split_node_info[gpu_index].src_keys = gpu.dcx_ptr.Temp1;
            // s12_result == sa_index 
            multi_split_node_info[gpu_index].src_values = gpu.dcx_ptr.Isa;
            multi_split_node_info[gpu_index].src_len = gpu.pd_elements;

            multi_split_node_info[gpu_index].dest_keys = gpu.dcx_ptr.Temp2;
            multi_split_node_info[gpu_index].dest_values = gpu.dcx_ptr.Temp3;
            multi_split_node_info[gpu_index].dest_len = gpu.pd_elements;

        }

        PartitioningFunctor<sa_index_t> f(mpd_per_gpu, NUM_GPUS - 1);
        mcontext.sync_all_streams();
        comm_world().barrier();
        mmulti_split.execKVAsync(multi_split_node_info, split_table, src_lens, dest_lens, f);

        mcontext.sync_default_streams();
        comm_world().barrier();
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            //                fprintf(stderr,"GPU %u, src: %zu, dest: %zu.\n", gpu_index, src_lens[gpu_index], dest_lens[gpu_index]);
            all2all_node_info[gpu_index].src_keys = gpu.dcx_ptr.Temp2;
            all2all_node_info[gpu_index].src_values = gpu.dcx_ptr.Temp3;
            all2all_node_info[gpu_index].src_len = gpu.pd_elements;

            all2all_node_info[gpu_index].dest_keys = gpu.dcx_ptr.Temp1;
            all2all_node_info[gpu_index].dest_values = gpu.dcx_ptr.Isa;
            all2all_node_info[gpu_index].dest_len = gpu.pd_elements;
        }

        mall2all.execKVAsync(all2all_node_info, split_table);
        mcontext.sync_all_streams();
        comm_world().barrier();
        {
            uint gpu_index = world_rank();
            SaGPU& gpu = mgpus[world_rank()];
            cub::DoubleBuffer<sa_index_t> keys(gpu.dcx_ptr.Temp1, gpu.dcx_ptr.Temp2);
            cub::DoubleBuffer<sa_index_t> values(gpu.dcx_ptr.Isa, gpu.dcx_ptr.Temp3);
            size_t temp_storage_bytes = 0;
            cudaError_t err = cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                keys,
                values,
                gpu.pd_elements, 0, mpd_per_gpu_max_bit,
                mcontext.get_gpu_default_stream(gpu_index));

            ASSERT(temp_storage_bytes <= mmemory_manager.get_additional_dcx_space_size());
            // void* temp;
            // cudaMallocAsync(&temp, temp_storage_bytes, mcontext.get_gpu_default_stream(gpu_index));
            err = cub::DeviceRadixSort::SortPairs(gpu.dcx_ptr.Temp4, temp_storage_bytes,
                keys,
                values,
                gpu.pd_elements, 0, mpd_per_gpu_max_bit,
                mcontext.get_gpu_default_stream(gpu_index));

            // cudaFreeAsync(temp, mcontext.get_gpu_default_stream(gpu_index));

            mgpus.back().pd_elements -= last_gpu_extra_elements;



            kernels::write_indices_sub2 _KLC_SIMPLE_(gpu.pd_elements, mcontext.get_gpu_default_stream(gpu_index))(values.Current(), gpu.dcx_ptr.Isa, gpu.pd_elements, last_gpu_extra_elements);
            CUERR;


        }
        mcontext.sync_all_streams();

        mmemory_manager.free_();
        comm_world().barrier();
        {

            // std::vector<sa_index_t> isa_local(mgpus[world_rank()].pd_elements);
            // std::vector<sa_index_t> isa(mgpus[world_rank()].pd_elements);

            // cudaMemcpy(isa_local.data(), mgpus[world_rank()].dcx_ptr.Isa, sizeof(sa_index_t) * mgpus[world_rank()].pd_elements, cudaMemcpyDeviceToHost);
            // std::vector<int> recv_counts_vec(NUM_GPUS);
            // for (size_t i = 0; i < NUM_GPUS; i++)
            // {
            //     recv_counts_vec[i] = mgpus[i].pd_elements;
            // }

            // auto isaglob = comm_world().gatherv(send_buf(isa_local), recv_counts(recv_counts_vec), root(0));
            // printf("[%lu] all suffixes received\n", world_rank());
            // if (world_rank() == 0) {
            //     char fileName[16];
            //     const char* text = "outputIsa";
            //     sprintf(fileName, "%s", text);
            //     std::ofstream out(fileName, std::ios::binary);
            //     if (!out) {
            //         std::cerr << "Could not open file\n";
            //         //return 1;
            //     }
            //     printf("isa 12 length: %lu\n", isaglob.size());

            //     out.write(reinterpret_cast<char*>(isaglob.data()), sizeof(sa_index_t) * isaglob.size());
            //     out.close();
            // }

            // std::vector<char> input_all = comm_world().gatherv(send_buf(std::span<char>(minput, mper_gpu)), root(0));
            // printf("[%lu] all input received\n", world_rank());

            // if (world_rank() == 0) {
            //     // input_all.resize(minput_len);
            //     // for (size_t i = 0; i < input_all.size(); i++)
            //     // {
            //     //     printf("%c", input_all[i]);
            //     // }
            //     printf("\n");
            //     FILE* file = fopen("outputReal", "rb");

            //     if (!file) {
            //         perror("Could not open file");
            //         return;
            //     }

            //     fseek(file, 0, SEEK_END);

            //     size_t len = ftell(file);

            //     if (len == 0)
            //     {
            //         printf("File is empty!");
            //     }

            //     fseek(file, 0, SEEK_SET);
            //     size_t realLen = len / sizeof(uint32_t);
            //     std::vector<sa_index_t> sa(realLen);

            //     if (fread(sa.data(), sizeof(uint32_t), realLen, file) != realLen) {
            //         printf("Error");
            //     }
            //     fclose(file);


            //     // auto sa = naive_suffix_sort(minput_len, input_all.data());
            //     printf("sorted sa, size: %lu, minput_len: %lu\n", sa.size(), minput_len);
            //     thrust::host_vector<sa_index_t> sampleSa(sa.begin(), sa.end());
            //     thrust::host_vector<sa_index_t> inverter(sampleSa.size());
            //     for (size_t i = 0; i < inverter.size(); i++)
            //     {
            //         inverter[i] = i;
            //     }
            //     thrust::sort_by_key(sampleSa.begin(), sampleSa.end(), inverter.begin());
            //     size_t satotal = 0;
            //     for (size_t i = 0; i < sa.size(); i++)
            //     {
            //         for (size_t c = 0; c < DCX::C; c++)
            //         {
            //             if (i % DCX::X == DCX::samplePosition[c]) {
            //                 sampleSa[satotal++] = ((sa_index_t)inverter[i]);
            //                 // printf("[%2lu]%c, %u: ", i, minput[i], (sa_index_t)inverter[i]);
            //                 break;
            //             }
            //         }
            //     }
            //     sampleSa.resize(satotal);
            //     printf("sa total: %lu\n", satotal);
            //     thrust::host_vector<sa_index_t> inverter2(sampleSa.size());
            //     for (size_t i = 0; i < inverter2.size(); i++)
            //     {
            //         inverter2[i] = i;
            //     }
            //     thrust::sort_by_key(sampleSa.begin(), sampleSa.end(), inverter2.begin());
            //     for (size_t i = 0; i < sampleSa.size(); i++)
            //     {
            //         sampleSa[i] = i + 1;
            //     }
            //     thrust::sort_by_key(inverter2.begin(), inverter2.end(), sampleSa.begin());
            //     printf("\n");
            //     printf("created isa\n");
            //     // for (size_t i = 0; i < sampleSa.size(); i++)
            //     // {
            //     //     printf("isa2 %lu: %u\n", i, sampleSa[i]);
            //     // }

            //     // for (size_t i = 0; i < isaglob.size(); i++)
            //     // {
            //     //     printf("isa %lu: %u\n", i, isaglob[i]);
            //     // }

            //     size_t max_prints = 0;
            //     for (size_t i = 0; i < isaglob.size(); i++)
            //     {
            //         if (isaglob[i] != sampleSa[i] && max_prints < 10) {
            //             printf("isa[%lu] %u != %u real Isa\n", i, isaglob[i], sampleSa[i]);
            //         }
            //         if (isaglob[i] != sampleSa[i]) {
            //             max_prints++;
            //         }
            //     }
            //     printf("wrong: %lu\n", max_prints);
            //     if (std::equal(isaglob.begin(), isaglob.end(), sampleSa.begin(), sampleSa.end())) {
            //         printf("equal isa!\n");
            //     }

            //     std::sort(isaglob.begin(), isaglob.end());
            //     std::vector<sa_index_t> compareIsa(isaglob.size());
            //     for (size_t i = 0; i < compareIsa.size(); i++)
            //     {
            //         compareIsa[i] = i + 1;
            //     }
            //     size_t write_counter = 0;
            //     for (size_t i = 0; i < compareIsa.size(); i++)
            //     {
            //         if (isaglob[i] != compareIsa[i] && write_counter < 30) {

            //             printf("[%lu] %u != %u\n", i, isaglob[i], compareIsa[i]);
            //             write_counter++;
            //         }
            //     }

            //     bool ascend = std::equal(compareIsa.begin(), compareIsa.end(), isaglob.begin(), isaglob.end());
            //     bool containsDuplicates = (std::unique(isaglob.begin(), isaglob.end()) != isaglob.end());
            //     printf("contains_dup: %s, is_ascending: %s\n", containsDuplicates ? "true" : "false", ascend ? "true" : "false");
            //     printf("mpd_per_gpu: %lu\n", mpd_per_gpu);
            // }
        }
        comm_world().barrier();

        //
        // printf("[%lu] after write indices s12\n", world_rank());
        // mmulti_split.execKVAsync(multi_split_node_info, split_table, src_lens, dest_lens, f);

        mcontext.sync_default_streams();
        // comm_world().barrier();
        // printf("[%lu] after execKVAsync s12\n", world_rank());
        // printArrayss << <1, 1 >> > ((sa_index_t*)mgpus[world_rank()].prepare_S12_ptr.S12_buffer2, (sa_index_t*)mgpus[world_rank()].prepare_S12_ptr.S12_result_half, mgpus[world_rank()].pd_elements, world_rank());

        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Multisplit);

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Out);
        size_t count = mgpus[world_rank()].num_elements - mgpus[world_rank()].pd_elements;

        thrust::device_vector<MergeSuffixes> merge_tuple_vec(mgpus[world_rank()].num_elements);

        // cudaMalloc(&merge_tuple, sizeof(MergeSuffixes) * mgpus[world_rank()].num_elements);
        // CUERR;
        // MergeSuffixes* merge_tuple_out;

        uint gpu_index = world_rank();
        SaGPU& gpu = mgpus[gpu_index];

        unsigned char* next_Input = nullptr;
        sa_index_t* next_Isa = nullptr;      //= (gpu_index + 1 < NUM_GPUS) ? mgpus[gpu_index + 1].prepare_S12_ptr.Isa : nullptr;
        // printf("[%lu] before sending\n", world_rank());
        if (mcontext.is_in_node()) {
            next_Isa = (gpu_index + 1 < NUM_GPUS) ? mgpus[gpu_index + 1].dcx_ptr.Isa : nullptr;
        }
        else {
            ncclGroupStart();
            if (gpu_index > 0)
            {
                NCCLCHECK(ncclSend(gpu.dcx_ptr.Isa, DCX::X, ncclUint32, gpu_index - 1, mcontext.get_nccl(), mcontext.get_streams(gpu_index)[gpu_index - 1]));
                NCCLCHECK(ncclSend(gpu.dcx_ptr.Input, DCX::X, ncclChar, gpu_index - 1, mcontext.get_nccl(), mcontext.get_streams(gpu_index)[gpu_index - 1]));
            }
            if (gpu_index + 1 < NUM_GPUS)
            {
                next_Isa = gpu.dcx_ptr.Isa + gpu.pd_elements;
                NCCLCHECK(ncclRecv(next_Isa, DCX::X, ncclUint32, gpu_index + 1, mcontext.get_nccl(), mcontext.get_gpu_default_stream(gpu_index)));

                next_Input = gpu.dcx_ptr.Input + gpu.num_elements;
                NCCLCHECK(ncclRecv(next_Input, DCX::X, ncclChar, gpu_index + 1, mcontext.get_nccl(), mcontext.get_gpu_default_stream(gpu_index)));
            }
            ncclGroupEnd();
        }
        mcontext.sync_all_streams();
        comm_world().barrier();
        // printf("[%lu] after sending\n", world_rank());

        const unsigned char* c_next_Input = mcontext.is_in_node() ? ((gpu_index + 1 < NUM_GPUS) ? mgpus[gpu_index + 1].dcx_ptr.Input : nullptr) : next_Input;


        kernels::prepare_SK_ind_kv _KLC_SIMPLE_(gpu.pd_elements, mcontext.get_gpu_default_stream(gpu_index))
            (gpu.dcx_ptr.Isa, gpu.dcx_ptr.Input,
                next_Isa, c_next_Input, gpu.offset, gpu.num_elements,
                thrust::raw_pointer_cast(merge_tuple_vec.data()), gpu.pd_elements, dcx);
        CUERR;
        // printArrayss << <1, 1 >> > (merge_tuple, mgpus[world_rank()].pd_elements, world_rank());
        mcontext.sync_all_streams();
        comm_world().barrier();
        // printf("[%lu] non samples-------------------------------------------\n", world_rank());

        size_t noSampleCount = 0;
        for (uint32_t i = 0; i < DCX::nonSampleCount; i++) {

            size_t count2 = (count / DCX::nonSampleCount);
            if (i < count % DCX::nonSampleCount) {
                count2++;
            }

            kernels::prepare_non_sample
                _KLC_SIMPLE_(count2, mcontext.get_gpu_default_stream(gpu_index))
                // << <1, 1, 0, mcontext.get_gpu_default_stream(gpu_index) >> >

                (gpu.dcx_ptr.Isa, gpu.dcx_ptr.Input, next_Isa, c_next_Input, gpu.offset, gpu.num_elements,
                    gpu.pd_elements,
                    thrust::raw_pointer_cast(merge_tuple_vec.data()) + gpu.pd_elements + noSampleCount, count2, DCX::nextNonSample[i], DCX::inverseSamplePosition[i]);
            CUERR;
            noSampleCount += count2;
        }
        // mcontext.sync_default_streams();
        // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(gpu_index) >> > (nonSamples, count, gpu_index);
        mcontext.sync_default_streams();
        comm_world().barrier();
        // printf("[%lu] after non samples\n", world_rank());
        cudaFree(dcx);
        CUERR;
        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Out);

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_All2All);
        // thrust::host_vector<MergeSuffixes> tuples_host = merge_tuple_vec;

        // std::vector<MergeSuffixes> host_vec(tuples_host.begin(), tuples_host.end());
        // printf("[%lu] all vec send\n", world_rank());
        // size_t all_num_elements = 0;
        // for (size_t gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)
        // {
        //     all_num_elements += mgpus[gpu_index].num_elements;
        // }

        // std::vector<MergeSuffixes> all_vec(all_num_elements);
        // comm_world().gatherv(send_buf(host_vec), recv_buf(all_vec), root(0));
        // printf("[%lu] all vec recv\n", world_rank());

        // std::vector<sa_index_t> h_sa(all_vec.size());
        // if (world_rank() == 0)
        // {
        //     {
        //         char fileName[16];
        //         const char* text = "outputTuples";
        //         sprintf(fileName, "%s", text);
        //         std::ofstream out(fileName, std::ios::binary);
        //         if (!out) {
        //             std::cerr << "Could not open file\n";
        //             //return 1;
        //         }
        //         printf("tuples 12 length: %lu\n", all_vec.size());

        //         out.write(reinterpret_cast<char*>(all_vec.data()), sizeof(MergeSuffixes) * all_vec.size());
        //         out.close();
        //     }

        // std::sort(all_vec.begin(), all_vec.end(), DCXComparatorHost{});
        // printf("[%lu] sorted\n", world_rank());

        // for (size_t i = 0; i < h_sa.size(); i++)
        // {
        //     h_sa[i] = all_vec[i].index;
        // }
        // {
        //     char fileName[16];
        //     const char* text = "outputSaHost";
        //     sprintf(fileName, "%s", text);
        //     std::ofstream out(fileName, std::ios::binary);
        //     if (!out) {
        //         std::cerr << "Could not open file\n";
        //     }
        //     printf("isa 12 length: %lu\n", h_sa.size());

        //     out.write(reinterpret_cast<char*>(h_sa.data()), sizeof(sa_index_t) * h_sa.size());
        //     out.close();
        // }

        // }
        // comm_world().barrier();
        SampleSort<MergeSuffixes, DCXComparatorDevice, NUM_GPUS>(merge_tuple_vec, std::min(size_t(32ULL * log(NUM_GPUS) / log(2.)), mgpus[NUM_GPUS - 1].num_elements / 2), DCXComparatorDevice{}, mcontext, mperf_measure);
        SegmentedSort<NUM_GPUS>(merge_tuple_vec, mcontext, mperf_measure);
        {
            // bool locally_sorted = thrust::is_sorted(merge_tuple_out_vec.begin(), merge_tuple_out_vec.end(), DCXComparatorDevice{});
            // printf("[%lu] is locally sorted: %s\n", world_rank(), locally_sorted ? "true" : "false");
            // thrust::host_vector<MergeSuffixes> locally_sorted_tuples = merge_tuple_out_vec;
            // std::vector<MergeSuffixes> locally_sorted_tuples_vec(locally_sorted_tuples.begin(), locally_sorted_tuples.end());

            // printArrayss << <1, 1 >> > (thrust::raw_pointer_cast(merge_tuple_out_vec.data()), 10, world_rank());
            // mcontext.sync_all_streams();
            // comm_world().barrier();
            // printArrayss << <1, 1 >> > (thrust::raw_pointer_cast(merge_tuple_out_vec.data()) + merge_tuple_out_vec.size() - 10, 10, world_rank() + NUM_GPUS);
            // mcontext.sync_all_streams();
            // comm_world().barrier();
            // std::vector<MergeSuffixes> globally_sorted_tuples(all_num_elements);
            // comm_world().gatherv(send_buf(locally_sorted_tuples_vec), recv_buf(globally_sorted_tuples), root(0));
            // if (world_rank() == 0) {
            //     bool globally_sorted = std::is_sorted(globally_sorted_tuples.begin(), globally_sorted_tuples.end(), DCXComparatorHost{});
            //     printf("[%lu] is globally sorted: %s\n", world_rank(), globally_sorted ? "true" : "false");

            //     bool sorted_equal_host = std::equal(globally_sorted_tuples.begin(), globally_sorted_tuples.end(), all_vec.begin(), all_vec.end());

            //     printf("[%lu] cpu sorted tuples equal: %s\n", world_rank(), sorted_equal_host ? "true" : "false");

            //     char fileName[30];
            //     const char* text = "outputSortedTuples";
            //     sprintf(fileName, "%s", text);
            //     std::ofstream out(fileName, std::ios::binary);
            //     if (!out) {
            //         std::cerr << "Could not open file\n";
            //         //return 1;
            //     }
            //     printf("tuples 12 length: %lu\n", globally_sorted_tuples.size());

            //     out.write(reinterpret_cast<char*>(globally_sorted_tuples.data()), sizeof(MergeSuffixes) * globally_sorted_tuples.size());
            //     out.close();

            // }

        }
        // MultiMerge<MergeSuffixes, DCXComparatorDevice, DCXComparatorHost, NUM_GPUS>(merge_tuple_vec, merge_tuple_out_vec, DCXComparatorDevice{}, DCXComparatorHost{}, mcontext);
        // merge_tuple_out_vec.swap(merge_tuple_vec);

        mcontext.sync_all_streams();

        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_All2All);
        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Into_Place);
        // printf("[%lu] sample sorted\n", world_rank());
        size_t out_num_elements = merge_tuple_vec.size();

        // printf("[%lu] num elements: %lu\n", world_rank(), out_num_elements);

        thrust::device_vector<sa_index_t> d_sa(out_num_elements);
        kernels::write_sa _KLC_SIMPLE_(out_num_elements, mcontext.get_gpu_default_stream(gpu_index))(thrust::raw_pointer_cast(merge_tuple_vec.data()), thrust::raw_pointer_cast(d_sa.data()), out_num_elements);
        mcontext.sync_all_streams();
        mmemory_manager.set_sa_length(out_num_elements);
        mmemory_manager.get_result_vec().swap(d_sa);

        {

            // std::vector<sa_index_t> all_sa(all_num_elements);
            // comm_world().gatherv(send_buf(sa), recv_buf(all_sa), root(0));
            // if (world_rank() == 0) {
            //     bool com = std::equal(h_sa.begin(), h_sa.end(), all_sa.begin(), all_sa.end());
            //     printf("CPU sort equal to Samplesort: %s\n", com ? "true" : "false");
            //     std::vector<sa_index_t> compareSa(all_sa.size());
            //     for (size_t i = 0; i < compareSa.size(); i++)
            //     {
            //         compareSa[i] = i;
            //     }
            //     size_t write_counter = 0;
            //     for (size_t i = 0; i < compareSa.size(); i++)
            //     {
            //         if (all_sa[i] != compareSa[i] && write_counter < 30) {

            //             printf("[%lu] %u != %u\n", i, all_sa[i], compareSa[i]);
            //             write_counter++;
            //         }
            //     }

            //     bool ascend = std::equal(compareSa.begin(), compareSa.end(), all_sa.begin(), all_sa.end());
            //     bool containsDuplicates = (std::unique(all_sa.begin(), all_sa.end()) != all_sa.end());
            //     printf("SA contains_dup: %s, is_ascending: %s\n", containsDuplicates ? "true" : "false", ascend ? "true" : "false");
            // }
        }

        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Into_Place);
    }



    void copy_result_to_host()
    {
        sa_index_t* d_sa = mmemory_manager.get_sa();
        size_t out_num_elements = mmemory_manager.get_result_vec().size();
        sa_index_t* sa = (sa_index_t*)malloc(sizeof(sa_index_t) * out_num_elements);

        cudaMemcpyAsync(sa, d_sa, out_num_elements * sizeof(sa_index_t), cudaMemcpyDeviceToHost, mcontext.get_gpu_default_stream(world_rank()));
        mmemory_manager.set_result_ptr(sa);
        mcontext.sync_all_streams();
        comm_world().barrier();

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
        merge_manager.merge(ranges, std::less<uint64_t>(), std::less<uint64_t>());
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

    if (argc != 4)
    {
        error("Usage: sa-test <ofile> <measfile> <ifile> !");
    }

    // for (int i = 0; i < 2; i++)
    // {

    comm_world().barrier();
    char* input = nullptr;

    size_t realLen = 0;
    size_t maxLength = size_t(1024 * 1024) * size_t(150 * NUM_GPUS);
    size_t inputLen = read_file_into_host_memory(&input, argv[3], realLen, sizeof(sa_index_t), maxLength, NUM_GPUS, 0);
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
    cudaMemcpyToSymbol(lookupNext, DCX::nextSample, sizeof(uint32_t) * DCX::X * DCX::X * 3, 0, cudaMemcpyHostToDevice);
    CUERR;
    SuffixSorter sorter(context, realLen, input);
    CUERR;
    // std::random_device rd;
    // std::mt19937 g(rd());
    // std::uniform_int_distribution<std::mt19937::result_type> randomDistChar(0, 255);
    // std::uniform_int_distribution<std::mt19937::result_type> randomDistSize(0, UINT64_MAX);
    // using T = MergeSuffixes;

    // for (size_t round = 0; round < 18; round++)
    // {
    //     size_t randomDataSize = 512 << round;
    //     // std::tuple<std::string, std::vector<T>> ;
    //     auto [text, data] = generate_data_dcx(randomDataSize, 1234 + round);
    //     comm_world().barrier();
    //     printf("[%lu] gen data\n", world_rank());
    //     auto data_on_pe = comm_world().scatter(send_buf(data), root(0));
    //     printf("[%lu] scatter\n", world_rank());
    //     // for (size_t i = 0; i < data_on_pe.size(); i++)
    //     // {
    //     //     printf("[%lu] data_on_pe[%lu]: %u\n", world_rank(), i, data_on_pe[i].index);
    //     // }
    //     thrust::host_vector<T> h_suffixes(data_on_pe.begin(), data_on_pe.end());
    //     // for (size_t i = 0; i < randomDataSize; i++)
    //     // {
    //     //     h_suffixes[i] = randomDistSize(g);
    //     // }

    //     thrust::device_vector<T> suffixes = h_suffixes;


    //     size_t out_size = 0;
    //     const int a = (int)(16 * log(NUM_GPUS) / log(2.));
    //     size_t bytes = sizeof(T) * randomDataSize;
    //     char sf[30];
    //     sprintf(sf, "sample_sort_%lu", bytes);
    //     thrust::device_vector<T> keys_out;

    //     t.synchronize_and_start(sf);
    //     SampleSort<T, DCXComparatorDevice, NUM_GPUS>(suffixes, a + 1, DCXComparatorDevice{}, context);
    //     context.sync_all_streams();
    //     comm_world().barrier();

    //     t.stop_and_append();

    //     thrust::host_vector<T> keys_out_host = keys_out;
    //     std::vector<T> vec_key_out_host(keys_out_host.begin(), keys_out_host.end());

    //     // if (!std::is_sorted(vec_key_out_host.begin(), vec_key_out_host.end())) {
    //     //     std::cerr << "GPU Samplesort does not sort input correctly locally" << std::endl;
    //     // }
    //     ASSERT(keys_out_host.size() > 1);
    //     // std::vector<T> keys_out_h(2);
    //     // keys_out_h[0] = vec_key_out_host[0];
    //     // keys_out_h[1] = vec_key_out_host.back();
    //     auto const out = comm_world().gatherv(send_buf(vec_key_out_host), root(0));
    //     context.sync_all_streams();
    //     comm_world().barrier();


    //     if (world_rank() == 0)
    //     {
    //         std::vector<size_t> sa = naive_suffix_sort(randomDataSize, text);
    //         bool const is_correct = std::equal(
    //             sa.begin(), sa.end(), out.begin(),
    //             out.end(), [](const auto& index, const auto& tuple) {
    //                 return index == tuple.index;
    //             });
    //         // if (!std::is_sorted(out.begin(), out.end())) {
    //         if (!is_correct) {
    //             std::cerr << "GPU Samplesort does not sort input correctly globally" << std::endl;
    //         }
    //     }

    //     size_t gb = 1 << 30;
    //     size_t num_GB = bytes / gb;
    //     printf("[%lu] elements: %10u,  %5lu GB, time: %15.9f\n", world_rank(), randomDataSize, num_GB);

    //     // cudaFree(suffixes);
    //     // cudaFree(temp_storage);

    // }
    // // auto& t = kamping::measurements::timer();
    // t.aggregate_and_print(
    //     kamping::measurements::SimpleJsonPrinter{ std::cout, {} });
    // std::cout << std::endl;
    // t.aggregate_and_print(kamping::measurements::FlatPrinter{});
    // std::cout << std::endl;
    // std::ofstream outFile2(argv[1], std::ios::app);
    // t.aggregate_and_print(
    //     kamping::measurements::SimpleJsonPrinter{ outFile2, {} });
    // std::cout << std::endl;
    // t.aggregate_and_print(kamping::measurements::FlatPrinter{});
    // std::cout << std::endl;
    // return;
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
    write_array_mpi(argv[1], sorter.get_result(), sorter.get_sa_length());
    // comm_world().barrier();
    sorter.done();

    CUERR;
    if (world_rank() == 0)
    {
        sorter.print_pd_stats();
        sorter.get_perf_measurements().print(argv[2]);
    }
    CUERR;

    cudaFreeHost(input);
    CUERR;
    // }
    // std::ofstream outFile("outputSampleSort", std::ios::app);
    // t.aggregate_and_print(
    //     kamping::measurements::SimpleJsonPrinter{ outFile, {} });
    // std::cout << std::endl;
    // t.aggregate_and_print(kamping::measurements::FlatPrinter{});
    // std::cout << std::endl;
    return 0;
}
