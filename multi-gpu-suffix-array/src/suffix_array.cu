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
#include <kamping/checking_casts.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/data_buffer.hpp>
#include <kamping/environment.hpp>
#include <kamping/measurements/printer.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/named_parameters.hpp>
#include <kamping/communicator.hpp>
#include "distrib_merge/distrib_merge.hpp"
#include <chrono>
// #include <nvToolsExt.h>
#include "moderngpu/kernel_mergesort.hxx"
static const uint NUM_GPUS = 4;

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
static_assert(NUM_GPUS <= 4, "At the moment, there is no node with more than 4 all-connected nodes. This is likely a configuration error.");

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
__global__ void printArrayss(MergeStageSuffixS12HalfValue* isa, MergeStageSuffixS12HalfKey* sa_rank, size_t size, size_t rank)
{
    printf("[%lu] key: ", rank);
    for (size_t i = 0; i < size; i++) {
        printf("%c, %c, %c, %u\n", isa[i].chars[0], isa[i].chars[1], isa[i].rank_p1p2);
    }
    printf("\n");
    printf("[%lu]  value: ", rank);
    for (size_t i = 0; i < size; i++) {

        printf("%u, %u\n", sa_rank[i].own_rank, sa_rank[i].index);
    }
    printf("\n");
    printf("---------------------------------------------------------------------------\n");
}
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
            assert(b.index % 3 != 0);
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
            assert(b.index % 3 == 0);
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

#include "prefix_doubling.hpp"

#define TIMER_START_PREPARE_FINAL_MERGE_STAGE(stage) mperf_measure.start_prepare_final_merge_stage(stage)
#define TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(stage) mperf_measure.stop_prepare_final_merge_stage(stage)

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

    void do_sa()
    {

        // TIMER_START_MAIN_STAGE(MainStages::Copy_Input);
        copy_input();
        // TIMER_STOP_MAIN_STAGE(MainStages::Copy_Input);

        TIMERSTART(Total);
        TIMER_START_MAIN_STAGE(MainStages::Produce_KMers);
        produce_kmers();
        TIMER_STOP_MAIN_STAGE(MainStages::Produce_KMers);
        //            mpd_sorter.dump("After K-Mers");

        mtook_pd_iterations = mpd_sorter.sort(4);

        //            mpd_sorter.dump("done");
        TIMER_START_MAIN_STAGE(MainStages::Prepare_S12_for_Merge);
        prepare_S12_for_merge();

        TIMER_STOP_MAIN_STAGE(MainStages::Prepare_S12_for_Merge);
        TIMER_START_MAIN_STAGE(MainStages::Prepare_S0_for_Merge);
        prepare_S0_for_merge();

        TIMER_STOP_MAIN_STAGE(MainStages::Prepare_S0_for_Merge);
        TIMER_START_MAIN_STAGE(MainStages::Final_Merge);

        final_merge();
        TIMER_STOP_MAIN_STAGE(MainStages::Final_Merge);
        TIMERSTOP(Total);
        mperf_measure.done();

        TIMER_START_MAIN_STAGE(MainStages::Copy_Results);
        copy_result_to_host();
        TIMER_STOP_MAIN_STAGE(MainStages::Copy_Results);

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
        ASSERT_MSG(mper_gpu >= 3, "Please give me more input.");

        // Ensure each gpu has a multiple of 3 because of triplets.
        mper_gpu = SDIV(mper_gpu, 3) * 3;

        ASSERT(minput_len > (NUM_GPUS - 1) * mper_gpu + 3); // Because of merge
        size_t last_gpu_elems = minput_len - (NUM_GPUS - 1) * mper_gpu;
        ASSERT(last_gpu_elems <= mper_gpu); // Because of merge.

        mreserved_len = SDIV(std::max(last_gpu_elems, mper_gpu) + 8, 12) * 12; // Ensure there are 12 elems more space.
        mreserved_len = std::max(mreserved_len, 1024ul);                       // Min len because of temp memory for CUB.

        mpd_reserved_len = SDIV(mreserved_len, 3) * 2;

        ms0_reserved_len = mreserved_len - mpd_reserved_len;

        auto cub_temp_mem = get_needed_cub_temp_memory(ms0_reserved_len, mpd_reserved_len);

        // Can do it this way since CUB temp memory is limited for large inputs.
        ms0_reserved_len = std::max(ms0_reserved_len, SDIV(cub_temp_mem.first, sizeof(MergeStageSuffix)));
        mpd_reserved_len = std::max(mpd_reserved_len, SDIV(cub_temp_mem.second, sizeof(MergeStageSuffix)));

        mmemory_manager.alloc(minput_len, mreserved_len, mpd_reserved_len, ms0_reserved_len, true);

        mpd_per_gpu = mper_gpu / 3 * 2;
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
        // FIXME: Isn't this just...: last_gpu_elems / 3 * 2 + ((last_gpu_elems % 3) == 2);
        mgpus.back().pd_elements = last_gpu_elems / 3 * 2 + (((last_gpu_elems % 3) != 0) ? ((last_gpu_elems - 1) % 3) : 0);
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
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];

            // Need the halo to the right for kmers...
            size_t copy_len = std::min(gpu.num_elements + sizeof(kmer_t), minput_len - gpu.offset);

            cudaSetDevice(mcontext.get_device_id(gpu_index));
            cudaMemcpyAsync(gpu.pd_ptr.Input, minput + gpu.offset, copy_len, cudaMemcpyHostToDevice,
                mcontext.get_gpu_default_stream(gpu_index));
            CUERR;
            if (gpu_index + 1 == NUM_GPUS)
            {
                cudaMemsetAsync(gpu.pd_ptr.Input + gpu.num_elements, 0, sizeof(kmer_t),
                    mcontext.get_gpu_default_stream(gpu_index));
                CUERR;
            }
        }

        mcontext.sync_default_streams();
    }

    void produce_kmers()
    {
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            cudaSetDevice(mcontext.get_device_id(gpu_index));
            //                kernels::produce_index_kmer_tuples _KLC_SIMPLE_(gpu.num_elements, mcontext.get_gpu_default_stream(gpu_index))
            //                        ((char*)gpu.input, offset, gpu.pd_index, gpu.pd_kmers, gpu.num_elements); CUERR;
            kernels::produce_index_kmer_tuples_12_64 _KLC_SIMPLE_(gpu.num_elements, mcontext.get_gpu_default_stream(gpu_index))((char*)gpu.pd_ptr.Input, gpu.pd_offset, gpu.pd_ptr.Isa, reinterpret_cast<ulong1*>(gpu.pd_ptr.Sa_rank),
                SDIV(gpu.num_elements, 12) * 12);
            CUERR;
        }
        kernels::fixup_last_four_12_kmers_64 << <1, 4, 0, mcontext.get_gpu_default_stream(NUM_GPUS - 1) >> > (reinterpret_cast<ulong1*>(mgpus.back().pd_ptr.Sa_rank) + mgpus.back().pd_elements - 4);
        mcontext.sync_default_streams();
        // for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        // {
        //     SaGPU& gpu = mgpus[gpu_index];
        //     printf("elements: %lu, offset %lu\n", gpu.num_elements, gpu.offset);
        //     printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(gpu_index) >> > (reinterpret_cast<char*>(gpu.pd_ptr.Sa_rank), gpu.pd_ptr.Isa, SDIV(gpu.num_elements, 12) * 12, gpu_index);
        //     mcontext.sync_default_streams();
        // }
    }

    void prepare_S12_for_merge()
    {
        std::array<MultiSplitNodeInfoT<sa_index_t, sa_index_t, sa_index_t>, NUM_GPUS> multi_split_node_info;
        std::array<All2AllNodeInfoT<MergeStageSuffixS12HalfKey, MergeStageSuffixS12HalfValue, sa_index_t>, NUM_GPUS> all2all_node_info;
        split_table_tt<sa_index_t, NUM_GPUS> split_table;
        std::array<sa_index_t, NUM_GPUS> dest_lens, src_lens;
        // {
        //     mcontext.sync_all_streams();
        //     size_t workinLen = 0;
        //     for (size_t gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)
        //     {
        //         SaGPU& gpu = mgpus[gpu_index];
        //         workinLen += gpu.pd_elements;
        //     }
        //     std::vector<sa_index_t> k(workinLen);
        //     size_t prefix_sum = 0;
        //     for (size_t gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)
        //     {
        //         SaGPU& gpu = mgpus[gpu_index];
        //         cudaMemcpy(k.data() + prefix_sum, gpu.prepare_S12_ptr.Isa, sizeof(sa_index_t) * gpu.pd_elements, cudaMemcpyDeviceToHost);
        //         prefix_sum += gpu.pd_elements;
        //     }

        //     char fileName[18];
        //     const char* text = "ISA";
        //     sprintf(fileName, "%s", text);
        //     std::ofstream out(fileName, std::ios::binary);
        //     if (!out) {
        //         std::cerr << "Could not open file\n";
        //         //return 1;
        //     }
        //     printf("sa isa length: %lu\n", k.size());

        //     out.write(reinterpret_cast<char*>(k.data()), sizeof(sa_index_t) * k.size());
        //     out.close();

        // }
        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Multisplit);
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            cudaSetDevice(mcontext.get_device_id(gpu_index));
            kernels::write_indices _KLC_SIMPLE_(gpu.pd_elements, mcontext.get_gpu_default_stream(gpu_index))((sa_index_t*)gpu.prepare_S12_ptr.S12_result, gpu.pd_elements);
            CUERR;
            // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(gpu_index) >> > (gpu.prepare_S12_ptr.Isa, (sa_index_t*)gpu.prepare_S12_ptr.S12_result, gpu.pd_elements, (size_t)gpu_index);
            mcontext.sync_default_streams();
            multi_split_node_info[gpu_index].src_keys = gpu.prepare_S12_ptr.Isa;
            multi_split_node_info[gpu_index].src_values = (sa_index_t*)gpu.prepare_S12_ptr.S12_result;
            multi_split_node_info[gpu_index].src_len = gpu.pd_elements;

            multi_split_node_info[gpu_index].dest_keys = (sa_index_t*)gpu.prepare_S12_ptr.S12_buffer2;
            multi_split_node_info[gpu_index].dest_values = (sa_index_t*)gpu.prepare_S12_ptr.S12_result_half;
            multi_split_node_info[gpu_index].dest_len = gpu.pd_elements;
            ;
            mcontext.get_device_temp_allocator(gpu_index).init(gpu.prepare_S12_ptr.S12_buffer1,
                mpd_reserved_len * sizeof(MergeStageSuffixS12));
        }
        S12PartitioningFunctor f(mpd_per_gpu, NUM_GPUS - 1);

        mmulti_split.execKVAsync(multi_split_node_info, split_table, src_lens, dest_lens, f);

        mcontext.sync_default_streams();
        // printf("multi\n");
        // for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        // {
        //     SaGPU& gpu = mgpus[gpu_index];
        //     printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(gpu_index) >> > ((sa_index_t*)gpu.prepare_S12_ptr.S12_buffer2, (sa_index_t*)gpu.prepare_S12_ptr.S12_result_half, gpu.pd_elements, (size_t)gpu_index);
        //     mcontext.sync_default_streams();
        // }
        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Multisplit);

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Out);

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            cudaSetDevice(mcontext.get_device_id(gpu_index));

            const sa_index_t* next_Isa = (gpu_index + 1 < NUM_GPUS) ? mgpus[gpu_index + 1].prepare_S12_ptr.Isa : nullptr;
            const unsigned char* next_Input = (gpu_index + 1 < NUM_GPUS) ? mgpus[gpu_index + 1].prepare_S12_ptr.Input : nullptr;

            kernels::prepare_S12_ind_kv _KLC_SIMPLE_(gpu.pd_elements, mcontext.get_gpu_default_stream(gpu_index))((sa_index_t*)gpu.prepare_S12_ptr.S12_result_half,
                gpu.prepare_S12_ptr.Isa, gpu.prepare_S12_ptr.Input,
                next_Isa, next_Input, gpu.offset, gpu.num_elements,
                mpd_per_gpu,
                gpu.prepare_S12_ptr.S12_buffer1, gpu.prepare_S12_ptr.S12_buffer1_half, gpu.pd_elements);
            CUERR;
        }

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            //                printf("GPU %u, sr    c: %u, dest: %u.\n", gpu_index, src_lens[gpu_index], dest_lens[gpu_index]);
            all2all_node_info[gpu_index].src_keys = gpu.prepare_S12_ptr.S12_buffer1;
            all2all_node_info[gpu_index].src_values = gpu.prepare_S12_ptr.S12_buffer1_half;
            all2all_node_info[gpu_index].src_len = gpu.pd_elements;

            all2all_node_info[gpu_index].dest_keys = gpu.prepare_S12_ptr.S12_buffer2;
            all2all_node_info[gpu_index].dest_values = gpu.prepare_S12_ptr.S12_buffer2_half;
            all2all_node_info[gpu_index].dest_len = gpu.pd_elements;

            all2all_node_info[gpu_index].temp_keys = reinterpret_cast<MergeStageSuffixS12HalfKey*>(gpu.prepare_S12_ptr.S12_result);
            all2all_node_info[gpu_index].temp_values = gpu.prepare_S12_ptr.S12_result_half;
            all2all_node_info[gpu_index].temp_len = mpd_reserved_len; // not sure...
        }
        mcontext.sync_default_streams();
        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Out);
        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_All2All);

        //            dump_prepare_s12("After split");

        // printf("alltoall\n");
        mall2all.execKVAsync(all2all_node_info, split_table);
        mcontext.sync_all_streams();
        // for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        // {
        //     SaGPU& gpu = mgpus[gpu_index];

        //     printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(gpu_index) >> > (gpu.prepare_S12_ptr.S12_buffer2_half, gpu.prepare_S12_ptr.S12_buffer2, gpu.pd_elements, (size_t)gpu_index);
        //     mcontext.sync_all_streams();
        // }
        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_All2All);

        //            dump_prepare_s12("After all2all");

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Into_Place);
        // nvtxRangePush("Sorting");
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            const uint SORT_DOWN_TO_BIT = 11;

            SaGPU& gpu = mgpus[gpu_index];
            cudaSetDevice(mcontext.get_device_id(gpu_index));

            cub::DoubleBuffer<uint64_t> keys(reinterpret_cast<uint64_t*>(gpu.prepare_S12_ptr.S12_buffer2),
                reinterpret_cast<uint64_t*>(gpu.prepare_S12_ptr.S12_buffer1));
            cub::DoubleBuffer<uint64_t> values(reinterpret_cast<uint64_t*>(gpu.prepare_S12_ptr.S12_buffer2_half),
                reinterpret_cast<uint64_t*>(gpu.prepare_S12_ptr.S12_buffer1_half));
            if (SORT_DOWN_TO_BIT < mpd_per_gpu_max_bit)
            {
                size_t temp_storage_size = 0;
                cudaError_t err = cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size, keys, values, gpu.pd_elements,
                    SORT_DOWN_TO_BIT, mpd_per_gpu_max_bit);
                CUERR_CHECK(err);
                //                printf("Needed temp storage: %zu, provided %zu.\n", temp_storage_size, ms0_reserved_len*sizeof(MergeStageSuffix));
                ASSERT(temp_storage_size <= mpd_reserved_len * sizeof(MergeStageSuffix));
                err = cub::DeviceRadixSort::SortPairs(gpu.prepare_S12_ptr.S12_result, temp_storage_size,
                    keys, values, gpu.pd_elements, SORT_DOWN_TO_BIT, mpd_per_gpu_max_bit,
                    mcontext.get_gpu_default_stream(gpu_index));
                CUERR_CHECK(err);
            }

            //                kernels::combine_S12_kv_non_coalesced _KLC_SIMPLE_(gpu.pd_elements, mcontext.get_gpu_default_stream(gpu_index))
            //                        (reinterpret_cast<MergeStageSuffixS12HalfKey*> (gpu.prepare_S12_ptr.S12_buffer2),
            //                         reinterpret_cast<MergeStageSuffixS12HalfValue*> ( gpu.prepare_S12_ptr.S12_buffer2_half),
            //                         gpu.prepare_S12_ptr.S12_result, gpu.pd_elements); CUERR

            kernels::combine_S12_kv_shared<BLOCK_SIZE, 2> _KLC_SIMPLE_ITEMS_PER_THREAD_(gpu.pd_elements, 2, mcontext.get_gpu_default_stream(gpu_index))(reinterpret_cast<MergeStageSuffixS12HalfKey*>(keys.Current()),
                reinterpret_cast<MergeStageSuffixS12HalfValue*>(values.Current()),
                gpu.prepare_S12_ptr.S12_result, gpu.pd_elements);
            CUERR;
        }
        mcontext.sync_default_streams();
        // nvtxRangePop();
        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Into_Place);

        //            dump_prepare_s12("After preparing S12");
        //            dump_final_merge("After preparing S12");
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

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            cudaSetDevice(mcontext.get_device_id(gpu_index));

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
            mcontext.get_device_temp_allocator(gpu_index).init(reinterpret_cast<MergeStageSuffixS0HalfKey*>(gpu.prepare_S0_ptr.S0_result),
                ms0_reserved_len * sizeof(MergeStageSuffixS0));
        }
        //            dump_prepare_s0("Before S0 merge");

        MergeManager merge_manager(mcontext, host_pinned_allocator);

        merge_manager.set_node_info(merge_nodes_info);

        std::vector<crossGPUReMerge::MergeRange> ranges;
        ranges.push_back({ 0, 0, (sa_index_t)NUM_GPUS - 1, (sa_index_t)(mgpus.back().num_elements - mgpus.back().pd_elements) });

        mcontext.sync_default_streams();

        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S0_Write_Out_And_Sort);

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S0_Merge);
        merge_manager.merge(ranges, S0Comparator());

        mcontext.sync_all_streams();
        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S0_Merge);

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S0_Combine);

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            cudaSetDevice(mcontext.get_device_id(gpu_index));

            size_t count = gpu.num_elements - gpu.pd_elements;

            const MergeStageSuffixS0HalfKey* sorted_and_merged_keys = is_buffer_2_current[gpu_index] ? gpu.prepare_S0_ptr.S0_buffer2_keys : gpu.prepare_S0_ptr.S0_buffer1_keys;

            const MergeStageSuffixS0HalfValue* sorted_and_merged_values = is_buffer_2_current[gpu_index] ? gpu.prepare_S0_ptr.S0_buffer2_values : gpu.prepare_S0_ptr.S0_buffer1_values;

            kernels::combine_S0_kv _KLC_SIMPLE_(count, mcontext.get_gpu_default_stream(gpu_index))(sorted_and_merged_keys, sorted_and_merged_values, gpu.prepare_S0_ptr.S0_result, count);
            CUERR;
        }
        mcontext.sync_default_streams();
        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S0_Combine);
        //            dump_final_merge("before final merge");
    }

    void final_merge()
    {
        distrib_merge::DistributedArray<MergeStageSuffix, int, sa_index_t, NUM_GPUS> inp_S12, inp_S0, result;
        // {
        //     mcontext.sync_all_streams();
        //     size_t workinLen = 0;
        //     for (size_t gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)
        //     {
        //         SaGPU& gpu = mgpus[gpu_index];
        //         workinLen += gpu.pd_elements;
        //     }
        //     std::vector<MergeStageSuffix> k(workinLen);
        //     size_t prefix_sum = 0;
        //     for (size_t gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)
        //     {
        //         SaGPU& gpu = mgpus[gpu_index];
        //         cudaMemcpy(k.data() + prefix_sum, gpu.merge_ptr.S12_result, sizeof(MergeStageSuffix) * gpu.pd_elements, cudaMemcpyDeviceToHost);
        //         prefix_sum += gpu.pd_elements;
        //     }

        //     char fileName[18];
        //     const char* text = "S12Main";
        //     sprintf(fileName, "%s", text);
        //     std::ofstream out(fileName, std::ios::binary);
        //     if (!out) {
        //         std::cerr << "Could not open file\n";
        //         //return 1;
        //     }
        //     printf("sa isa length: %lu\n", k.size());

        //     out.write(reinterpret_cast<char*>(k.data()), sizeof(MergeStageSuffix) * k.size());
        //     out.close();

        // } {
        //     mcontext.sync_all_streams();
        //     size_t workinLen = 0;
        //     for (size_t gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)
        //     {
        //         SaGPU& gpu = mgpus[gpu_index];
        //         workinLen += gpu.num_elements - gpu.pd_elements;
        //     }
        //     std::vector<MergeStageSuffix> k(workinLen);
        //     size_t prefix_sum = 0;
        //     for (size_t gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)
        //     {
        //         SaGPU& gpu = mgpus[gpu_index];
        //         cudaMemcpy(k.data() + prefix_sum, gpu.merge_ptr.S0_result, sizeof(MergeStageSuffix) * (gpu.num_elements - gpu.pd_elements), cudaMemcpyDeviceToHost);
        //         prefix_sum += gpu.num_elements - gpu.pd_elements;
        //     }

        //     char fileName[18];
        //     const char* text = "S0Main";
        //     sprintf(fileName, "%s", text);
        //     std::ofstream out(fileName, std::ios::binary);
        //     if (!out) {
        //         std::cerr << "Could not open file\n";
        //         //return 1;
        //     }
        //     printf("sa isa length: %lu\n", k.size());

        //     out.write(reinterpret_cast<char*>(k.data()), sizeof(MergeStageSuffix) * k.size());
        //     out.close();

        // }
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];

            const size_t S0_count = gpu.num_elements - gpu.pd_elements;
            const size_t S12_count = gpu.pd_elements;

            const size_t result_count = gpu.num_elements;
            inp_S12[gpu_index] = { gpu_index, (sa_index_t)S12_count, gpu.merge_ptr.S12_result, nullptr, nullptr, nullptr };
            inp_S0[gpu_index] = { gpu_index, (sa_index_t)S0_count, gpu.merge_ptr.S0_result, nullptr, nullptr, nullptr };
            result[gpu_index] = { gpu_index, (sa_index_t)result_count, gpu.merge_ptr.S12_result, nullptr, gpu.merge_ptr.buffer, nullptr };
            mcontext.get_device_temp_allocator(gpu_index).init(gpu.merge_ptr.remaining_storage,
                gpu.merge_ptr.remaining_storage_size);
        }
        auto h_temp_mem = mmemory_manager.get_host_temp_mem();
        QDAllocator qd_alloc_h_temp(h_temp_mem.first, h_temp_mem.second);
        distrib_merge::DistributedMerge<MergeStageSuffix, int, sa_index_t, NUM_GPUS, DistribMergeTopology>::
            merge_async(inp_S12, inp_S0, result, MergeCompFunctor(), false, mcontext, qd_alloc_h_temp);

        //            dump_final_merge("after final merge");

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            cudaSetDevice(mcontext.get_device_id(gpu_index));
            kernels::from_merge_suffix_to_index _KLC_SIMPLE_(gpu.num_elements, mcontext.get_gpu_default_stream(gpu_index))(gpu.merge_ptr.S12_result, gpu.merge_ptr.result, gpu.num_elements);
            CUERR;
        }
        mcontext.sync_default_streams();
    }

    void copy_result_to_host()
    {
        sa_index_t* h_result = mmemory_manager.get_h_result();
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            cudaSetDevice(mcontext.get_device_id(gpu_index));
            // printf("[%u] result length: %lu\n", gpu_index, gpu.num_elements);
            cudaMemcpyAsync(h_result + gpu.offset, gpu.merge_ptr.result, gpu.num_elements * sizeof(sa_index_t),
                cudaMemcpyDeviceToHost, mcontext.get_gpu_default_stream(gpu_index));
            CUERR;
        }
        mcontext.sync_default_streams();
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
void alltoallMeasure(MultiGPUContext<NUM_GPUS>& context, int max) {

    for (int i = max; i <= max; i++) {
        MultiSplit<NUM_GPUS> multi_split(context);
        All2All<NUM_GPUS> all2all(context);
        std::array<sa_index_t*, NUM_GPUS> d_A_send;
        std::array<sa_index_t*, NUM_GPUS> d_A_recv;
        std::array<void*, NUM_GPUS> temp_buffer;
        std::array<sa_index_t*, NUM_GPUS> d_A_recv_temp;
        size_t N = 2 << i;

        // Allocate memory for A on CPU
        sa_index_t* A = (sa_index_t*)malloc(N * sizeof(sa_index_t));

        // Initialize all elements of A to random values
        for (size_t i = 0; i < N; i++) {
            A[i] = i;
        }

        std::shuffle(A, A + N, std::default_random_engine());
        size_t per_gpu = N / NUM_GPUS;

        std::array<size_t, NUM_GPUS> temp_storages;
        // printf("N: %u, per_gpu: %u\n", N, per_gpu);
        for (size_t i = 0; i < NUM_GPUS; i++) {
            // cudaStream_t stream = context.get_gpu_default_stream(i);
            sa_index_t* d_A, * d_A_rec, * d_A_te, * d_A_recv_tem;
            cudaMalloc(&d_A, per_gpu * sizeof(sa_index_t));
            d_A_send[i] = d_A;
            cudaMemcpy(d_A_send[i], A + per_gpu * i, per_gpu * sizeof(sa_index_t), cudaMemcpyHostToDevice);

            cudaMalloc(&d_A_rec, per_gpu * sizeof(sa_index_t));
            d_A_recv[i] = d_A_rec;
            cudaMemset(d_A_recv[i], 0, per_gpu * sizeof(sa_index_t));

            cudaMalloc(&d_A_recv_tem, per_gpu * sizeof(sa_index_t));
            d_A_recv_temp[i] = d_A_recv_tem;
            cudaMemset(d_A_recv_temp[i], 0, per_gpu * sizeof(sa_index_t));

            cub::DeviceRadixSort::SortKeys(nullptr, temp_storages[i],
                d_A_recv[i], d_A_recv[i], per_gpu);
            void* temp;
            temp_storages[i] = std::max(temp_storages[i], 1024ul);
            temp_storages[i] = std::max(temp_storages[i], per_gpu);
            cudaMalloc(&temp, temp_storages[i]);
            temp_buffer[i] = temp;
            cudaMemset(temp_buffer[i], 0, temp_storages[i]);


            // printArray << <1, 1 >> > (d_A_send[i], d_A_send[i], per_gpu, i);
        }

        context.sync_default_streams();
        int loop_count = 1;
        std::array<MultiSplitNodeInfoT<sa_index_t, sa_index_t, sa_index_t>, NUM_GPUS> multi_split_node_info;
        std::array<All2AllNodeInfoT<sa_index_t, sa_index_t, sa_index_t>, NUM_GPUS> all2all_node_info;
        split_table_tt<sa_index_t, NUM_GPUS> split_table;
        std::array<sa_index_t, NUM_GPUS> dest_lens, src_lens;
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            // context.sync_default_streams();
            multi_split_node_info[gpu_index].src_keys = d_A_send[gpu_index];
            multi_split_node_info[gpu_index].src_values = d_A_send[gpu_index];
            multi_split_node_info[gpu_index].src_len = per_gpu;

            multi_split_node_info[gpu_index].dest_keys = d_A_recv_temp[gpu_index];
            multi_split_node_info[gpu_index].dest_values = d_A_recv_temp[gpu_index];
            multi_split_node_info[gpu_index].dest_len = per_gpu;
            context.get_device_temp_allocator(gpu_index).init(temp_buffer[gpu_index], temp_storages[gpu_index] * sizeof(sa_index_t));
        }

        for (uint src = 0; src < NUM_GPUS; ++src)
        {
            for (uint dst = 0; dst < NUM_GPUS; ++dst)
            {
                split_table[src][dst] = per_gpu / NUM_GPUS;
            }
        }

        PartitioningFunctor<uint> f(per_gpu, NUM_GPUS - 1);
        //multi_split.execAsync(multi_split_node_info, split_table, src_lens, dest_lens, f);

        context.sync_default_streams();

        // Warm-up loop

        for (int i = 1; i <= 0; i++) {

            for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
            {
                all2all_node_info[gpu_index].src_keys = d_A_recv_temp[gpu_index];
                all2all_node_info[gpu_index].src_values = d_A_recv_temp[gpu_index];
                all2all_node_info[gpu_index].src_len = per_gpu;

                all2all_node_info[gpu_index].dest_keys = d_A_recv[gpu_index];
                all2all_node_info[gpu_index].dest_values = d_A_recv[gpu_index];
                all2all_node_info[gpu_index].dest_len = per_gpu;
            }
            all2all.execAsync(all2all_node_info, split_table);
            context.sync_all_streams();
        }

        // for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        // {
        //     size_t temp_storage = 0;
        //     cub::DeviceRadixSort::SortKeys(nullptr, temp_storage,
        //         d_A_recv[gpu_index], d_A_recv[gpu_index], per_gpu);

        //     ASSERT(temp_storage <= temp_storages[gpu_index]);
        //     cub::DeviceRadixSort::SortKeys(temp_buffer[gpu_index], temp_storage,
        //         d_A_recv[gpu_index], d_A_send[gpu_index], per_gpu, 0, 32, context.get_gpu_default_stream(gpu_index));
        // }
        // context.sync_all_streams();
        // for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        // {
        //     cudaMemcpy(&A[gpu_index], &d_A_send[gpu_index][per_gpu - 1], sizeof(sa_index_t), cudaMemcpyDeviceToHost);
        // }
        // printf("1: %u, 2: %u: 3: %u: 4: %u\n", A[0], A[1], A[2], A[3]);
        // printArray << <1, 1 >> > (d_A_send[0], d_A_send[1], per_gpu, 3);

        // Time ping-pong for loop_count iterations of data transfer size 8*N bytes
        double elapsed_time;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

        for (int i = 1; i <= loop_count; i++) {
            for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
            {
                all2all_node_info[gpu_index].src_keys = d_A_recv_temp[gpu_index];
                all2all_node_info[gpu_index].src_values = d_A_recv_temp[gpu_index];
                all2all_node_info[gpu_index].src_len = per_gpu;

                all2all_node_info[gpu_index].dest_keys = d_A_recv[gpu_index];
                all2all_node_info[gpu_index].dest_values = d_A_recv[gpu_index];
                all2all_node_info[gpu_index].dest_len = per_gpu;
            }
            all2all.execAsync(all2all_node_info, split_table);
            context.sync_all_streams();
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::chrono::duration<double> elapsed_seconds = end - start;
        elapsed_time = elapsed_seconds.count();
        long int num_B = 8 * N;
        long int B_in_GB = 1 << 30;
        double num_GB = (double)num_B / (double)B_in_GB;
        double avg_time_per_transfer = elapsed_time / (2.0 * (double)loop_count);

        printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB / avg_time_per_transfer);
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            cudaFree(d_A_send[gpu_index]);
            cudaFree(d_A_recv[gpu_index]);
            cudaFree(d_A_recv_temp[gpu_index]);
            cudaFree(temp_buffer[gpu_index]);
        }
        free(A);
    }

}

void sample_sort_merge_measure(MultiGPUContext<NUM_GPUS>& mcontext) {
    using merge_types = crossGPUReMerge::mergeTypes<uint64_t, uint64_t>;
    using MergeManager = crossGPUReMerge::ReMergeManager<NUM_GPUS, merge_types, ReMergeTopology>;
    using MergeNodeInfo = crossGPUReMerge::MergeNodeInfo<merge_types>;

    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<std::mt19937::result_type> randomDistChar(0, UINT64_MAX);
    size_t rounds = 21;
    for (size_t r = 0; r < rounds; r++)
    {
        size_t data_size = (128UL << r) - 1UL;
        std::array<MergeNodeInfo, NUM_GPUS> merge_nodes_info;

        std::array<uint64_t*, NUM_GPUS> d_keys_gpu;
        std::array<uint64_t*, NUM_GPUS> d_values_gpu;
        size_t temp_storage_size = sizeof(uint64_t) * data_size * 2;
        std::array<void*, NUM_GPUS> temps;
        for (size_t gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)
        {
            cudaSetDevice(mcontext.get_device_id(gpu_index));
            std::vector<uint64_t> h_keys(data_size);
            std::vector<uint64_t> h_values(data_size);

            for (size_t j = 0; j < data_size; j++)
            {
                h_keys[j] = randomDistChar(g);
                h_values[j] = randomDistChar(g);
            }
            uint64_t* d_keys;
            cudaMalloc(&d_keys, sizeof(uint64_t) * 2 * data_size);
            cudaMemcpy(d_keys, h_keys.data(), data_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
            // uint64_t* d_values;
            // cudaMalloc(&d_values, sizeof(uint64_t) * 2 * data_size);
            // // cudaMemset(d_values, 0, sizeof(uint64_t) * data_size);
            // cudaMemcpy(d_values, h_values.data(), data_size * sizeof(uint64_t), cudaMemcpyHostToDevice);

            d_keys_gpu[gpu_index] = d_keys;
            // d_values_gpu[gpu_index] = d_values;


            void* temp;
            cudaMalloc(&temp, temp_storage_size);
            mcontext.get_device_temp_allocator(gpu_index).init(temp, temp_storage_size);

            merge_nodes_info[gpu_index] = { data_size, 0, (sa_index_t)gpu_index, d_keys_gpu[gpu_index], nullptr , d_keys_gpu[gpu_index] + data_size,  nullptr,  nullptr, nullptr };
            mcontext.get_mgpu_default_context_for_device(gpu_index).set_device_temp_mem(temp, temp_storage_size);
            temps[gpu_index] = temp;
        }
        uint64_t* h_temp_mem;
        cudaMallocHost(&h_temp_mem, temp_storage_size);
        CUERR;
        memset(h_temp_mem, 0, temp_storage_size);
        QDAllocator host_pinned_allocator(h_temp_mem, temp_storage_size);
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
        for (size_t gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)
        {
            cudaSetDevice(mcontext.get_device_id(gpu_index));
            mgpu::mergesort(d_keys_gpu[gpu_index], data_size, std::less<uint64_t>(), mcontext.get_mgpu_default_context_for_device(gpu_index));
        }
        mcontext.sync_all_streams();
        t.stop();
        t.start("merge");
        for (size_t gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)
        {
            mcontext.get_mgpu_default_context_for_device(gpu_index).reset_temp_memory();
        }
        merge_manager.merge(ranges, std::less<uint64_t>());
        mcontext.sync_all_streams();
        t.stop();
        t.stop();
        size_t mb = 1 << 20;
        double num_mB = (double)bytes / (double)mb;

        printf("elements: %lu, %8.3f MB\n", data_size, num_mB);
        // cudaMemcpy(h_temp_mem, d_keys + data_size, sizeof(uint64_t) * data_size, cudaMemcpyDeviceToHost);
        // for (size_t i = 0; i < data_size; i++)
        // {
        //     printf("[%lu] merged key[%3lu]: %20lu\n", world_rank(), i, h_temp_mem[i]);
        // }
        cudaFreeHost(h_temp_mem);
        for (size_t gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)
        {
            cudaSetDevice(mcontext.get_device_id(gpu_index));
            cudaFree(temps[gpu_index]);
            cudaFree(d_keys_gpu[gpu_index]);
            // cudaFree(d_values_gpu[gpu_index]);
        }
    }
}

int main(int argc, char** argv)
{
    using namespace kamping;
    kamping::Environment e;
    Communicator comm;
    if (argc != 4)
    {
        error("Usage: sa-test <ofile> <measureFile> <ifile>!");
    }

    char* input = nullptr;
    cudaSetDevice(0);
    size_t realLen;
    size_t maxLength = size_t(1024 * 1024) * size_t(150 * 3 * NUM_GPUS);
    size_t inputLen = read_file_into_host_memory(&input, argv[3], realLen, sizeof(sa_index_t), maxLength, 0);
#ifdef DGX1_TOPOLOGY
    //    const std::array<uint, NUM_GPUS> gpu_ids { 0, 3, 2, 1,  5, 6, 7, 4 };
    //    const std::array<uint, NUM_GPUS> gpu_ids { 1, 2, 3, 0,    4, 7, 6, 5 };
    //    const std::array<uint, NUM_GPUS> gpu_ids { 3, 2, 1, 0,    4, 5, 6, 7 };
    const std::array<uint, NUM_GPUS> gpu_ids{ 3, 2, 1, 0, 4, 7, 6, 5 };

    MultiGPUContext<NUM_GPUS> context(&gpu_ids);
#else 
    const std::array<uint, NUM_GPUS> gpu_ids{ 0,0,0,0 };
    MultiGPUContext<NUM_GPUS> context;
    // alltoallMeasure(context, std::stoi(argv[1]));
    // return 0;
    // sample_sort_merge_measure(context);
    // auto& t = kamping::measurements::timer();
    // t.aggregate_and_print(
    //     kamping::measurements::SimpleJsonPrinter{ std::cout, {} });
    // std::cout << std::endl;
    // t.aggregate_and_print(kamping::measurements::FlatPrinter{});
    // std::cout << std::endl;
    // std::ofstream outFile(argv[2], std::ios::app);
    // t.aggregate_and_print(
    //     kamping::measurements::SimpleJsonPrinter{ outFile, {} });
    // std::cout << std::endl;
    // t.aggregate_and_print(kamping::measurements::FlatPrinter{});
    // std::cout << std::endl;
    // return 0;

#endif
    SuffixSorter sorter(context, realLen, input);
    sorter.alloc();

    sorter.do_sa();

    // write_array(argv[1], sorter.get_result(), realLen);

    sorter.done();

    sorter.print_pd_stats();
    sorter.get_perf_measurements().print(argv[2]);

    cudaFreeHost(input);
    CUERR;
}
