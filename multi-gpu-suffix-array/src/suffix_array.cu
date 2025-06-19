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
#include <kamping/data_buffer.hpp>
#include <kamping/environment.hpp>
#include <kamping/measurements/printer.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/named_parameters.hpp>
#include <kamping/communicator.hpp>
#include <kamping/p2p/recv.hpp>
#include <kamping/p2p/send.hpp>

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
        TIMERSTART(Total);

        TIMER_START_MAIN_STAGE(MainStages::Copy_Input);
        copy_input();
        TIMER_STOP_MAIN_STAGE(MainStages::Copy_Input);
        // printf("Copy Input\n");

        TIMER_START_MAIN_STAGE(MainStages::Produce_KMers);
        produce_kmers();
        // printf("Produce kmers\n");

        TIMER_STOP_MAIN_STAGE(MainStages::Produce_KMers);

        //            mpd_sorter.dump("After K-Mers");

        mtook_pd_iterations = mpd_sorter.sort(4);

        //            mpd_sorter.dump("done");
        TIMER_START_MAIN_STAGE(MainStages::Prepare_S12_for_Merge);
        prepare_S12_for_merge();
        //
        mcontext.sync_all_streams();
        printf("[%lu] prepare s12 for merge done\n", world_rank());
        comm_world().barrier();
        //

        TIMER_STOP_MAIN_STAGE(MainStages::Prepare_S12_for_Merge);
        TIMER_START_MAIN_STAGE(MainStages::Prepare_S0_for_Merge);
        prepare_S0_for_merge();
        //
        mcontext.sync_all_streams();
        printf("[%lu] prepare s0 for merge done\n", world_rank());
        comm_world().barrier();
        //
        TIMER_STOP_MAIN_STAGE(MainStages::Prepare_S0_for_Merge);
        TIMER_START_MAIN_STAGE(MainStages::Final_Merge);
        final_merge();
        //
        mcontext.sync_all_streams();
        printf("[%lu] final merge done\n", world_rank());
        comm_world().barrier();
        //
        TIMER_STOP_MAIN_STAGE(MainStages::Final_Merge);
        TIMER_START_MAIN_STAGE(MainStages::Copy_Results);
        copy_result_to_host();
        //
        mcontext.sync_all_streams();
        printf("[%lu] complete\n", world_rank());
        comm_world().barrier();
        //
        TIMER_STOP_MAIN_STAGE(MainStages::Copy_Results);
        TIMERSTOP(Total);

        mperf_measure.done();
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
        printf("minput_len: %lu, mper_gpu %lu\n", minput_len, mper_gpu);
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
        //for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        //{
        auto gpu_index = world_rank();
        SaGPU& gpu = mgpus[gpu_index];

        // Need the halo to the right for kmers...
        size_t copy_len = std::min(gpu.num_elements + sizeof(kmer_t), minput_len - gpu.offset);

        cudaSetDevice(mcontext.get_device_id(gpu_index));
        cudaMemcpyAsync(gpu.pd_ptr.Input, minput + gpu.offset, copy_len, cudaMemcpyHostToDevice,
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
        //for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        //{
        auto gpu_index = world_rank();
        SaGPU& gpu = mgpus[gpu_index];

        cudaSetDevice(mcontext.get_device_id(gpu_index));
        //                kernels::produce_index_kmer_tuples _KLC_SIMPLE_(gpu.num_elements, mcontext.get_gpu_default_stream(gpu_index))
        //                        ((char*)gpu.input, offset, gpu.pd_index, gpu.pd_kmers, gpu.num_elements); CUERR;
        kernels::produce_index_kmer_tuples_12_64 _KLC_SIMPLE_(gpu.num_elements, mcontext.get_gpu_default_stream(gpu_index))((char*)gpu.pd_ptr.Input, gpu.pd_offset, gpu.pd_ptr.Isa, reinterpret_cast<ulong1*>(gpu.pd_ptr.Sa_rank),
            SDIV(gpu.num_elements, 12) * 12);
        CUERR;
        //}
        if (gpu_index == NUM_GPUS - 1)
        {
            kernels::fixup_last_four_12_kmers_64 << <1, 4, 0, mcontext.get_gpu_default_stream(gpu_index) >> > (reinterpret_cast<ulong1*>(mgpus.back().pd_ptr.Sa_rank) + mgpus.back().pd_elements - 4);
        }
        mcontext.sync_default_streams();
    }

    void prepare_S12_for_merge()
    {
        std::array<MultiSplitNodeInfoT<sa_index_t, sa_index_t, sa_index_t>, NUM_GPUS> multi_split_node_info;
        std::array<All2AllNodeInfoT<MergeStageSuffixS12HalfKey, MergeStageSuffixS12HalfValue, sa_index_t>, NUM_GPUS> all2all_node_info;
        split_table_tt<sa_index_t, NUM_GPUS> split_table;
        std::array<sa_index_t, NUM_GPUS> dest_lens, src_lens;

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Multisplit);
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            if (world_rank() == gpu_index)
            {

                // cudaSetDevice(0);
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

        mmulti_split.execKVAsync(multi_split_node_info, split_table, src_lens, dest_lens, f);

        mcontext.sync_default_streams();

        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Multisplit);

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Out);

        //for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            uint gpu_index = world_rank();
            SaGPU& gpu = mgpus[gpu_index];
            cudaSetDevice(mcontext.get_device_id(gpu_index));

            const sa_index_t* next_Isa = nullptr; //= (gpu_index + 1 < NUM_GPUS) ? mgpus[gpu_index + 1].prepare_S12_ptr.Isa : nullptr;
            const unsigned char* next_Input = nullptr; //= (gpu_index + 1 < NUM_GPUS) ? mgpus[gpu_index + 1].prepare_S12_ptr.Input : nullptr;

            if (gpu_index > 0) {
                std::span<sa_index_t> sbIsa(gpu.prepare_S12_ptr.Isa, 1);
                comm_world().isend(send_buf(sbIsa), send_count(1), tag(0), destination((size_t)gpu_index - 1));

                std::span<const unsigned char> sbInput(gpu.prepare_S12_ptr.Input, 1);
                comm_world().isend(send_buf(sbInput), send_count(1), tag(1), destination((size_t)gpu_index - 1));
            }
            if (gpu_index + 1 < NUM_GPUS) {
                sa_index_t* tempIsa = mcontext.get_device_temp_allocator(gpu_index).get<sa_index_t>(1);
                std::span<sa_index_t> rbIsa(tempIsa, 1);
                comm_world().recv(recv_buf(rbIsa), tag(0), recv_count(1));
                next_Isa = tempIsa;
                unsigned char* tempInput = mcontext.get_device_temp_allocator(gpu_index).get<unsigned char>(1);
                std::span<unsigned char> rbInput(tempInput, 1);
                comm_world().recv(recv_buf(rbInput), tag(1), recv_count(1));
                next_Input = tempInput;
            }

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
        //
        mcontext.get_device_temp_allocator(world_rank()).reset();
        //
        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Out);
        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_All2All);

        //            dump_prepare_s12("After split");

        mall2all.execKVAsync(all2all_node_info, split_table);
        mcontext.sync_all_streams();
        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_All2All);

        //            dump_prepare_s12("After all2all");

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Into_Place);

        //for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            uint gpu_index = world_rank();
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
            // cudaSetDevice(mcontext.get_device_id(gpu_index));
            size_t count = gpu.num_elements - gpu.pd_elements;
            if (world_rank() == gpu_index) {

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
            }

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

            if (world_rank() == gpu_index) {
                mcontext.get_device_temp_allocator(gpu_index).init(reinterpret_cast<MergeStageSuffixS0HalfKey*>(gpu.prepare_S0_ptr.S0_result),
                    ms0_reserved_len * sizeof(MergeStageSuffixS0));
            }
        }
        //            dump_prepare_s0("Before S0 merge");

        MergeManager merge_manager(mcontext, host_pinned_allocator);

        merge_manager.set_node_info(merge_nodes_info);

        std::vector<crossGPUReMerge::MergeRange> ranges;
        ranges.push_back({ 0, 0, (sa_index_t)NUM_GPUS - 1, (sa_index_t)(mgpus.back().num_elements - mgpus.back().pd_elements) });

        mcontext.sync_default_streams();

        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S0_Write_Out_And_Sort);

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S0_Merge);
        merge_manager.merge(ranges, S0Comparator(), nullptr);

        mcontext.sync_all_streams();
        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S0_Merge);

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S0_Combine);

        //for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            uint gpu_index = world_rank();
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

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {

            SaGPU& gpu = mgpus[gpu_index];

            const size_t S0_count = gpu.num_elements - gpu.pd_elements;
            const size_t S12_count = gpu.pd_elements;
            const size_t result_count = gpu.num_elements;
            inp_S12[gpu_index] = { gpu_index, (sa_index_t)S12_count, gpu.merge_ptr.S12_result, nullptr, nullptr, nullptr };
            inp_S0[gpu_index] = { gpu_index, (sa_index_t)S0_count, gpu.merge_ptr.S0_result, nullptr, nullptr, nullptr };
            result[gpu_index] = { gpu_index, (sa_index_t)result_count, gpu.merge_ptr.S12_result, nullptr, gpu.merge_ptr.buffer, nullptr };
            if (world_rank() == gpu_index) {
                mcontext.get_device_temp_allocator(gpu_index).init(gpu.merge_ptr.remaining_storage,
                    gpu.merge_ptr.remaining_storage_size);
            }
        }
        auto h_temp_mem = mmemory_manager.get_host_temp_mem();
        QDAllocator qd_alloc_h_temp(h_temp_mem.first, h_temp_mem.second);
        distrib_merge::DistributedMerge<MergeStageSuffix, int, sa_index_t, NUM_GPUS, DistribMergeTopology>::
            merge_async(inp_S12, inp_S0, result, MergeCompFunctor(), false, mcontext, qd_alloc_h_temp);

        mcontext.sync_default_streams();
        printf("[%lu] merge async done\n", world_rank());
        comm_world().barrier();
        //            dump_final_merge("after final merge");

        //for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            uint gpu_index = world_rank();
            SaGPU& gpu = mgpus[gpu_index];
            // cudaSetDevice(mcontext.get_device_id(gpu_index));
            kernels::from_merge_suffix_to_index _KLC_SIMPLE_(gpu.num_elements, mcontext.get_gpu_default_stream(gpu_index))(gpu.merge_ptr.S12_result, gpu.merge_ptr.result, gpu.num_elements);
            CUERR;
        }
        mcontext.sync_default_streams();
    }

    void copy_result_to_host()
    {
        sa_index_t* h_result = mmemory_manager.get_h_result();
        //for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        //{
        uint gpu_index = world_rank();
        SaGPU& gpu = mgpus[gpu_index];
        cudaSetDevice(mcontext.get_device_id(gpu_index));
        cudaMemcpyAsync(h_result + gpu.offset, gpu.merge_ptr.result, gpu.num_elements * sizeof(sa_index_t),
            cudaMemcpyDeviceToHost, mcontext.get_gpu_default_stream(gpu_index));
        CUERR;
        //}
        mcontext.sync_default_streams();
        std::vector<sa_index_t> recv;
        recv.clear();
        std::span<sa_index_t> sb(h_result + gpu.offset, gpu.num_elements);
        auto [sendCounts] = comm_world().gatherv(send_buf(sb), recv_buf<resize_to_fit>(recv), recv_counts_out());
        int sumCounts = 0;
        int i = 0;
        for (auto count : sendCounts) {
            ASSERT(count == mgpus[i].num_elements);
            memcpy(h_result + mgpus[i].offset, recv.data() + sumCounts, sizeof(sa_index_t) * count);
            sumCounts += count;
            i++;
        }
        // for (int gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index) {
        //     std::span<sa_index_t> buffer(h_result + gpu.offset, gpu.num_elements);
        //     comm_world().bcast(send_recv_buf(buffer), root(gpu_index));
        // }
        //std::span<sa_index_t> rb(h_result, );

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
int main(int argc, char** argv)
{
    using namespace kamping;
    kamping::Environment e;
    Communicator comm;
    // int world_rank = comm.rank_signed();

    // printf("* Rank: %d\n", comm.rank_signed());
    // // std::vector<int> input2(2u * comm.size(), comm.rank_signed());
    // // std::vector<int> output = comm.alltoall(send_buf(input2));
    // // printf("Rank: %d, Size: %d, SRank: %d\n", comm.rank(), output[0], output[1]);
    // printf("* Allocate memory [%d],GPU\n", world_rank);
    // int* d_a;
    // if (cudaMalloc((void**)&d_a, 1000 * sizeof(int)) != cudaSuccess)
    // {
    //     cudaMemset(d_a, 0, 1000 * sizeof(int));

    //     printf("Error malloc", world_rank);
    //     return 1;
    // }

    // int err = 0;
    // MPI_Status status;
    // // From [1],GPU to [0],GPU
    // if (world_rank == 1)
    // {
    //     printf("* Send from [%d],GPU Data %d\n", world_rank, d_a[0]);
    //     // cudaMemset(d_a, 1, 1000 * sizeof(int));

    //     comm.send(d_a, destination(0));
    // }
    // else if (world_rank == 0)
    // {
    //     comm.recv(d_a);
    //     // comm.recv<int *>(recv_buf(d_a));
    //     printf("* Receive to [%d],GPU Data %d\n", world_rank, d_a[0]);
    // }
    // // auto received = comm.sendrecv<int>(send_buf(input), destination(dest));
    // printf("* Free memory on [%d],GPU\n", world_rank);
    // cudaFree(d_a);

    // e.finalize();
    // return 0;

    if (argc != 4)
    {
        error("Usage: sa-test <ifile> <ofile>!");
    }

    char* input = nullptr;

    cudaSetDevice(0);
    //char* pci;
    //cudaDeviceGetPCIBusId(pci, 100, 0);
    //std::cout << pci << std::endl;

    size_t realLen;
    size_t inputLen = read_file_into_host_memory(&input, argv[3], realLen, sizeof(sa_index_t), 0);

#ifdef DGX1_TOPOLOGY
    //    const std::array<uint, NUM_GPUS> gpu_ids { 0, 3, 2, 1,  5, 6, 7, 4 };
    //    const std::array<uint, NUM_GPUS> gpu_ids { 1, 2, 3, 0,    4, 7, 6, 5 };
    //    const std::array<uint, NUM_GPUS> gpu_ids { 3, 2, 1, 0,    4, 5, 6, 7 };
    const std::array<uint, NUM_GPUS> gpu_ids{ 3, 2, 1, 0, 4, 7, 6, 5 };

    MultiGPUContext<NUM_GPUS> context(&gpu_ids);
#else
    const std::array<uint, NUM_GPUS> gpu_ids2{ 0,0,0,0 };

    MultiGPUContext<NUM_GPUS> context(&gpu_ids2);
    // printf("Multi GPU Context\n");
#endif
    SuffixSorter sorter(context, realLen, input);
    // printf("Suffix Sorter\n");

    sorter.alloc();
    // printf("Alloc\n");

    // auto stringPath = ((std::string)argv[3]);
    // int pos = stringPath.find_last_of("/\\");
    // auto fileName = (pos == std::string::npos) ? argv[3] : stringPath.substr(pos + 1);

    // auto& t = kamping::measurements::timer();
    // t.synchronize_and_start(fileName);

    sorter.do_sa();

    // t.stop();
    for (int i = 0; i < realLen; i++) {
        printf("[%lu]: %u: %s\n", world_rank(), sorter.get_result()[i], input + sorter.get_result()[i]);
    }
    //write_array(argv[2], sorter.get_result(), realLen);

    sorter.done();

    //sorter.print_pd_stats();
    //sorter.get_perf_measurements().print();

    cudaFreeHost(input);
    CUERR;

    return 0;
    // std::ofstream outFile(argv[1], std::ios::app);
    // t.aggregate_and_print(
    //     kamping::measurements::SimpleJsonPrinter{ outFile, {} });
    // std::cout << std::endl;
    // t.aggregate_and_print(kamping::measurements::FlatPrinter{});
    // std::cout << std::endl;
}
