#ifndef PREFIX_DOUBLING_HPP
#define PREFIX_DOUBLING_HPP

#include <array>
#include <cmath>

#include "moderngpu/kernel_load_balance.hxx"
#include "moderngpu/kernel_segsort.hxx"
#include "my_mgpu_context.hxx"

#include "suffix_types.h"
#include "remerge/remergemanager.hpp"
#include "suffixarraymemorymanager.hpp"
#include "suffix_array_kernels.cuh"
#include "suffix_array_templated_kernels.cuh"
#include <span>
#include <kamping/p2p/recv.hpp>
#include <kamping/p2p/send.hpp>

// #define DEBUG_SET_ZERO_TO_SEE_BETTER
// #define DUMP_EVERYTHING
__global__ void printArray(uint32_t* key, uint32_t* value, size_t size, size_t rank)
{
    for (size_t i = 0; i < size; i++) {

        printf("[%lu]: Isa 1: %u, Sa_index 2: %u\n", rank, key[i], value[i]);


    }
    printf("---------------------------------------------------------------------------\n");
}
__global__ void printArray(uint64_t* key, uint64_t* value, size_t size, size_t rank)
{
    for (size_t i = 0; i < size; i++) {

        printf("[%lu]: sa_rank 1: %lu, old_ranks 2: %lu\n", rank, key[i], value[i]);


    }
    printf("---------------------------------------------------------------------------\n");
}
template<typename key_>
__global__ void printArray(key_* key, key_* value, size_t size, size_t rank)
{
    for (size_t i = 0; i < size; i++) {

        printf("[%lu]: sa_rank 1: %lu, old_ranks 2: %lu\n", rank, key[i], value[i]);


    }
    printf("---------------------------------------------------------------------------\n");
}
struct MaxFunctor
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(const T& a, const T& b) const
    {
        return (b > a) ? b : a;
    }
};

struct NotEqualsFunctor
{
    sa_index_t compare;
    __forceinline__
        NotEqualsFunctor(sa_index_t compare_) : compare(compare_) {}
    __device__ __forceinline__ bool operator()(const sa_index_t& a) const
    {
        return (a != compare);
    }
};

template <typename value_t>
struct PartitioningFunctor : public std::unary_function<value_t, uint32_t>
{
    sa_index_t split_divisor;
    uint max_v;

    __forceinline__
        PartitioningFunctor(sa_index_t split_divisor_, uint max_v_)
        : split_divisor(split_divisor_), max_v(max_v_)
    {
    }

    __host__ __device__ __forceinline__ uint32_t operator()(value_t x) const
    {
        return min(((x) / split_divisor), max_v);
    }
};

template <typename value_t>
struct PartioningFunctorFilteringZeroes
{
    sa_index_t split_divisor;
    uint max_v;

    __forceinline__
        PartioningFunctorFilteringZeroes(sa_index_t split_divisor_, uint max_v_)
        : split_divisor(split_divisor_), max_v(max_v_)
    {
    }

    __host__ __device__ __forceinline__ uint operator()(value_t x) const
    {
        return ((x) == 0) ? 0 : min(((x) / split_divisor), max_v);
    }
};

#define TIMER_START_MAIN_STAGE(stage) mperf_measure.start_main_stage(stage)
#define TIMER_STOP_MAIN_STAGE(stage) mperf_measure.stop_main_stage(stage)

#define TIMER_START_LOOP_STAGE(stage) mperf_measure.start_loop_stage(stage)
#define TIMER_STOP_LOOP_STAGE(stage) mperf_measure.stop_loop_stage(stage)

#define TIMER_START_WRITE_ISA_STAGE(stage) mperf_measure.start_write_isa_stage(stage)
#define TIMER_STOP_WRITE_ISA_STAGE(stage) mperf_measure.stop_write_isa_stage(stage)

#define TIMER_START_FETCH_RANK_STAGE(stage) mperf_measure.start_fetch_rank_stage(stage)
#define TIMER_STOP_FETCH_RANK_STAGE(stage) mperf_measure.stop_fetch_rank_stage(stage)

class PrefixDoublingSuffixSorter
{
    static const int BLOCK_SIZE = 1024;
    static const size_t MAX_GRID_SIZE = 2048;

    using Context = MultiGPUContext<NUM_GPUS>;

    using merge_types = crossGPUReMerge::mergeTypes<sa_index_t, sa_index_t>;
    using ReMergeManager = crossGPUReMerge::ReMergeManager<NUM_GPUS, merge_types, ReMergeTopology>;
    using ReMergeNodeInfo = crossGPUReMerge::MergeNodeInfo<merge_types>;
    using MemoryManager = SuffixArrayMemoryManager<Context::num_gpus, sa_index_t>;
    using MainStages = perf_rec::MainStages;
    using LoopStages = perf_rec::LoopStages;
    using WriteISAStages = perf_rec::WriteISAStages;
    using FetchRankStages = perf_rec::FetchRankStages;

    struct SaGPUIterationStats
    {
        size_t num_elements;
        size_t num_segments;
    };

    using PDStats = std::array<SaGPUIterationStats, 36>;

    struct SaGPU
    {
        uint index;
        sa_index_t* Sa_index;
        sa_index_t* Sa_rank;
        sa_index_t* Isa;
        sa_index_t* Segment_heads;
        sa_index_t* Old_ranks;
        sa_index_t* Temp1;
        sa_index_t* Temp2;
        sa_index_t* Temp3;
        sa_index_t* Temp4;

        size_t working_len;
        size_t isa_len;
        size_t offset;
        size_t num_segments;
        sa_index_t old_rank_start, old_rank_end;
        sa_index_t first_segment_end, last_segment_start;
        sa_index_t rank_of_first_entry_within_segment;

        PDStats stats;
    };

    Context& mcontext;
    sa_index_t* mhost_temp_mem;
    size_t mhost_temp_mem_size;
    QDAllocator mhost_temp_pinned_allocator;
    MemoryManager& mmemory_manager;
    ReMergeManager mremerge_manager;
    MultiSplit<NUM_GPUS>& mmulti_split;
    All2All<NUM_GPUS>& mall2all;
    SuffixArrayPerformanceMeasurements& mperf_measure;

    std::array<SaGPU, NUM_GPUS> mgpus;
    SaGPU mdebugHostGPU;

    std::array<ReMergeNodeInfo, NUM_GPUS> mmerge_nodes_info;

    size_t minput_len;
    size_t misa_divisor, mlast_gpu_len;
    sa_index_t mwrite_isa_sort_high_bit;
    size_t mreserved_len;
    size_t madditional_temp_storage_size;

public:
    PrefixDoublingSuffixSorter(Context& context,
        MemoryManager& memory_manager,
        MultiSplit<NUM_GPUS>& multi_split,
        All2All<NUM_GPUS>& all2all,
        SuffixArrayPerformanceMeasurements& perf_measure)
        : mcontext(context),
        mmemory_manager(memory_manager),
        mremerge_manager(context, mhost_temp_pinned_allocator),
        mmulti_split(multi_split), mall2all(all2all),
        mperf_measure(perf_measure)
    {
    }

    void init(size_t input_len, size_t per_gpu, size_t last_gpu_len, size_t reserved_len)
    {
        minput_len = input_len;
        misa_divisor = per_gpu;
        mreserved_len = reserved_len;
        mlast_gpu_len = last_gpu_len;
        madditional_temp_storage_size = mmemory_manager.get_additional_pd_space_size();

        mhost_temp_mem = (sa_index_t*)mmemory_manager.get_host_temp_mem().first;
        mhost_temp_mem_size = mmemory_manager.get_host_temp_mem().second;
        mhost_temp_pinned_allocator.init(mhost_temp_mem, mhost_temp_mem_size);

        mwrite_isa_sort_high_bit = std::min(sa_index_t(log2(float(misa_divisor))) + 1,
            sa_index_t(sizeof(sa_index_t) * 8));

        mgpus[0].offset = 0;

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            //uint gpu_index = world_rank();
            // Not sure whether we have to associate the device with host memory planned for it.
            SaGPU& gpu = mgpus[gpu_index];
            gpu.index = gpu_index;
            gpu.num_segments = 0;
            gpu.rank_of_first_entry_within_segment = 0;

            gpu.working_len = per_gpu;

            if (gpu_index + 1 == NUM_GPUS)
            {
                gpu.working_len = last_gpu_len;
            }

            gpu.isa_len = gpu.working_len;

            assign_ptrs(gpu, mmemory_manager.get_pd_arrays(gpu_index));

            if (gpu_index > 0)
            {
                gpu.offset = mgpus[gpu_index - 1].offset + mgpus[gpu_index - 1].working_len;
            }
        }
        mdebugHostGPU.num_segments = 0;
#ifdef ENABLE_DUMPING
        assign_ptrs(mdebugHostGPU, mmemory_manager.get_host_pd_arrays());
#endif
    }

    static void assign_ptrs(SaGPU& gpu, const PDArrays& pd_ptr)
    {
        gpu.Sa_index = pd_ptr.Sa_index;
        gpu.Sa_rank = pd_ptr.Sa_rank;
        gpu.Isa = pd_ptr.Isa;
        gpu.Segment_heads = pd_ptr.Segment_heads;
        gpu.Old_ranks = pd_ptr.Old_ranks;
        gpu.Temp1 = pd_ptr.Temp1;
        gpu.Temp2 = pd_ptr.Temp2;
        gpu.Temp3 = pd_ptr.Temp3;
        gpu.Temp4 = pd_ptr.Temp4;
    }

    void print_stats(size_t iterations) const
    {
        printf("Num elements in prefix-doubling:\n");
        printf("           ");
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            char tmp[32];
            snprintf(tmp, 32, "GPU %d", gpu_index);
            printf("%10s ", tmp);
        }
        printf("\n");
        for (size_t it = 0; it < iterations + 2; ++it)
        {
            if (it == 0)
                printf("Initial         ");
            else if (it == 1)
                printf("Initial compact ");
            else
                printf("Iteration %2zu    ", it - 1);
            for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
            {
                // Not sure whether we have to associate the device with host memory planned for it.
                const SaGPU& gpu = mgpus[gpu_index];
                printf("%10zu ", gpu.stats[it].num_elements);
            }
            printf("\n");
        }
        printf("Num segments in prefix-doubling:\n");
        for (size_t it = 0; it < iterations + 2; ++it)
        {
            if (it == 0)
                printf("Initial         ");
            else if (it == 1)
                printf("Initial compact ");
            else
                printf("Iteration %2zu    ", it - 1);
            for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
            {
                // Not sure whether we have to associate the device with host memory planned for it.
                const SaGPU& gpu = mgpus[gpu_index];
                printf("%10zu ", gpu.stats[it].num_segments);
            }
            printf("\n");
        }
        printf("\n");
    }

    size_t sort(sa_index_t h_initial)
    {
        initial_sort_64();

#ifdef DUMP_EVERYTHING
        dump("After initial sort");
#endif
        //
        mcontext.sync_all_streams();
        printf("[%lu] Initial sort done\n", world_rank());
        comm_world().barrier();
        //

        TIMER_START_MAIN_STAGE(MainStages::Initial_Ranking);
        write_initial_ranks();
        TIMER_STOP_MAIN_STAGE(MainStages::Initial_Ranking);

#ifdef DUMP_EVERYTHING
        dump("Initial ranking");
#endif
        //
        mcontext.sync_all_streams();
        printf("[%lu] Write initial ranks done\n", world_rank());
        comm_world().barrier();
        //
        TIMER_START_MAIN_STAGE(MainStages::Initial_Write_To_ISA);
        write_to_isa(true);
        //
        mcontext.sync_all_streams();
        printf("[%lu] Write to isa done\n", world_rank());
        comm_world().barrier();
        //

        TIMER_STOP_MAIN_STAGE(MainStages::Initial_Write_To_ISA);
#ifdef DUMP_EVERYTHING
        dump("Initial write to ISA");
#endif

        register_numbers(0);

        TIMER_START_MAIN_STAGE(MainStages::Initial_Compacting);
        bool done = false;
        done = compact();
        //
        mcontext.sync_all_streams();
        printf("[%lu] done: %s\n", world_rank(), done ? "true" : "false");
        comm_world().barrier();
        //

        TIMER_STOP_MAIN_STAGE(MainStages::Initial_Compacting);

#ifdef DUMP_EVERYTHING
        dump("Initial compact");
#endif

        sa_index_t h = h_initial;

        mperf_measure.start_loop();

        size_t iterations = 0;

        register_numbers(1);

        while (!done)
        {

            TIMER_START_LOOP_STAGE(LoopStages::Fetch_Rank);
            fetch_rank_for_sorting(h);
            //
            mcontext.sync_all_streams();
            printf("[%lu] iteration: [%lu], fetch rank for sorting done\n", world_rank(), iterations);
            comm_world().barrier();
            //

            TIMER_STOP_LOOP_STAGE(LoopStages::Fetch_Rank);

#ifdef DUMP_EVERYTHING
            dump("After fetch");
#endif

            do_segmented_sort();
            //
            mcontext.sync_all_streams();
            printf("[%lu] iteration: [%lu], do_segmented_sort done\n", world_rank(), iterations);
            comm_world().barrier();
            //
#ifdef DUMP_EVERYTHING
            dump("After sort");
#endif

            TIMER_START_LOOP_STAGE(LoopStages::Rebucket);
            rebucket();
            //
            mcontext.sync_all_streams();
            printf("[%lu] iteration: [%lu], rebucket done\n", world_rank(), iterations);
            comm_world().barrier();
            //


            TIMER_STOP_LOOP_STAGE(LoopStages::Rebucket);

#ifdef DUMP_EVERYTHING
            dump("After rebucket");
#endif

            TIMER_START_LOOP_STAGE(LoopStages::Write_Isa);
            write_to_isa();
            //
            mcontext.sync_all_streams();
            printf("[%lu] iteration: [%lu], write to isa done\n", world_rank(), iterations);
            comm_world().barrier();
            //

            TIMER_STOP_LOOP_STAGE(LoopStages::Write_Isa);

#ifdef DUMP_EVERYTHING
            dump("After write isa");
#endif

            h *= 2;

            TIMER_START_LOOP_STAGE(LoopStages::Compacting);
            done = compact();
            //
            mcontext.sync_all_streams();
            printf("[%lu] iteration: [%lu] compact 2 done\n", world_rank(), iterations);
            comm_world().barrier();
            //

            TIMER_STOP_LOOP_STAGE(LoopStages::Compacting);

            //                if (h > 1024*1024*1024) {
            //                    warning("\nAborting!\n");
            //                    break;
            //                }

#ifdef DUMP_EVERYTHING
            dump("After compact");
#endif

            mperf_measure.next_iteration();
            ++iterations;
            register_numbers(iterations + 1);
        }

        //            TIMER_START_MAIN_STAGE(MainStages::Final_Transpose);
        //            transpose_isa();
        //            TIMER_STOP_MAIN_STAGE(MainStages::Final_Transpose);
        mcontext.sync_all_streams();
        //
        printf("[%lu] prefix doubling done\n", world_rank());
        comm_world().barrier();
        //

        return iterations;
    }

private:
    // Sorting Sa_rank to Old_Ranks, Isa to Sa_index
    void initial_sort()
    {
        TIMER_START_MAIN_STAGE(MainStages::Initial_Sort);

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            cudaSetDevice(mcontext.get_device_id(gpu_index));

            size_t temp_storage_bytes = 0;

            cudaError_t err = cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                gpu.Sa_rank, gpu.Old_ranks, gpu.Isa, gpu.Sa_index,
                gpu.working_len, 0, sizeof(sa_index_t) * 8,
                mcontext.get_gpu_default_stream(gpu.index));

            CUERR_CHECK(err);
            //                if (gpu_index == 0) {
            //                    printf("\nTemp storage required for initial sort: %zu bytes, available: %zu.\n", temp_storage_bytes,
            //                           (3 * mreserved_len + madditional_temp_storage_size)* sizeof(sa_index_t));
            //                }

            ASSERT(temp_storage_bytes <= (3 * mreserved_len + madditional_temp_storage_size) * sizeof(sa_index_t));
            //                temp_storage_bytes = (3 * mreserved_len + madditional_temp_storage_size)* sizeof(sa_index_t);
            err = cub::DeviceRadixSort::SortPairs(gpu.Temp2, temp_storage_bytes,
                gpu.Sa_rank, gpu.Old_ranks, gpu.Isa, gpu.Sa_index,
                gpu.working_len, 0, sizeof(sa_index_t) * 8,
                mcontext.get_gpu_default_stream(gpu.index));
            CUERR_CHECK(err);
            // Now Sa_rank is sorted to Old_ranks,
            // Isa is sorted to Sa_Index
            // Temp2, 3, 4 used as temp space

            //                printf("GPU %u, working len: %zu\n", gpu_index, gpu.working_len);
            mmerge_nodes_info[gpu_index] = { gpu.working_len, gpu.working_len,
                                            gpu_index,
                                            gpu.Old_ranks, gpu.Sa_index,
                                            gpu.Sa_rank, gpu.Isa,
                                            gpu.Temp2, gpu.Temp4 };
            mcontext.get_device_temp_allocator(gpu_index).init(gpu.Temp2, mreserved_len * 3 * sizeof(sa_index_t));
        }
        mremerge_manager.set_node_info(mmerge_nodes_info);
        mcontext.sync_default_streams();
        TIMER_STOP_MAIN_STAGE(MainStages::Initial_Sort);

        TIMER_START_MAIN_STAGE(MainStages::Initial_Merge);

        std::vector<crossGPUReMerge::MergeRange> ranges;
        ranges.push_back({ 0, 0, (sa_index_t)NUM_GPUS - 1, (sa_index_t)mgpus.back().working_len });

        mremerge_manager.merge(ranges, mgpu::less_t<sa_index_t>());
        TIMER_STOP_MAIN_STAGE(MainStages::Initial_Merge);
    }

    // Sorting Sa_rank to Old_Ranks, Isa to Sa_index
    void initial_sort_64()
    {

        const size_t SORT_DOWN_TO = 16;
        const size_t SORT_DOWN_TO_LAST = 13;

        using initial_merge_types = crossGPUReMerge::mergeTypes<uint64_t, sa_index_t>;
        using InitialMergeManager = crossGPUReMerge::ReMergeManager<NUM_GPUS, initial_merge_types, ReMergeTopology>;
        using InitialMergeNodeInfo = crossGPUReMerge::MergeNodeInfo<initial_merge_types>;

        TIMER_START_MAIN_STAGE(MainStages::Initial_Sort);

        InitialMergeManager merge_manager(mcontext, mhost_temp_pinned_allocator);

        std::array<InitialMergeNodeInfo, NUM_GPUS> merge_nodes_info;

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            //uint gpu_index = mcontext.world_rank;
            SaGPU& gpu = mgpus[gpu_index];
            if (world_rank() == gpu_index) {
                cudaSetDevice(mcontext.get_device_id(gpu_index));

                size_t temp_storage_bytes = 0;

                const size_t SORT_DOWN_TO_G = gpu_index == NUM_GPUS - 1 ? SORT_DOWN_TO_LAST : SORT_DOWN_TO;



                cudaError_t err = cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                    reinterpret_cast<uint64_t*>(gpu.Sa_rank),
                    reinterpret_cast<uint64_t*>(gpu.Old_ranks),
                    gpu.Isa, gpu.Sa_index,
                    gpu.working_len, SORT_DOWN_TO_G, sizeof(ulong1) * 8,
                    mcontext.get_gpu_default_stream(gpu.index));

                CUERR_CHECK(err);



                //                if (gpu_index == 0) {
                    //                    printf("\nTemp storage required for initial sort: %zu bytes, available: %zu.\n", temp_storage_bytes,
                    //                           (3 * mreserved_len + madditional_temp_storage_size)* sizeof(sa_index_t));
                    //                }

                ASSERT(temp_storage_bytes <= (3 * mreserved_len + madditional_temp_storage_size) * sizeof(sa_index_t));
                //                temp_storage_bytes = (3 * mreserved_len + madditional_temp_storage_size)* sizeof(sa_index_t);

                //printArray << <1, 1, 0, mcontext.get_gpu_default_stream(gpu.index) >> > (reinterpret_cast<uint64_t*>(gpu.Sa_rank), gpu.Isa, gpu.working_len, true);

                //printArray << <1, 1, 0, mcontext.get_gpu_default_stream(gpu.index) >> > (reinterpret_cast<uint64_t*>(gpu.Old_ranks), gpu.Sa_index, gpu.working_len, false);

                err = cub::DeviceRadixSort::SortPairs(gpu.Temp2, temp_storage_bytes,
                    reinterpret_cast<uint64_t*>(gpu.Sa_rank),
                    reinterpret_cast<uint64_t*>(gpu.Old_ranks),
                    gpu.Isa, gpu.Sa_index,
                    gpu.working_len, SORT_DOWN_TO_G, sizeof(ulong1) * 8,
                    mcontext.get_gpu_default_stream(gpu.index));
                CUERR_CHECK(err);

                //printArray << <1, 1, 0, mcontext.get_gpu_default_stream(gpu.index) >> > (reinterpret_cast<uint64_t*>(gpu.Sa_rank), gpu.Isa, gpu.working_len, true);

                //printArray << <1, 1, 0, mcontext.get_gpu_default_stream(gpu.index) >> > (reinterpret_cast<uint64_t*>(gpu.Old_ranks), gpu.Sa_index, gpu.working_len, false);






                // Now Sa_rank is sorted to Old_ranks,
                // Isa is sorted to Sa_Index
                // Temp2, 3, 4 used as temp space




                    // Span<uint64_t> d(reinterpret_cast<uint64_t*>(gpu.Old_ranks), gpu.working_len);

                    //printf("pointer gpu.Old_ranks: %lu", reinterpret_cast<uint64_t*>(gpu.Old_ranks));

                // std::span<uint64_t> sendBuf(reinterpret_cast<uint64_t*>(gpu.Old_ranks), gpu.working_len);
                // size_t worldSize = comm_world().size();
                // std::vector<int> recCounts(worldSize);
                // for (int i = 0; i < worldSize; i++) {
                //     recCounts.push_back(mgpus[i].working_len);
                // }
                // std::vector<std::span<uint64_t>> recBuf(worldSize);
                // for (int i = 0; i < worldSize; i++) {
                //     std::span<uint64_t> n(reinterpret_cast<uint64_t*>(mgpus[i].Old_ranks), mgpus[i].working_len);
                //     recBuf.push_back(n);
                // }
                // comm_world().allgatherv(send_buf(sendBuf), recv_buf(recBuf), recv_counts(recCounts));
                //printArray << <1, 1, 0, mcontext.get_gpu_default_stream(gpu_index) >> > (gpu.Isa, gpu.Sa_index, gpu.working_len, gpu_index);
                //printArray << <1, 1, 0, mcontext.get_gpu_default_stream(gpu_index) >> > (reinterpret_cast<uint64_t*>(gpu.Sa_rank), reinterpret_cast<uint64_t*>(gpu.Old_ranks), gpu.working_len, gpu_index);
            }


            // printf("Before pointer: %lu\n", reinterpret_cast<uint64_t*>(gpu.Old_ranks));
            // std::span<uint64_t> dd(reinterpret_cast<uint64_t*>(gpu.Old_ranks), gpu.working_len);
            // comm_world().recv(recv_buf(dd), recv_count(gpu.working_len));


            // working_len == num_elements
            merge_nodes_info[gpu_index] = { gpu.working_len, gpu.working_len, gpu_index,
                reinterpret_cast<uint64_t*>(gpu.Old_ranks), gpu.Sa_index,
                reinterpret_cast<uint64_t*>(gpu.Sa_rank), gpu.Isa,
                reinterpret_cast<uint64_t*>(gpu.Temp2), gpu.Temp4 };
            mcontext.get_device_temp_allocator(gpu_index).init(gpu.Temp2, mreserved_len * 3 * sizeof(sa_index_t));

        }

        merge_manager.set_node_info(merge_nodes_info);

        // mcontext.sync_default_streams();

        // for (int i = 0; i < world_size();i++) {
            // all2allDevicePointer(reinterpret_cast<uint64_t*>(mgpus[i].Old_ranks), i, mgpus[i].working_len);
            // if (world_rank() == i) {
            //     std::span<uint64_t> sendBuf(reinterpret_cast<uint64_t*>(mgpus[i].Old_ranks), mgpus[i].working_len);
            //     for (int j = 0; j < world_size(); j++) {
            //         if (i == j) {
            //             continue;
            //         }
            //         comm_world().send(send_buf(sendBuf), send_count(mgpus[i].working_len), tag(i), destination(j));
            //     }
            // }
            // else {
            //     std::span<uint64_t> recB(reinterpret_cast<uint64_t*>(mgpus[i].Old_ranks), mgpus[i].working_len);
            //     comm_world().recv(recv_buf(recB), tag(i), recv_count(mgpus[i].working_len));
            // }
        // }

        mcontext.sync_default_streams();

        TIMER_STOP_MAIN_STAGE(MainStages::Initial_Sort);
        TIMER_START_MAIN_STAGE(MainStages::Initial_Merge);

        std::vector<crossGPUReMerge::MergeRange> ranges;
        ranges.push_back({ 0, 0, (sa_index_t)NUM_GPUS - 1, (sa_index_t)mgpus.back().working_len });

        merge_manager.merge(ranges, mgpu::less_t<uint64_t>());
        //printArray << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (merge_nodes_info[world_rank()].key_buffer, merge_nodes_info[world_rank()].keys, merge_nodes_info[world_rank()].num_elements, world_rank());
        //printArray << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (merge_nodes_info[world_rank()].value_buffer, merge_nodes_info[world_rank()].values, merge_nodes_info[world_rank()].num_elements, world_rank());
        mcontext.sync_default_streams();

        TIMER_STOP_MAIN_STAGE(MainStages::Initial_Merge);
    }

    void write_initial_ranks()
    {
        using rank_t = uint64_t;

        //for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        //{
        uint gpu_index = world_rank();
        SaGPU& gpu = mgpus[gpu_index];
        cudaSetDevice(mcontext.get_device_id(gpu_index));
        //printf("initial\n");
        const rank_t* last_element_prev = nullptr;

        mcontext.get_device_temp_allocator(gpu_index).reset();
        if (world_rank() < world_size() - 1) {
            std::span<rank_t> sb(reinterpret_cast<rank_t*>(gpu.Old_ranks) + gpu.working_len - 1, 1);
            comm_world().send(send_buf(sb), send_count(1), destination(world_rank() + 1));
        }
        if (gpu_index > 0)
        {
            rank_t* temp = mcontext.get_device_temp_allocator(gpu_index).get<rank_t>(1);
            std::span<rank_t> rb(temp, 1);
            comm_world().recv(recv_buf(rb), recv_count(1));

            //  last element of previous gpu
            last_element_prev = temp;//&reinterpret_cast<const rank_t*>(mgpus[gpu_index - 1].Old_ranks)[mgpus[gpu_index - 1].working_len - 1];
        }
        //printf("last element\n");
        kernels::write_ranks_diff_multi _KLC_SIMPLE_(gpu.working_len, mcontext.get_gpu_default_stream(gpu_index))(reinterpret_cast<const rank_t*>(gpu.Old_ranks), last_element_prev, gpu.offset + 1, 0, gpu.Temp1, gpu.working_len);
        CUERR;
        //}
        //printf("write ranks diff multi\n");
        mcontext.sync_default_streams();
        mcontext.get_device_temp_allocator(gpu_index).reset();

        do_max_scan_on_ranks();
    }

    // From Temp1 to Ranks
    void do_max_scan_on_ranks()
    {
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            // uint gpu_index = world_rank();
            SaGPU& gpu = mgpus[gpu_index];
            if (gpu.working_len > 0)
            {
                if (gpu.index == world_rank())
                {

                    cudaSetDevice(mcontext.get_device_id(gpu_index));
                    // Temp1 --> Sa_rank
                    // uses: Temp3, 4

                    MaxFunctor max_op;

                    size_t temp_storage_bytes = 0;
                    //printf("inclusive scan\n");
                    cudaError_t err = cub::DeviceScan::InclusiveScan(nullptr, temp_storage_bytes, gpu.Temp1,
                        gpu.Sa_rank, max_op, gpu.working_len,
                        mcontext.get_gpu_default_stream(gpu_index));
                    CUERR_CHECK(err)

                        // Run inclusive prefix max-scan
                        ASSERT(temp_storage_bytes < 2 * mreserved_len * sizeof(sa_index_t));
                    err = cub::DeviceScan::InclusiveScan(gpu.Temp3, temp_storage_bytes, gpu.Temp1, gpu.Sa_rank,
                        max_op, gpu.working_len, mcontext.get_gpu_default_stream(gpu_index));
                    //printf("done inclusive scan\n");
                    mcontext.sync_default_streams();
                    //printf("inclusive scan, rank: %lu\n", world_rank());
                    CUERR_CHECK(err);
                    cudaMemcpyAsync(mhost_temp_mem + gpu_index, gpu.Sa_rank + gpu.working_len - 1,
                        sizeof(sa_index_t), cudaMemcpyDeviceToHost,
                        mcontext.get_gpu_default_stream(gpu_index));
                    CUERR;
                    mcontext.sync_default_streams();
                    //printf("memcpy, rank: %lu\n", world_rank());

                }


                // Now temp1 is written to Sa_rank
            }
            else
            {
                mhost_temp_mem[gpu_index] = 0;
            }

        }
        mcontext.sync_default_streams();
        //printf("sync streams\n");
        // Send mhost_temp_mem[world_rank()] to all other processes 

        // for (int i = 0; i < world_size(); i++) {
        //     printf("check temp mem: %u, rank: %lu\n", mhost_temp_mem[i], world_rank());
        // }

        std::span<uint32_t> sb(mhost_temp_mem + world_rank(), 1);
        std::span<uint32_t> rb(mhost_temp_mem, world_size());
        comm_world().allgather(send_buf(sb), recv_buf(rb));


        for (uint i = 1; i < NUM_GPUS; ++i)
        {
            mhost_temp_mem[i] = std::max(mhost_temp_mem[i], mhost_temp_mem[i - 1]);
        }

        //for (uint gpu_index = 1; gpu_index < NUM_GPUS; ++gpu_index)
        //{
        if (world_rank() > 0) {
            uint gpu_index = world_rank();
            SaGPU& gpu = mgpus[gpu_index];
            if (gpu.working_len > 0)
            {
                // printf("gpu.working length %lu, rank: %lu\n", gpu.working_len, world_rank());
                cudaSetDevice(mcontext.get_device_id(gpu_index));
                kernels::write_if_eq _KLC_SIMPLE_(gpu.working_len, mcontext.get_gpu_default_stream(gpu_index))(gpu.Sa_rank, gpu.Sa_rank, 0, mhost_temp_mem[gpu_index - 1], gpu.working_len);
                CUERR;
            }
        }

        mcontext.sync_default_streams();
        // printf("after write if eq, rank: %lu\n", world_rank());
        //}
    }

    bool compact()
    {
#ifdef DEBUG_SET_ZERO_TO_SEE_BETTER
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            cudaSetDevice(mcontext.get_device_id(gpu_index));
            cudaMemsetAsync(gpu.Old_ranks, 0, gpu.working_len * sizeof(sa_index_t), mcontext.get_gpu_default_stream(gpu_index));
            cudaMemsetAsync(gpu.Segment_heads, 0, gpu.working_len * sizeof(sa_index_t), mcontext.get_gpu_default_stream(gpu_index));
        }
        mcontext.sync_default_streams();
#endif
        printf("[%lu] before send compact\n", world_rank());

        //printf("[%lu] compacting\n", world_rank());
        //for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            uint gpu_index = world_rank();
            SaGPU& gpu = mgpus[gpu_index];

            // for First_rank_next
            if (gpu_index > 0) {

                // is the next process receiving ?
                if (mgpus[gpu_index - 1].working_len > 0) {
                    std::span<sa_index_t> sb(gpu.Sa_rank, 1);
                    comm_world().isend(send_buf(sb), send_count(1), tag(gpu_index), destination((size_t)gpu_index - 1));
                }
            }

            // for Last_rank_prev
            if (gpu_index < NUM_GPUS - 1) {
                // is the next process receiving ?
                if (mgpus[gpu_index + 1].working_len > 0) {
                    std::span<sa_index_t> sb(gpu.Sa_rank + gpu.working_len - 1, 1);
                    comm_world().isend(send_buf(sb), send_count(1), tag(gpu_index), destination((size_t)gpu_index + 1));
                }
            }

            printf("[%lu] after isend compact\n", world_rank());
            if (gpu.working_len > 0)
            {
                cudaSetDevice(mcontext.get_device_id(gpu_index));
                const sa_index_t* Last_rank_prev;
                const sa_index_t* First_rank_next;

                mcontext.get_device_temp_allocator(gpu_index).reset();
                if (gpu_index < NUM_GPUS - 1) {
                    sa_index_t* temp = mcontext.get_device_temp_allocator(gpu_index).get<sa_index_t>(1);
                    std::span<sa_index_t> rb(temp, 1);
                    comm_world().recv(recv_buf(rb), tag(gpu_index + 1), recv_count(1));
                    First_rank_next = temp;
                }
                else {
                    First_rank_next = nullptr;
                }
                if (gpu_index > 0) {
                    sa_index_t* temp = mcontext.get_device_temp_allocator(gpu_index).get<sa_index_t>(1);
                    std::span<sa_index_t> rb(temp, 1);
                    comm_world().recv(recv_buf(rb), tag(gpu_index - 1), recv_count(1));
                    Last_rank_prev = temp;
                }
                else {
                    Last_rank_prev = nullptr;
                }
                printf("[%lu] after recv compact\n", world_rank());
                //First_rank_next = (gpu_index + 1 < NUM_GPUS) ? mgpus[gpu_index + 1].Sa_rank
                //    : nullptr;
                // pointer to const sa_index_t, last element from next gpu Sa_rank
                //Last_rank_prev = (gpu_index > 0) ? mgpus[gpu_index - 1].Sa_rank + mgpus[gpu_index - 1].working_len - 1
                //    : nullptr;
                kernels::write_compact_flags_multi _KLC_SIMPLE_(gpu.working_len, mcontext.get_gpu_default_stream(gpu_index))(gpu.Sa_rank, Last_rank_prev, First_rank_next, gpu.Temp1, gpu.working_len);
                CUERR;

                size_t temp_storage_bytes = 0;
                cudaError_t err = cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, gpu.Sa_index,
                    gpu.Temp1, gpu.Temp2, gpu.Temp3, gpu.working_len,
                    mcontext.get_gpu_default_stream(gpu_index));
                CUERR_CHECK(err)

                    //            printf("\nTemp storage required for compacting: %zu bytes.\n", temp_storage_bytes);
                    // Run selection
                    ASSERT(temp_storage_bytes < mreserved_len * sizeof(sa_index_t));
                err = cub::DeviceSelect::Flagged(gpu.Temp4, temp_storage_bytes, gpu.Sa_index, gpu.Temp1, gpu.Temp2,
                    gpu.Temp3, gpu.working_len);
                // compact Sa_index -> Temp2 according to flags from Temp1

                CUERR_CHECK(err);

                err = cub::DeviceSelect::Flagged(gpu.Temp4, temp_storage_bytes, gpu.Sa_rank,
                    gpu.Temp1, gpu.Old_ranks, gpu.Temp3, gpu.working_len,
                    mcontext.get_gpu_default_stream(gpu_index));
                // --> compact Sa_rank -> Old_Ranks according to flags from Temp1
                CUERR_CHECK(err);

                cudaMemcpyAsync(mhost_temp_mem + gpu_index, gpu.Temp3, sizeof(sa_index_t), cudaMemcpyDeviceToHost,
                    mcontext.get_gpu_default_stream(gpu_index));
                CUERR;
            }

        }

        mcontext.sync_default_streams();
        //dont need Last_rank_prev, First_rank_next anymore
        mcontext.get_device_temp_allocator(world_rank()).reset();
        //printf("[%lu] after first kernel phase\n", world_rank());
        {
            std::span<sa_index_t> sb(mhost_temp_mem + world_rank(), 1);
            std::span<sa_index_t> rb(mhost_temp_mem, world_size());
            comm_world().allgather(send_buf(sb), recv_buf(rb));
        }
        //printf("[%lu] after first allgather\n", world_rank());

#ifdef DEBUG_SET_ZERO_TO_SEE_BETTER
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            cudaSetDevice(mcontext.get_device_id(gpu_index));
            cudaMemsetAsync(gpu.Sa_index, 0, gpu.working_len * sizeof(sa_index_t), mcontext.get_gpu_default_stream(gpu_index));
            cudaMemsetAsync(gpu.Sa_rank, 0, gpu.working_len * sizeof(sa_index_t), mcontext.get_gpu_default_stream(gpu_index));
        }
        mcontext.sync_default_streams();
#endif
        sa_index_t lens_after_compacting;

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            if (gpu.working_len > 0)
            {


                if (gpu_index == world_rank())
                    lens_after_compacting = mhost_temp_mem[gpu_index];
                //                    printf("\nGPU %u: After compacting, there are %u entries left!", gpu_index,
                //                           lens_after_compacting[gpu_index]);

                if (mhost_temp_mem[gpu_index] == 0)
                {
                    gpu.working_len = 0;
                    gpu.num_segments = 0;
                }
            }
            //printf("[%lu] gpu[%u].working length: %lu\n", world_rank(), gpu_index, gpu.working_len);
        }

        // Ok, now we need to identify segments, these are local.
        NotEqualsFunctor select_op(SA_INDEX_T_MAX);

        bool finished = true;
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            if (gpu.working_len > 0)
            {
                finished = false;
                if (gpu_index == world_rank()) {

                    cudaSetDevice(mcontext.get_device_id(gpu_index));

                    kernels::write_ranks_diff _KLC_SIMPLE_((size_t)lens_after_compacting,
                        mcontext.get_gpu_default_stream(gpu_index))(gpu.Old_ranks, gpu.Temp1, 0, SA_INDEX_T_MAX, lens_after_compacting);
                    CUERR;
                    // Ranks diff --> Temp1

                    //                ASSERT(temp_storage_bytes < reserved_len * sizeof(sa_index_t));
                    size_t temp_storage_bytes = mreserved_len * sizeof(sa_index_t);
                    cudaError_t err = cub::DeviceSelect::If(gpu.Temp4, temp_storage_bytes,
                        gpu.Temp1, gpu.Segment_heads, gpu.Temp3,
                        lens_after_compacting, select_op,
                        mcontext.get_gpu_default_stream(gpu_index));
                    // Write to Segments_heads from Temp1 ...
                    CUERR_CHECK(err);
                    cudaMemcpyAsync(mhost_temp_mem + gpu_index + NUM_GPUS, gpu.Temp3, sizeof(sa_index_t), cudaMemcpyDeviceToHost,
                        mcontext.get_gpu_default_stream(gpu_index));
                    CUERR;

                    // unrelated: copy first and last rank so we can see if there are split-buckets
                    cudaMemcpyAsync(mhost_temp_mem + 2 * NUM_GPUS + gpu_index, gpu.Old_ranks,
                        sizeof(sa_index_t), cudaMemcpyDeviceToHost,
                        mcontext.get_gpu_default_stream(gpu_index));
                    CUERR;
                    printf("671,PrefixDoubling %d", 0);
                    cudaMemcpyAsync(gpu.Sa_index, gpu.Temp2, lens_after_compacting * sizeof(sa_index_t),
                        cudaMemcpyDeviceToDevice, mcontext.get_streams(gpu_index)[1]);
                    CUERR;
                    printf("671,PrefixDoubling %d", 1);
                    if (lens_after_compacting > 0)
                    {
                        cudaMemcpyAsync(mhost_temp_mem + 3 * NUM_GPUS + gpu_index, gpu.Old_ranks + lens_after_compacting - 1,
                            sizeof(sa_index_t), cudaMemcpyDeviceToHost,
                            mcontext.get_gpu_default_stream(gpu_index));
                        CUERR;
                    }
                }
            }
        }

        //printf("[%lu] finished: %s\n", world_rank(), finished ? "true" : "false");

        if (finished)
            return true;

        mcontext.sync_default_streams();
        //printf("[%lu] after second kernel phase\n", world_rank());
        {
            //could be prettier
            std::span<sa_index_t> sb(mhost_temp_mem + world_rank() + NUM_GPUS, 1);
            std::span<sa_index_t> rb(mhost_temp_mem + NUM_GPUS, world_size());
            comm_world().allgather(send_buf(sb), recv_buf(rb));
        }
        {
            std::span<sa_index_t> sb(mhost_temp_mem + world_rank() + 2 * NUM_GPUS, 1);
            std::span<sa_index_t> rb(mhost_temp_mem + 2 * NUM_GPUS, world_size());
            comm_world().allgather(send_buf(sb), recv_buf(rb));
        }
        {
            std::span<sa_index_t> sb(mhost_temp_mem + world_rank() + 3 * NUM_GPUS, 1);
            std::span<sa_index_t> rb(mhost_temp_mem + 3 * NUM_GPUS, world_size());
            comm_world().allgather(send_buf(sb), recv_buf(rb));
        }
        //printf("[%lu] after second allgather\n", world_rank());

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            // Recover lens from above
            gpu.old_rank_start = *(mhost_temp_mem + 2 * NUM_GPUS + gpu_index);
            gpu.old_rank_end = *(mhost_temp_mem + 3 * NUM_GPUS + gpu_index);
            if (gpu.working_len > 0)
            {
                // cudaSetDevice(mcontext.get_device_id(gpu_index));

                gpu.working_len = mhost_temp_mem[gpu_index];
                gpu.num_segments = mhost_temp_mem[gpu_index + NUM_GPUS];

                //                    printf("\nGPU %u has %zu segments left.", gpu_index, gpu.num_segments);

                //                    // Need end segment at the end for CUB segmented sort...
                //                    cudaMemcpyAsync(gpu.Segment_heads + gpu.num_segments, mhost_temp_mem +  NUM_GPUS + gpu_index,
                //                                    sizeof(sa_index_t), cudaMemcpyHostToDevice, mcontext.get_gpu_default_stream(gpu_index)); CUERR;

                ASSERT(gpu.num_segments > 0);

                if (gpu.num_segments > 1)
                {
                    if (gpu_index == world_rank()) {

                        // Copy first segment end
                        cudaMemcpyAsync(mhost_temp_mem + 2 * NUM_GPUS + gpu_index, gpu.Segment_heads + 1,
                            sizeof(sa_index_t), cudaMemcpyDeviceToHost, mcontext.get_gpu_default_stream(gpu_index));
                        CUERR;
                    }
                }
                else
                {
                    *(mhost_temp_mem + 2 * NUM_GPUS + gpu_index) = gpu.working_len;
                }
                if (gpu_index == world_rank()) {
                    // Copy last segment start
                    cudaMemcpyAsync(mhost_temp_mem + 3 * NUM_GPUS + gpu_index, gpu.Segment_heads + gpu.num_segments - 1,
                        sizeof(sa_index_t), cudaMemcpyDeviceToHost, mcontext.get_gpu_default_stream(gpu_index));
                    CUERR;
                }
            }
            //printf("[%lu] gpu[%u].working length: %lu, gpu[%u].num_segments: %lu\n", world_rank(), gpu_index, gpu.working_len, gpu_index, gpu.num_segments);


        }
        mcontext.sync_all_streams();
        //printf("[%lu] after segment update\n", world_rank());
        {
            //could be more efficient if num_segments is checked before sending data
            std::span<sa_index_t> sb(mhost_temp_mem + world_rank() + 2 * NUM_GPUS, 1);
            std::span<sa_index_t> rb(mhost_temp_mem + 2 * NUM_GPUS, world_size());
            comm_world().allgather(send_buf(sb), recv_buf(rb));
        }
        {
            std::span<sa_index_t> sb(mhost_temp_mem + world_rank() + 3 * NUM_GPUS, 1);
            std::span<sa_index_t> rb(mhost_temp_mem + 3 * NUM_GPUS, world_size());
            comm_world().allgather(send_buf(sb), recv_buf(rb));
        }
        //printf("[%lu] after third allgather\n", world_rank());
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            // Recover lens from above
            if (gpu.working_len > 0)
            {
                gpu.first_segment_end = *(mhost_temp_mem + 2 * NUM_GPUS + gpu_index);
                gpu.last_segment_start = *(mhost_temp_mem + 3 * NUM_GPUS + gpu_index);
                //                    printf("\nGPU %u, first segment end: %u, last segment start: %u", gpu_index,
                //                           gpu.first_segment_end, gpu.last_segment_start);
            }
        }

        return false;
    }

    // Sa_rank, Sa_index --> Isa
    void write_to_isa(bool initial = false)
    {
        const sa_index_t sort_threshold = 524288; // empirically, could need adjusting
        const sa_index_t low_bit = 13;
        const sa_index_t high_bit = mwrite_isa_sort_high_bit;

        std::array<MultiSplitNodeInfoT<sa_index_t, sa_index_t, sa_index_t>, NUM_GPUS> multi_split_node_info;
        std::array<All2AllNodeInfoT<sa_index_t, sa_index_t, sa_index_t>, NUM_GPUS> all2all_node_info;
        split_table_tt<sa_index_t, NUM_GPUS> split_table;
        std::array<sa_index_t, NUM_GPUS> dest_lens, src_lens;

        TIMER_START_WRITE_ISA_STAGE(WriteISAStages::Multisplit);

        // Can be initialized upfront.
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            multi_split_node_info[gpu_index].src_keys = gpu.Sa_index;
            multi_split_node_info[gpu_index].src_values = gpu.Sa_rank;
            multi_split_node_info[gpu_index].src_len = gpu.working_len;

            multi_split_node_info[gpu_index].dest_keys = gpu.Temp1;
            multi_split_node_info[gpu_index].dest_values = gpu.Temp2;
            multi_split_node_info[gpu_index].dest_len = gpu.working_len;

            if (gpu_index == world_rank()) {
                //printArray << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (gpu.Sa_index, gpu.Sa_rank, gpu.working_len, gpu_index);
                mcontext.get_device_temp_allocator(gpu_index).init(gpu.Temp3, mreserved_len * 2 * sizeof(sa_index_t));
            }
        }
        printf("[%lu] before multi execKVAsync\n", world_rank());
        PartitioningFunctor<uint> f(misa_divisor, NUM_GPUS - 1);
        mmulti_split.execKVAsync(multi_split_node_info, split_table, src_lens, dest_lens, f);

        mcontext.sync_default_streams();
        printf("[%lu] after multi execKVAsync\n", world_rank());

        TIMER_STOP_WRITE_ISA_STAGE(WriteISAStages::Multisplit);

        //            dump("After split");

        //            print_split_table(split_table, "Write to isa");

        TIMER_START_WRITE_ISA_STAGE(WriteISAStages::All2All);

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            //                fprintf(stderr,"GPU %u, src: %zu, dest: %zu.\n", gpu_index, src_lens[gpu_index], dest_lens[gpu_index]);
            all2all_node_info[gpu_index].src_keys = gpu.Temp1;
            all2all_node_info[gpu_index].src_values = gpu.Temp2;
            all2all_node_info[gpu_index].src_len = gpu.working_len;

            all2all_node_info[gpu_index].dest_keys = gpu.Old_ranks;
            all2all_node_info[gpu_index].dest_values = gpu.Segment_heads;
            all2all_node_info[gpu_index].dest_len = gpu.isa_len;

            all2all_node_info[gpu_index].temp_keys = gpu.Temp3;
            all2all_node_info[gpu_index].temp_values = gpu.Temp4;
            all2all_node_info[gpu_index].temp_len = gpu.isa_len;
        }
        printf("[%lu] before all2all execKVAsync\n", world_rank());
        mall2all.execKVAsync(all2all_node_info, split_table);
        mcontext.sync_all_streams();
        printf("[%lu] after all2all execKVAsync\n", world_rank());

        TIMER_STOP_WRITE_ISA_STAGE(WriteISAStages::All2All);

        TIMER_START_WRITE_ISA_STAGE(WriteISAStages::Sort);

        std::array<std::pair<sa_index_t*, sa_index_t*>, NUM_GPUS> sorted_buff;

        bool sorting = false;
        // printf("[%lu] after all2all execKVAsync\n", world_rank());

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            if (gpu_index == world_rank())
            {
                if (dest_lens[gpu_index] > sort_threshold)
                {
                    sorting = true;
                    cudaSetDevice(mcontext.get_device_id(gpu_index));
                    kernels::sub_value _KLC_SIMPLE_((size_t)dest_lens[gpu_index], mcontext.get_gpu_default_stream(gpu_index))(gpu.Old_ranks, gpu.Temp1, gpu.offset, dest_lens[gpu_index]);
                    CUERR;

                    size_t temp_storage;

                    cub::DoubleBuffer<sa_index_t> d_keys(gpu.Temp1, gpu.Old_ranks);
                    cub::DoubleBuffer<sa_index_t> d_values(gpu.Segment_heads, gpu.Temp2);
                    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage, d_keys, d_values, dest_lens[gpu_index],
                        low_bit, high_bit, mcontext.get_gpu_default_stream(gpu_index));

                    //                    printf("Write to ISA: temp storage: %zu, reserved_len*2: %zu\n", temp_storage, mreserved_len*2);
                    ASSERT(temp_storage <= (mreserved_len * 2 + madditional_temp_storage_size) * sizeof(sa_index_t));

                    cub::DeviceRadixSort::SortPairs(gpu.Temp3, temp_storage, d_keys, d_values, dest_lens[gpu_index],
                        low_bit, high_bit, mcontext.get_gpu_default_stream(gpu_index));
                    sorted_buff[gpu_index].first = d_keys.Current();
                    sorted_buff[gpu_index].second = d_values.Current();
                }
            }
        }
        if (sorting)
        {
            mcontext.sync_default_streams();
        }
        printf("[%lu] isa stage\n", world_rank());
        TIMER_STOP_WRITE_ISA_STAGE(WriteISAStages::Sort);

        TIMER_START_WRITE_ISA_STAGE(WriteISAStages::WriteIsa);
        //printf("after sorting sync, rank %lu\n", world_rank());
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            if (gpu_index == world_rank())
            {
                cudaSetDevice(mcontext.get_device_id(gpu_index));
                if (dest_lens[gpu_index] > sort_threshold)
                {
                    if (initial)
                    {
                        kernels::write_to_isa_2_shared_all<BLOCK_SIZE, 8> _KLC_SIMPLE_ITEMS_PER_THREAD_((size_t)dest_lens[gpu_index], 8, mcontext.get_gpu_default_stream(gpu_index))(sorted_buff[gpu_index].second, sorted_buff[gpu_index].first, gpu.isa_len,
                            gpu.Isa, dest_lens[gpu_index]);
                        CUERR;
                    }
                    else
                    {
                        kernels::write_to_isa_2_shared_most<BLOCK_SIZE, 8> _KLC_SIMPLE_ITEMS_PER_THREAD_((size_t)dest_lens[gpu_index], 8, mcontext.get_gpu_default_stream(gpu_index))(sorted_buff[gpu_index].second, sorted_buff[gpu_index].first, gpu.isa_len,
                            gpu.Isa, dest_lens[gpu_index]);
                        CUERR;

                        //                        kernels::write_to_isa_2 _KLC_SIMPLE_((size_t)dest_lens[gpu_index], mcontext.get_gpu_default_stream(gpu_index))
                        //                                (sorted_buff[gpu_index].second, sorted_buff[gpu_index].first,
                        //                                 gpu.Isa, dest_lens[gpu_index], gpu.isa_len); CUERR;
                    }
                }
                else if (dest_lens[gpu_index] > 0)
                {
                    kernels::write_to_isa_sub_offset _KLC_SIMPLE_((size_t)dest_lens[gpu_index], mcontext.get_gpu_default_stream(gpu_index))(gpu.Segment_heads, gpu.Old_ranks,
                        gpu.Isa, gpu.offset, dest_lens[gpu_index], gpu.isa_len);
                    CUERR;
                }
            }
        }
        mcontext.sync_default_streams();
        TIMER_STOP_WRITE_ISA_STAGE(WriteISAStages::WriteIsa);

    }

    static void transpose_split_table(const split_table_tt<sa_index_t, NUM_GPUS>& split_table_in,
        split_table_tt<sa_index_t, NUM_GPUS>& split_table_out)
    {
        for (uint src_gpu = 0; src_gpu < NUM_GPUS; src_gpu++)
        {
            for (uint dest_gpu = 0; dest_gpu < NUM_GPUS; dest_gpu++)
            {
                split_table_out[src_gpu][dest_gpu] = split_table_in[dest_gpu][src_gpu];
            }
        }
    }

    void fetch_rank_for_sorting(sa_index_t h)
    {
        std::array<MultiSplitNodeInfoT<sa_index_t, sa_index_t, sa_index_t>, NUM_GPUS> multi_split_node_info;
        std::array<All2AllNodeInfoT<sa_index_t, void, sa_index_t>, NUM_GPUS> all2all_node_info;
        split_table_tt<sa_index_t, NUM_GPUS> split_table;
        split_table_tt<sa_index_t, NUM_GPUS> split_table_back;
        std::array<sa_index_t, NUM_GPUS> src_lens, dest_lens;

        TIMER_START_FETCH_RANK_STAGE(FetchRankStages::Prepare_Indices);
        // for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            uint gpu_index = world_rank();
            SaGPU& gpu = mgpus[gpu_index];
            if (gpu.working_len > 0)
            {
                // cudaSetDevice(mcontext.get_device_id(gpu_index));

                kernels::write_sa_index_adding_h _KLC_SIMPLE_(SDIV(gpu.working_len, 2), mcontext.get_gpu_default_stream(gpu_index))(gpu.Sa_index, h, minput_len - 1, gpu.Temp1, gpu.Temp2, gpu.working_len);
                CUERR;
            }
        }

        // Could be initialized upfront.
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            multi_split_node_info[gpu_index].src_keys = gpu.Temp1;
            multi_split_node_info[gpu_index].src_values = gpu.Temp2;
            multi_split_node_info[gpu_index].src_len = gpu.working_len;

            multi_split_node_info[gpu_index].dest_keys = gpu.Temp3;
            multi_split_node_info[gpu_index].dest_values = gpu.Temp4;
            multi_split_node_info[gpu_index].dest_len = gpu.isa_len; // FIXME
            if (world_rank() == gpu_index)
                mcontext.get_device_temp_allocator(gpu_index).init(gpu.Sa_rank, mreserved_len * sizeof(sa_index_t));
        }

        //            PartioningFunctorFilteringZeroes<uint> f(misa_divisor, NUM_GPUS-1);
        PartitioningFunctor<uint> f(misa_divisor, NUM_GPUS - 1);

        mcontext.sync_default_streams(); // NOT NEEDED
        TIMER_STOP_FETCH_RANK_STAGE(FetchRankStages::Prepare_Indices);

        TIMER_START_FETCH_RANK_STAGE(FetchRankStages::Multisplit);
        mmulti_split.execKVAsync(multi_split_node_info, split_table, src_lens, dest_lens, f);

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            all2all_node_info[gpu_index].src_keys = gpu.Temp3;
            all2all_node_info[gpu_index].src_len = (size_t)src_lens[gpu_index];

            all2all_node_info[gpu_index].dest_keys = gpu.Temp1;
            all2all_node_info[gpu_index].dest_len = gpu.isa_len;

            all2all_node_info[gpu_index].temp_keys = gpu.Sa_rank;
            all2all_node_info[gpu_index].temp_len = gpu.isa_len;
        }

        mcontext.sync_default_streams();
        TIMER_STOP_FETCH_RANK_STAGE(FetchRankStages::Multisplit);

        //            print_split_table(split_table);

        TIMER_START_FETCH_RANK_STAGE(FetchRankStages::All2AllForth);
        mall2all.execAsync(all2all_node_info, split_table);
        mcontext.sync_all_streams();
        TIMER_STOP_FETCH_RANK_STAGE(FetchRankStages::All2AllForth);

        //            for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index) {
        //                printf("\nGPU %u: src len: %u, dest len: %u", gpu_index, src_lens[gpu_index], dest_lens[gpu_index]);
        //            }

        TIMER_START_FETCH_RANK_STAGE(FetchRankStages::Fetch);
        //for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            uint gpu_index = world_rank();
            SaGPU& gpu = mgpus[gpu_index];
            if (dest_lens[gpu_index] > 0)
            {
                cudaSetDevice(mcontext.get_device_id(gpu_index));
                kernels::fetch_isa_multi _KLC_SIMPLE_((size_t)dest_lens[gpu_index], mcontext.get_gpu_default_stream(gpu_index))(gpu.Temp1, gpu.Isa, gpu.offset, gpu.Temp3, dest_lens[gpu_index], gpu.isa_len);
                CUERR;
                // Needed, will be synced with all2all back:
                cudaMemsetAsync(gpu.Sa_rank, 0, gpu.working_len * sizeof(sa_index_t), mcontext.get_streams(gpu_index)[1]);
            }
        }
        mcontext.sync_default_streams();
        TIMER_STOP_FETCH_RANK_STAGE(FetchRankStages::Fetch);

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            all2all_node_info[gpu_index].src_keys = gpu.Temp3;
            all2all_node_info[gpu_index].dest_keys = gpu.Temp1;
            all2all_node_info[gpu_index].src_len = gpu.isa_len;
            all2all_node_info[gpu_index].dest_len = gpu.isa_len;

            all2all_node_info[gpu_index].temp_keys = gpu.Sa_rank;
            all2all_node_info[gpu_index].temp_len = gpu.isa_len;
        }

        transpose_split_table(split_table, split_table_back);

        TIMER_START_FETCH_RANK_STAGE(FetchRankStages::All2AllBack);
        mall2all.execAsync(all2all_node_info, split_table_back);
        mcontext.sync_all_streams();
        TIMER_STOP_FETCH_RANK_STAGE(FetchRankStages::All2AllBack);

        TIMER_START_FETCH_RANK_STAGE(FetchRankStages::WriteRanks);
        //for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            uint gpu_index = world_rank();
            SaGPU& gpu = mgpus[gpu_index];
            if (src_lens[gpu_index] > 0)
            {
                cudaSetDevice(mcontext.get_device_id(gpu_index));
                kernels::write_to_rank _KLC_SIMPLE_((size_t)src_lens[gpu_index], mcontext.get_gpu_default_stream(gpu_index))(gpu.Temp1, gpu.Temp4, gpu.Sa_rank, src_lens[gpu_index]);
                CUERR;
            }
        }
        mcontext.sync_default_streams();
        TIMER_STOP_FETCH_RANK_STAGE(FetchRankStages::WriteRanks);
    }

    void print_split_table(const split_table_tt<sa_index_t, NUM_GPUS>& split_table, const char* s = nullptr) const
    {
        std::cout << "MultiSplit table (" << (s ? s : "null") << "):\n";
        for (uint src_gpu = 0; src_gpu < split_table.size(); ++src_gpu)
        {
            for (uint dst_gpu = 0; dst_gpu < split_table[src_gpu].size(); ++dst_gpu)
            {

                std::cout << (dst_gpu == 0 ? "| " : "")
                    << split_table[src_gpu][dst_gpu]
                    << (dst_gpu + 1 == split_table[src_gpu].size() ? " |\n" : " ");
            }
        }
    }

    void dump_numbers(sa_index_t h, const char* str = nullptr)
    {
        if (str)
            printf("\n%s:\n", str);
        else
            printf("\nIteration h=%u:\n", h);

        for (uint g = 0; g < NUM_GPUS; ++g)
        {
            SaGPU& gpu = mgpus[g];
            printf("GPU %u: %zu elements, %zu segments.\n", g, gpu.working_len, gpu.num_segments);
        }
    }

    void register_numbers(sa_index_t iteration)
    {
        for (uint g = 0; g < NUM_GPUS; ++g)
        {
            SaGPU& gpu = mgpus[g];
            gpu.stats[iteration].num_segments = gpu.num_segments;
            gpu.stats[iteration].num_elements = gpu.working_len;
        }
    }

    void do_segmented_sort()
    {
        TIMER_START_LOOP_STAGE(LoopStages::Segmented_Sort);

        // Sort in-place: Sa_rank, Sa_index
        // Uses: Temp2, Temp3, Temp4
        mgpu::less_t<sa_index_t> less;

        //for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            uint gpu_index = world_rank();
            SaGPU& gpu = mgpus[gpu_index];
            if (gpu.working_len > 1)
            {
                cudaSetDevice(mcontext.get_device_id(gpu_index));
                mgpu::my_mpgu_context_t& mgpu_context = mcontext.get_mgpu_default_context_for_device(gpu_index);

                sa_index_t* temp = gpu.Temp1;
                mgpu_context.set_device_temp_mem(temp, sizeof(sa_index_t) * mreserved_len * 4);
                mgpu::segmented_sort(gpu.Sa_rank, gpu.Sa_index, gpu.working_len, gpu.Segment_heads,
                    gpu.num_segments, less, mgpu_context);
            }
        }

        // Now let's plan the merge process.
        std::vector<crossGPUReMerge::MergeRange> ranges;

        sa_index_t segment_rank_offset = 0;
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            mmerge_nodes_info[gpu_index] = { gpu.working_len, gpu.working_len, gpu_index,
                                            gpu.Sa_rank, gpu.Sa_index,
                                            gpu.Temp1, gpu.Temp2,
                                            gpu.Temp3, gpu.Temp4 };
            if (gpu_index == world_rank())
                mcontext.get_device_temp_allocator(gpu_index).init(gpu.Temp3, mreserved_len * 2 * sizeof(sa_index_t));
            gpu.rank_of_first_entry_within_segment = 0;
            if (gpu_index > 0)
            {
                SaGPU& last_gpu = mgpus[gpu_index - 1];
                if (gpu.working_len > 0 && last_gpu.working_len > 0)
                {
                    if (gpu.old_rank_start == last_gpu.old_rank_end)
                    {
                        // Need to merge
                        if (!ranges.empty() && mgpus[ranges.back().end.node].old_rank_start == gpu.old_rank_start)
                        {
                            ranges.back().end.node = gpu.index;
                            ranges.back().end.index = gpu.first_segment_end;
                            segment_rank_offset += last_gpu.working_len - last_gpu.last_segment_start;
                        }
                        else
                        {
                            ranges.push_back({ last_gpu.index, last_gpu.last_segment_start, gpu.index,
                                              gpu.first_segment_end });
                            segment_rank_offset = last_gpu.working_len - last_gpu.last_segment_start;
                        }
                        gpu.rank_of_first_entry_within_segment = segment_rank_offset;
                    }
                }
            }
            //                printf("\nGPU %u, rank of first entry into segment: %u\n", gpu.index, gpu.rank_of_first_entry_within_segment);
        }
        //            for (const crossGPUMerge::MergeRange& r : ranges) {
        //                printf("\nMerge range: From %u, %u to %u, %u\n", r.start.node, r.start.index, r.end.node, r.end.index);
        //            }

        mremerge_manager.set_node_info(mmerge_nodes_info);

        mcontext.sync_default_streams(); // Wait for sorting to finish.
        TIMER_STOP_LOOP_STAGE(LoopStages::Segmented_Sort);

        //            dump("Before merge");
        TIMER_START_LOOP_STAGE(LoopStages::Merge);
        mremerge_manager.merge(ranges, mgpu::less_t<sa_index_t>());
        //            dump("After merge");
        TIMER_STOP_LOOP_STAGE(LoopStages::Merge);
    }

public: // Needs to be public because lamda wouldn't work otherwise...
    void rebucket()
    {
        //for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            uint gpu_index = world_rank();
            cudaSetDevice(mcontext.get_device_id(gpu_index));
            SaGPU& gpu = mgpus[gpu_index];
            if (gpu.working_len > 0)
            {
                sa_index_t* old_ranks = gpu.Old_ranks;
                sa_index_t* inner_segment_ranks = gpu.Sa_rank;
                sa_index_t* output_ranks = gpu.Temp1;
                sa_index_t* temp = gpu.Temp3;
                sa_index_t* Rank_prev_gpu = nullptr;
                sa_index_t rank_of_first_entry_within_segment = gpu.rank_of_first_entry_within_segment;

                // should be mreserved_len * 2 * sizeof(sa_index_t) but 1 extra for Rank_prev_gpu
                mcontext.get_device_temp_allocator(gpu_index).init(temp, mreserved_len * 3 * sizeof(sa_index_t));

                if (gpu_index < NUM_GPUS - 1 && gpu.old_rank_start == mgpus[gpu_index + 1].old_rank_end) {
                    std::span<sa_index_t> sb(gpu.Sa_rank + gpu.working_len - 1, 1);
                    comm_world().isend(send_buf(sb), send_count(1), destination((size_t)gpu_index + 1));
                }

                if (gpu_index > 0 && mgpus[gpu_index - 1].working_len > 0 && gpu.old_rank_start == mgpus[gpu_index - 1].old_rank_end)
                {
                    // maybe we need cudaMalloc here
                    sa_index_t* tempRank = mcontext.get_device_temp_allocator(world_rank()).get<sa_index_t>(1);
                    std::span<sa_index_t> rb(tempRank, 1);
                    comm_world().recv(recv_buf(rb), recv_count(1));
                    Rank_prev_gpu = tempRank;//mgpus[gpu_index - 1].Sa_rank + mgpus[gpu_index - 1].working_len - 1;
                }


                auto my_lambda = [=] __device__(int index, int seg, int index_within_seg)
                {
                    sa_index_t r = inner_segment_ranks[index];
                    sa_index_t seg_rank = old_ranks[index];
                    sa_index_t adjusted_index_within_seg = index_within_seg;

                    if (seg == 0)
                    {
                        adjusted_index_within_seg += rank_of_first_entry_within_segment;
                    }

                    sa_index_t new_rank;
                    if (index_within_seg > 0)
                    {
                        sa_index_t prev_r = inner_segment_ranks[index - 1];
                        if (r != prev_r)
                        {
                            new_rank = seg_rank + adjusted_index_within_seg;
                        }
                        else
                        {
                            new_rank = 0;
                        }
                    }
                    else if (index == 0)
                    {
                        if (Rank_prev_gpu)
                        {
                            sa_index_t prev_r = *Rank_prev_gpu;
                            if (r == prev_r)
                            {
                                new_rank = 0;
                            }
                            else
                            {
                                new_rank = seg_rank + adjusted_index_within_seg;
                            }
                        }
                        else
                        {
                            new_rank = seg_rank;
                        }
                    }
                    else
                    {
                        new_rank = seg_rank;
                    }
                    output_ranks[index] = new_rank;
                };
                mgpu::transform_lbs(my_lambda, gpu.working_len, (int*)gpu.Segment_heads,
                    gpu.num_segments, mcontext.get_mgpu_default_context_for_device(gpu_index));
                CUERR
            }
        }
        mcontext.sync_default_streams();

        do_max_scan_on_ranks();
    }

    void transpose_isa()
    {
        std::array<MultiSplitNodeInfoT<sa_index_t, sa_index_t, sa_index_t>, NUM_GPUS> multi_split_node_info;
        std::array<All2AllNodeInfoT<sa_index_t, sa_index_t, sa_index_t>, NUM_GPUS> all2all_node_info;
        split_table_tt<sa_index_t, NUM_GPUS> split_table;
        std::array<sa_index_t, NUM_GPUS> dest_lens, src_lens;

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            ASSERT(gpu.isa_len > 0);
            cudaSetDevice(mcontext.get_device_id(gpu_index));
            kernels::prepare_isa_transform _KLC_SIMPLE_(SDIV(gpu.isa_len, 2), mcontext.get_gpu_default_stream(gpu_index))(gpu.Isa, gpu.offset, gpu.Temp1, gpu.Temp2, gpu.isa_len);
            CUERR;
        }

        // Can be initialized upfront.
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            multi_split_node_info[gpu_index].src_keys = gpu.Temp1;
            multi_split_node_info[gpu_index].src_values = gpu.Temp2;
            multi_split_node_info[gpu_index].src_len = gpu.isa_len;

            multi_split_node_info[gpu_index].dest_keys = gpu.Temp3;
            multi_split_node_info[gpu_index].dest_values = gpu.Temp4;
            multi_split_node_info[gpu_index].dest_len = gpu.isa_len;
            mcontext.get_device_temp_allocator(gpu_index).init(gpu.Segment_heads,
                mreserved_len * sizeof(sa_index_t));
        }

        PartitioningFunctor<sa_index_t> f(misa_divisor, NUM_GPUS - 1);

        mcontext.sync_default_streams(); // NOT NEEDED

        mmulti_split.execKVAsync(multi_split_node_info, split_table, src_lens, dest_lens, f);

        mcontext.sync_default_streams();

        //            print_split_table(split_table, "transpose isa");

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            //                printf("GPU %u, sr    c: %u, dest: %u.\n", gpu_index, src_lens[gpu_index], dest_lens[gpu_index]);
            all2all_node_info[gpu_index].src_keys = gpu.Temp3;
            all2all_node_info[gpu_index].src_values = gpu.Temp4;
            all2all_node_info[gpu_index].src_len = gpu.isa_len;

            all2all_node_info[gpu_index].dest_keys = gpu.Temp1;
            all2all_node_info[gpu_index].dest_values = gpu.Temp2;
            all2all_node_info[gpu_index].dest_len = gpu.isa_len;

            all2all_node_info[gpu_index].temp_keys = gpu.Segment_heads;
            all2all_node_info[gpu_index].temp_values = gpu.Old_ranks;
            all2all_node_info[gpu_index].temp_len = gpu.isa_len;
        }

        mall2all.execKVAsync(all2all_node_info, split_table);
        mcontext.sync_all_streams();

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            if (dest_lens[gpu_index] > 0)
            {
                cudaSetDevice(mcontext.get_device_id(gpu_index));
                kernels::write_to_isa_sub_offset _KLC_SIMPLE_((size_t)dest_lens[gpu_index], mcontext.get_gpu_default_stream(gpu_index))(gpu.Temp2, gpu.Temp1, gpu.Sa_index, gpu.offset, dest_lens[gpu_index], gpu.isa_len);
                CUERR;
            }
        }
        mcontext.sync_default_streams();
    }

#ifdef ENABLE_DUMPING
    void dump(const char* caption = nullptr)
    {
        if (caption)
        {
            printf("\n%s:\n", caption);
        }
        for (uint g = 0; g < NUM_GPUS; ++g)
        {
            mmemory_manager.copy_down_for_inspection(g);
            printf("\nGPU %u:\nIndex\tSa_index  \tSa_rank   \tTemp1     \tTemp2     \tIsa       \tTemp3     \tTemp4     \tOld_Rank  \tSegment Heads"
                "\tSa_rank as kmer\tOld_rank as kmer\n\n",
                g);
            size_t limit = std::min(misa_divisor, mreserved_len);
            for (int i = 0; i < limit; ++i)
            {
                std::string kmer1 = to_kmer64(reinterpret_cast<uint64_t*>(mdebugHostGPU.Sa_rank)[i]);
                std::string kmer2 = to_kmer64(reinterpret_cast<uint64_t*>(mdebugHostGPU.Old_ranks)[i]);
                size_t l3 = mgpus[g].working_len * 2 - mreserved_len;
                if (i < 100 || i >= limit - 100 || (i > mgpus[g].working_len - 100 && i <= mgpus[g].working_len) || i >= l3 - 100 && i <= l3)
                    printf("%4d.\t%10u\t%10u\t%10u\t%10u\t%10u\t%10u\t%10u\t%10u\t%10u\t%4s\t%4s\n", i, mdebugHostGPU.Sa_index[i], mdebugHostGPU.Sa_rank[i],
                        mdebugHostGPU.Temp1[i], mdebugHostGPU.Temp2[i],
                        mdebugHostGPU.Isa[i], mdebugHostGPU.Temp3[i], mdebugHostGPU.Temp4[i],
                        mdebugHostGPU.Old_ranks[i], mdebugHostGPU.Segment_heads[i],
                        kmer1.c_str(), kmer2.c_str());
            }
        }
    }

    static std::string to_kmer(sa_index_t value)
    {
        char kmer[5];
        kmer[4] = 0;
        sa_index_t v = value & ~3;
        *((sa_index_t*)kmer) = __builtin_bswap32(v);

        if ((value & 3) == 3)
        {
            return std::string(kmer);
        }
        else
        {
            if ((value & 3) == 2)
            {
                return std::string(kmer) + " (before last)";
            }
            else
                return std::string(kmer) + " (last)";
        }
    }

    static std::string to_kmer64(uint64_t value)
    {
        char kmer[9];
        kmer[8] = 0;
        uint64_t v = value & ~(7ull << 13);
        *((sa_index_t*)kmer + 1) = __builtin_bswap32(v & ((1ull << 32ull) - 1ull));
        *((sa_index_t*)kmer) = __builtin_bswap32(v >> 32ull);

        uint64_t ll = (value & (7ull << 13)) >> 13;

        if (ll == 7ull)
        {
            return std::string(kmer);
        }
        else
        {
            if (ll == 3)
            {
                return std::string(kmer) + " (4th last)";
            }
            if (ll == 2)
            {
                return std::string(kmer) + " (3rd last)";
            }
            if (ll == 1)
            {
                return std::string(kmer) + " (2nd last)";
            }
            else
                return std::string(kmer) + " (last)";
        }
    }

    static std::string to_old_kmer(sa_index_t value)
    {
        char kmer[5];
        kmer[4] = 0;
        *((sa_index_t*)kmer) = __builtin_bswap32(value);
        return std::string(kmer);
    }
#endif
};

#endif // PREFIX_DOUBLING_HPP
