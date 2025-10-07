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
// #include <nvToolsExt.h>
#include "thrust/device_vector.h"
#include "thrust/device_ptr.h"
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

__global__ void printArrayss(const unsigned char* input, size_t size, size_t rank)
{
    for (size_t i = 0; i < size; i++) {

        printf("[%lu] input: %c\n", rank, input[i]);
        // unsigned char* kmerI = reinterpret_cast<*>(kmer[i]);
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

    D_DCX* dcx;

public:
    SuffixSorter(Context& context, size_t len, char* input)
        : mcontext(context), mmemory_manager(context),
        mmulti_split(context), mall2all(context),
        mperf_measure(32),
        mpd_sorter(mcontext, mmemory_manager, mmulti_split, mall2all, mperf_measure),
        minput(input), minput_len(len)
    {
        D_DCX host_dcx;
        memcpy(host_dcx.samplePosition, DCX::samplePosition, sizeof(uint) * DCX::C);
        memcpy(host_dcx.nextSample, DCX::nextSample, sizeof(uint) * DCX::X * DCX::X);
        memcpy(host_dcx.nextNonSample, DCX::nextNonSample, sizeof(uint) * (DCX::X - DCX::C));
        memcpy(host_dcx.inverseSamplePosition, DCX::inverseSamplePosition, sizeof(uint) * DCX::X);
        cudaMalloc(&dcx, sizeof(D_DCX));
        cudaMemcpy(dcx, &host_dcx, sizeof(D_DCX), cudaMemcpyHostToDevice);
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

        // create_tuples();
        TIMER_STOP_MAIN_STAGE(MainStages::Prepare_S12_for_Merge);
        TIMER_START_MAIN_STAGE(MainStages::Prepare_S0_for_Merge);
        // prepare_S0_for_merge();

        TIMER_STOP_MAIN_STAGE(MainStages::Prepare_S0_for_Merge);
        TIMER_START_MAIN_STAGE(MainStages::Final_Merge);

        // final_merge();
        TIMER_STOP_MAIN_STAGE(MainStages::Final_Merge);
        TIMERSTOP(Total);
        mperf_measure.done();

        // TIMER_START_MAIN_STAGE(MainStages::Copy_Results);
        // copy_result_to_host();
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

        ASSERT(minput_len > (NUM_GPUS - 1) * mper_gpu + DCX::X); // Because of merge
        size_t last_gpu_elems = minput_len - (NUM_GPUS - 1) * mper_gpu;
        ASSERT(last_gpu_elems <= mper_gpu); // Because of merge.

        mreserved_len = SDIV(std::max(last_gpu_elems, mper_gpu) + 8, DCX::X * 2) * DCX::X * 2; // Ensure there are 12 elems more space.
        mreserved_len = std::max(mreserved_len, 1024ul);                       // Min len because of temp memory for CUB.

        mpd_reserved_len = SDIV(mreserved_len, DCX::X) * DCX::C;

        ms0_reserved_len = mreserved_len - mpd_reserved_len;

        auto cub_temp_mem = get_needed_cub_temp_memory(ms0_reserved_len, mpd_reserved_len);

        // Can do it this way since CUB temp memory is limited for large inputs.
        ms0_reserved_len = std::max(ms0_reserved_len, SDIV(cub_temp_mem.first, sizeof(MergeSuffixes)));
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

        cub::DoubleBuffer<MergeSuffixes> key(nullptr, nullptr);

        size_t temp_storage_size_S0 = 0;
        size_t temp_storage_size_S12 = 0;
        size_t temp_storage_size_SK = 0;
        cudaError_t err = cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size_S0,
            keys, values, S0_count, 0, 40);
        CUERR_CHECK(err);
        err = cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_size_S12,
            keys, values, S12_count, 0, 40);
        CUERR_CHECK(err);
        err = cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_size_SK,
            key, S0_count + S12_count, index_decomposer{}, 0, sizeof(MergeSuffixes) * 8);
        CUERR_CHECK(err);
        return { temp_storage_size_SK, temp_storage_size_S12 };
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
            kernels::produce_index_kmer_tuples_12_64_dc7 _KLC_SIMPLE_(gpu.num_elements, mcontext.get_gpu_default_stream(gpu_index))((char*)gpu.pd_ptr.Input, gpu.pd_offset, gpu.pd_ptr.Isa, reinterpret_cast<ulong1*>(gpu.pd_ptr.Sa_rank),
                SDIV(gpu.num_elements, DCX::X * 2) * DCX::X * 2);
            CUERR;
        }

        kernels::fixup_last_four_12_kmers_64 << <1, 4, 0, mcontext.get_gpu_default_stream(NUM_GPUS - 1) >> > (reinterpret_cast<ulong1*>(mgpus.back().pd_ptr.Sa_rank) + mgpus.back().pd_elements - 4);
        mcontext.sync_default_streams();
        printf("elements: %lu\n", mgpus[0].num_elements);
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(gpu_index) >> > (reinterpret_cast<char*>(mgpus[gpu_index].pd_ptr.Sa_rank), mgpus[gpu_index].pd_ptr.Isa, SDIV(mgpus[gpu_index].num_elements, DCX::X * 2) * DCX::X * 2, gpu_index);
            mcontext.sync_default_streams();
        }
        // exit(1);
    }


    void prepare_S12_for_merge()
    {
        std::array<MultiSplitNodeInfoT<sa_index_t, sa_index_t, sa_index_t>, NUM_GPUS> multi_split_node_info;
        std::array<All2AllNodeInfoT<MergeSuffixes, char, sa_index_t>, NUM_GPUS> all2all_node_info;
        split_table_tt<sa_index_t, NUM_GPUS> split_table;
        std::array<sa_index_t, NUM_GPUS> dest_lens, src_lens;



        // for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        // {
        //     SaGPU& gpu = mgpus[gpu_index];
        //     printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(0) >> > (gpu.Isa, gpu.Sa_rank, gpu.working_len, 0);
        //     printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(0) >> > (gpu.Sa_index, gpu.Old_ranks, gpu.working_len, 0);
        //     mcontext.sync_default_streams();
        // }


        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Multisplit);
        std::vector<sa_index_t> isaglob;
        isaglob.reserve(minput_len);
        size_t prefixsum = 0;
        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            cudaSetDevice(mcontext.get_device_id(gpu_index));
            // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(gpu_index) >> > (gpu.prepare_S12_ptr.Isa, (sa_index_t*)gpu.prepare_S12_ptr.S12_result, gpu.pd_elements, gpu_index);
            cudaMemcpy(isaglob.data() + prefixsum, gpu.prepare_S12_ptr.Isa, gpu.pd_elements * sizeof(sa_index_t), cudaMemcpyDeviceToHost);
            prefixsum += gpu.pd_elements;
            mcontext.sync_default_streams();
            kernels::write_indices _KLC_SIMPLE_(gpu.pd_elements, mcontext.get_gpu_default_stream(gpu_index))((sa_index_t*)gpu.prepare_S12_ptr.S12_result, gpu.pd_elements);
            CUERR;

            multi_split_node_info[gpu_index].src_keys = gpu.prepare_S12_ptr.Isa;
            // s12_result == sa_index 
            multi_split_node_info[gpu_index].src_values = (sa_index_t*)gpu.prepare_S12_ptr.S12_result;
            multi_split_node_info[gpu_index].src_len = gpu.pd_elements;

            multi_split_node_info[gpu_index].dest_keys = (sa_index_t*)gpu.prepare_S12_ptr.S12_buffer2;
            multi_split_node_info[gpu_index].dest_values = (sa_index_t*)gpu.prepare_S12_ptr.S12_result_half;
            multi_split_node_info[gpu_index].dest_len = gpu.pd_elements;

            mcontext.get_device_temp_allocator(gpu_index).init(gpu.prepare_S12_ptr.S12_buffer1,
                mpd_reserved_len * sizeof(MergeSuffixes));
        }
        isaglob.resize(prefixsum);
        isaglob.shrink_to_fit();
        std::sort(isaglob.begin(), isaglob.end());
        std::vector<sa_index_t> compareIsa(isaglob.size());
        for (size_t i = 0; i < compareIsa.size(); i++)
        {
            compareIsa[i] = i + 1;
        }
        printf("isa[0] %u\n", isaglob[0]);
        printf("isa[1] %u\n", isaglob[1]);
        printf("isa[2] %u\n", isaglob[2]);
        bool ascend = std::equal(compareIsa.begin(), compareIsa.end(), isaglob.begin(), isaglob.end());
        bool containsDuplicates = (std::unique(isaglob.begin(), isaglob.end()) != isaglob.end());
        printf("contains_dup: %s, is_ascending: %s\n", containsDuplicates ? "true" : "false", ascend ? "true" : "false");
        printf("mpd_per_gpu: %lu\n", mpd_per_gpu);
        S12PartitioningFunctor f(mpd_per_gpu, NUM_GPUS - 1);

        // mmulti_split.execKVAsync(multi_split_node_info, split_table, src_lens, dest_lens, f);

        mcontext.sync_default_streams();

        std::array<MergeSuffixes*, NUM_GPUS> skSample;

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];

            // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(gpu_index) >> > ((sa_index_t*)mgpus[gpu_index].prepare_S12_ptr.S12_buffer2, (sa_index_t*)mgpus[gpu_index].prepare_S12_ptr.S12_result_half, mgpus[gpu_index].pd_elements, gpu_index);
            mcontext.sync_default_streams();
            printf("[%u] %lu num elements\n", gpu_index, mgpus[gpu_index].num_elements);
            printf("[%u] %lu pd elements\n", gpu_index, mgpus[gpu_index].pd_elements);
            printf("[%u] %lu offset\n", gpu_index, mgpus[gpu_index].offset);
        }
        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Multisplit);

        TIMER_START_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Out);

        for (uint gpu_index = 0; gpu_index < NUM_GPUS; ++gpu_index)
        {
            SaGPU& gpu = mgpus[gpu_index];
            cudaSetDevice(mcontext.get_device_id(gpu_index));

            const sa_index_t* next_Isa = (gpu_index + 1 < NUM_GPUS) ? mgpus[gpu_index + 1].prepare_S12_ptr.Isa : nullptr;
            const unsigned char* next_Input = (gpu_index + 1 < NUM_GPUS) ? mgpus[gpu_index + 1].prepare_S12_ptr.Input : nullptr;
            MergeSuffixes* sk;
            cudaMalloc(&sk, sizeof(MergeSuffixes) * 2 * gpu.pd_elements);
            skSample[gpu_index] = sk;
            kernels::prepare_SK_ind_kv
                _KLC_SIMPLE_(gpu.pd_elements, mcontext.get_gpu_default_stream(gpu_index))
                // << <1, 1, 0, mcontext.get_gpu_default_stream(gpu_index) >> >

                ((sa_index_t*)gpu.prepare_S12_ptr.S12_result_half,
                    gpu.prepare_S12_ptr.Isa, gpu.prepare_S12_ptr.Input, next_Isa, next_Input, gpu.offset, gpu.num_elements,
                    mpd_per_gpu,
                    (MergeSuffixes*)sk, gpu.pd_elements, dcx);
            CUERR;
            mcontext.sync_all_streams();
            thrust::device_vector<size_t> d_samples_pos(4);
            thrust::device_vector<MergeSuffixes> d_samples;
            d_samples.reserve(4);
            kernels::writeSamples << < 1, 4, 0, mcontext.get_gpu_default_stream(gpu_index) >> > (thrust::raw_pointer_cast(d_samples_pos.data()), sk, thrust::raw_pointer_cast(d_samples.data()), 4ULL);
            // kernels::prepare_S12_ind_kv _KLC_SIMPLE_(gpu.pd_elements, mcontext.get_gpu_default_stream(gpu_index))((sa_index_t*)gpu.prepare_S12_ptr.S12_result_half,
            //     gpu.prepare_S12_ptr.Isa, gpu.prepare_S12_ptr.Input,
            //     next_Isa, next_Input, gpu.offset, gpu.num_elements,
            //     mpd_per_gpu,
            //     gpu.prepare_S12_ptr.S12_buffer1, gpu.prepare_S12_ptr.S12_buffer1_half, gpu.pd_elements);
            // CUERR;

            // printArrayss << <1, 1, 0, mcontext.get_gpu_default_stream(gpu_index) >> > ((MergeSuffixes*)sk, gpu.pd_elements, gpu_index);

        }


        exit(0);
        TIMER_STOP_PREPARE_FINAL_MERGE_STAGE(FinalMergeStages::S12_Write_Into_Place);

        //            dump_prepare_s12("After preparing S12");
        //            dump_final_merge("After preparing S12");
    }


    // template<size_t SAMPLE_COUNT>
    // void SampleSort(MergeSuffixes* sk, MergeSuffixes* output, size_t size, size_t per_gpu) {
    //     std::random_device rd;
    //     std::mt19937 g(rd());
    //     std::uniform_int_distribution<std::mt19937::result_type> randomDist(0, per_gpu - 1);
    //     size_t* sample_pos = (size_t*)malloc(sizeof(size_t) * SAMPLE_COUNT);
    //     size_t* sample_pos_device;
    //     cudaMalloc(&sample_pos_device, sizeof(size_t) * SAMPLE_COUNT);
    //     std::array<MergeSuffixes*, NUM_GPUS> samples;
    //     MergeSuffixes* allSamples;
    //     cudaMalloc(&allSamples, sizeof(MergeSuffixes) * SAMPLE_COUNT * NUM_GPUS);
    //     for (size_t i = 0; i < NUM_GPUS; i++)
    //     {
    //         MergeSuffixes* samples_gpu;
    //         cudaMalloc(&samples_gpu, sizeof(MergeSuffixes) * SAMPLE_COUNT);
    //         samples[i] = samples_gpu;
    //     }


    //     const int a = (int)(16 * log(NUM_GPUS) / log(2.));
    //     for (uint gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)
    //     {
    //         for (size_t i = 0; i < SAMPLE_COUNT; i++)
    //         {
    //             sample_pos[i] = randomDist(g);
    //         }
    //         cudaMemcpy(sample_pos_device, sample_pos, sizeof(size_t) * SAMPLE_COUNT, cudaMemcpyHostToDevice);
    //         // kernels::writeSamples << <1, SAMPLE_COUNT >> > (sample_pos_device, sk[gpu_index], samples[gpu_index]);
    //         cudaFree(sample_pos_device);
    //         cudaMemcpy(allSamples + gpu_index * SAMPLE_COUNT, samples[gpu_index], sizeof(MergeSuffixes) * SAMPLE_COUNT, cudaMemcpyDeviceToDevice);
    //         cudaFree(samples[gpu_index]);
    //     }
    //     free(sample_pos);
    //     size_t temp_storage_size = 0;
    //     cub::DeviceMergeSort::SortKeys(nullptr, temp_storage_size, allSamples, SAMPLE_COUNT * NUM_GPUS, DC7Comparator());
    //     void* temp;
    //     cudaMalloc(&temp, temp_storage_size);
    //     cub::DeviceMergeSort::SortKeys(temp, temp_storage_size, allSamples, SAMPLE_COUNT * NUM_GPUS, DC7Comparator());
    //     selectSplitter << <1, NUM_GPUS - 1 >> > (allSamples, SAMPLE_COUNT);


    //     // kernels::sampleSort<NUM_GPUS> << <1, size, 0, mcontext.get_gpu_default_stream(0) >> > (sk, output, sk, 0, 0, DC7Comparator{});
    // }

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
int main(int argc, char** argv)
{
    if (argc != 4)
    {
        error("Usage: sa-test <ofile> <measureFile> <ifile>!");
    }

    char* input = nullptr;
    cudaSetDevice(0);
    size_t realLen;
    size_t maxLength = size_t(1024 * 1024) * size_t(250 * NUM_GPUS);
    size_t inputLen = read_file_into_host_memory(&input, argv[3], realLen, sizeof(sa_index_t), maxLength, 0);
#ifdef DGX1_TOPOLOGY
    //    const std::array<uint, NUM_GPUS> gpu_ids { 0, 3, 2, 1,  5, 6, 7, 4 };
    //    const std::array<uint, NUM_GPUS> gpu_ids { 1, 2, 3, 0,    4, 7, 6, 5 };
    //    const std::array<uint, NUM_GPUS> gpu_ids { 3, 2, 1, 0,    4, 5, 6, 7 };
    const std::array<uint, NUM_GPUS> gpu_ids{ 3, 2, 1, 0, 4, 7, 6, 5 };

    MultiGPUContext<NUM_GPUS> context(&gpu_ids);
#else 
    const std::array<uint, NUM_GPUS> gpu_ids{ 0,0,0,0 };
    MultiGPUContext<NUM_GPUS> context(&gpu_ids);
#endif
    SuffixSorter sorter(context, realLen, input);
    sorter.alloc();

    sorter.do_sa();
    for (int i = 0; i < realLen; i++) {
        printf("%s\n", input + sorter.get_result()[i]);
    }
    write_array(argv[1], sorter.get_result(), realLen);

    sorter.done();

    sorter.print_pd_stats();
    sorter.get_perf_measurements().print(argv[2]);

    cudaFreeHost(input);
    CUERR;
}
