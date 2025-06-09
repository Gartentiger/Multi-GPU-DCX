#ifndef SUFFIXARRAYMEMORYMANAGER_H
#define SUFFIXARRAYMEMORYMANAGER_H

#include "gossip/context.cuh"
#include "cuda_helpers.h"
#include <array>
#include <cstdint>

#include "suffix_types.h"

// #define ENABLE_DUMPING

struct PDArrays
{
    sa_index_t* Sa_index;
    sa_index_t* Sa_rank;
    sa_index_t* Isa;
    sa_index_t* Segment_heads;
    sa_index_t* Old_ranks;
    sa_index_t* Temp1;
    sa_index_t* Temp2;
    sa_index_t* Temp3;
    sa_index_t* Temp4;
    unsigned char* Input;
};

struct PrepareS12Arrays
{
    sa_index_t* Isa;
    const unsigned char* Input;
    MergeStageSuffix* S12_result;
    MergeStageSuffixS12HalfKey* S12_buffer1;
    MergeStageSuffixS12HalfKey* S12_buffer2;
    MergeStageSuffixS12HalfValue* S12_result_half;
    MergeStageSuffixS12HalfValue* S12_buffer1_half;
    MergeStageSuffixS12HalfValue* S12_buffer2_half;
};

struct PrepareS0Arrays
{
    sa_index_t* Isa;
    const unsigned char* Input;
    MergeStageSuffixS12* S12_result;
    MergeStageSuffixS0* S0_result;
    MergeStageSuffixS0HalfKey* S0_buffer1_keys;
    MergeStageSuffixS0HalfKey* S0_buffer2_keys;
    MergeStageSuffixS0HalfValue* S0_result_2nd_half;
    MergeStageSuffixS0HalfValue* S0_buffer1_values;
    MergeStageSuffixS0HalfValue* S0_buffer2_values;
};

struct MergeS12S0Arrays
{
    MergeStageSuffix* S12_result;
    MergeStageSuffix* S0_result;
    union
    {
        MergeStageSuffix* buffer;
        sa_index_t* result;
    };
    void* remaining_storage;
    size_t remaining_storage_size;
};

// Class for managing memory for suffix array creation.
template <uint NUM_GPUS, typename value_t>
class SuffixArrayMemoryManager
{
public:
    static const size_t ALIGN_BYTES = 256;
    static const size_t NUM_PD_ARRAYS = 9;
    static const size_t HOST_TEMP_MEM_SIZE = 1024 * NUM_GPUS;

    static const size_t HALF_MERGE_STAGE_SUFFIX_SIZE = sizeof(MergeStageSuffix) / 2;

    static_assert(sizeof(MergeStageSuffixS12) == sizeof(MergeStageSuffixS0),
        "The sizes of the structs for the merge stage have to match!");

    static_assert(HALF_MERGE_STAGE_SUFFIX_SIZE % 2 == 0,
        "Expect size of merge suffix to be even.");

    static_assert(HALF_MERGE_STAGE_SUFFIX_SIZE == 8,
        "Expect half a merge stage suffix to fit within 64 bits.");
    static_assert(sizeof(MergeStageSuffixS0HalfKey) == HALF_MERGE_STAGE_SUFFIX_SIZE &&
        sizeof(MergeStageSuffixS0HalfValue) == HALF_MERGE_STAGE_SUFFIX_SIZE,
        "Half merge suffix sizes should be one half of merge suffixes.");

    SuffixArrayMemoryManager(const SuffixArrayMemoryManager&) = delete;
    SuffixArrayMemoryManager& operator=(const SuffixArrayMemoryManager&) = delete;

    SuffixArrayMemoryManager(MultiGPUContext<NUM_GPUS>& context_)
        : mcontext(context_)
    {
    }

    const sa_index_t* get_h_result() const
    {
        return mh_result;
    }

    sa_index_t* get_h_result()
    {
        return mh_result;
    }

    size_t get_additional_pd_space_size() const
    {
        return madditional_pd_space_size;
    }

    const PDArrays& get_pd_arrays(uint node) const
    {
        ASSERT(node < NUM_GPUS);
        return marrays_pd[node];
    }

    const PrepareS12Arrays& get_prepare_S12_arrays(uint node) const
    {
        ASSERT(node < NUM_GPUS);
        return marrays_prepare_S12[node];
    }

    const PrepareS0Arrays& get_prepare_S0_arrays(uint node) const
    {
        ASSERT(node < NUM_GPUS);
        return marrays_prepare_S0[node];
    }

    const MergeS12S0Arrays& get_merge_S12_S0_arrays(uint node) const
    {
        ASSERT(node < NUM_GPUS);
        return marrays_merge_S12_S0[node];
    }

#ifdef ENABLE_DUMPING
    const PDArrays& get_host_pd_arrays() const
    {
        return marrays_pd[NUM_GPUS];
    }

    const PrepareS12Arrays& get_host_prepare_S12_arrays() const
    {
        return marrays_prepare_S12[NUM_GPUS];
    }

    const PrepareS0Arrays& get_host_prepare_S0_arrays() const
    {
        return marrays_prepare_S0[NUM_GPUS];
    }

    const MergeS12S0Arrays& get_host_merge_S12_S0_arrays() const
    {
        return marrays_merge_S12_S0[NUM_GPUS];
    }
#endif

    std::pair<void*, size_t> get_host_temp_mem() const
    {
        return std::make_pair(mhost_temp_mem, HOST_TEMP_MEM_SIZE);
    }

    void alloc(size_t input_len, size_t min_gpu_len, size_t min_pd_len, size_t min_S0_len, bool zero)
    {

        mpd_array_aligned_len = align_len(min_pd_len, sizeof(sa_index_t));
        minput_aligned_len = align_len(min_gpu_len, 1);

        mhalf_merge_suffix_s12_aligned_len = align_len(min_pd_len, HALF_MERGE_STAGE_SUFFIX_SIZE);
        mhalf_merge_suffix_s0_aligned_len = align_len(min_S0_len, HALF_MERGE_STAGE_SUFFIX_SIZE);

        size_t two_halves_s12_len = SDIV(2 * mhalf_merge_suffix_s12_aligned_len * HALF_MERGE_STAGE_SUFFIX_SIZE, sizeof(MergeStageSuffix));
        size_t two_halves_s0_len = SDIV(2 * mhalf_merge_suffix_s0_aligned_len * HALF_MERGE_STAGE_SUFFIX_SIZE, sizeof(MergeStageSuffix));

        // Pick these so that we can also fit half-buffers into one whole buffer.
        mmerge_suffix_s12_aligned_len = align_len(two_halves_s12_len, sizeof(MergeStageSuffix));
        mmerge_suffix_s0_aligned_len = align_len(two_halves_s0_len, sizeof(MergeStageSuffix));

        size_t pd_total_bytes = NUM_PD_ARRAYS * mpd_array_aligned_len * sizeof(sa_index_t) + minput_aligned_len;

        size_t prepareS12_total_bytes = mpd_array_aligned_len * sizeof(sa_index_t) + minput_aligned_len +
            3 * sizeof(MergeStageSuffix) * mmerge_suffix_s12_aligned_len;

        size_t prepareS0_total_bytes = mpd_array_aligned_len * sizeof(sa_index_t) + minput_aligned_len +
            sizeof(MergeStageSuffix) * mmerge_suffix_s12_aligned_len +
            3 * sizeof(MergeStageSuffix) * mmerge_suffix_s0_aligned_len;

        size_t merge_total_bytes = 2 * (mmerge_suffix_s12_aligned_len + mmerge_suffix_s0_aligned_len) * sizeof(MergeStageSuffix);

        malloc_size = std::max(std::max(std::max(prepareS0_total_bytes, prepareS12_total_bytes), pd_total_bytes), merge_total_bytes);

        printf("Allocating %zu K per node (%zu K for prefix doubling, %zu K for prepare_S_12, %zu K for prepare_S_0, "
            "%zu K for final merge).\n",
            malloc_size / 1024, pd_total_bytes / 1024, prepareS12_total_bytes / 1024,
            prepareS0_total_bytes / 1024, merge_total_bytes / 1024);

        // Place this at the end.
        minput_offset = align_down(malloc_size - minput_aligned_len);
        // Place this one before the end.
        misa_offset = align_down(minput_offset - mpd_array_aligned_len * sizeof(sa_index_t));

        madditional_pd_space_size = (misa_offset - 8 * mpd_array_aligned_len * sizeof(sa_index_t));

        for (uint gpu = 0; gpu < NUM_GPUS; ++gpu)
        {
            // if (world_rank() == gpu)
            // {
            cudaSetDevice(mcontext.get_device_id(gpu));
            cudaMalloc(&malloc_base[gpu], malloc_size);
            CUERR;
            // cudaSetDevice(mcontext.get_device_id(gpu));
            // cudaMalloc(&malloc_base[gpu], malloc_size);
            if (zero)
            {
                cudaMemsetAsync(malloc_base[gpu], 0, malloc_size,
                    mcontext.get_gpu_default_stream(gpu));
                CUERR;
            }
            marrays_pd[gpu] = make_pd_arrays(malloc_base[gpu]);
            marrays_prepare_S12[gpu] = make_prepare_S12_arrays(malloc_base[gpu]);
            marrays_prepare_S0[gpu] = make_prepare_S0_arrays(malloc_base[gpu]);
            marrays_merge_S12_S0[gpu] = make_merge_S12_S0_arrays(malloc_base[gpu]);
            //}
        }
        if (world_rank() == NUM_GPUS - 1)
        {
            cudaMallocHost(&mh_result, input_len * sizeof(sa_index_t));
            CUERR;
            cudaMallocHost(&mhost_temp_mem, HOST_TEMP_MEM_SIZE);
            CUERR;
        }

#ifdef ENABLE_DUMPING
        // Debugging:
        cudaMallocHost(&mhost_alloc_base, malloc_size);
        CUERR;

        marrays_pd[NUM_GPUS] = make_pd_arrays(mhost_alloc_base);
        marrays_prepare_S12[NUM_GPUS] = make_prepare_S12_arrays(mhost_alloc_base);
        marrays_prepare_S0[NUM_GPUS] = make_prepare_S0_arrays(mhost_alloc_base);
        marrays_merge_S12_S0[NUM_GPUS] = make_merge_S12_S0_arrays(mhost_alloc_base);
#endif
    }

    void free()
    {
        for (uint gpu = 0; gpu < NUM_GPUS; ++gpu)
        {
            if (gpu == world_rank())
            {

                cudaSetDevice(mcontext.get_device_id(gpu));
                cudaFree(malloc_base[gpu]);
            }
        }
        if (world_rank() == NUM_GPUS - 1)
        {
            cudaFreeHost(mhost_temp_mem);
            cudaFreeHost(mh_result);
        }
#ifdef ENABLE_DUMPING
        cudaFreeHost(mhost_alloc_base);
#endif
    }

#ifdef ENABLE_DUMPING
    void copy_down_for_inspection(uint gpu)
    {
        cudaSetDevice(mcontext.get_device_id(gpu));
        cudaMemcpy(mhost_alloc_base, malloc_base[gpu], malloc_size, cudaMemcpyDeviceToHost);
    }
#endif

private:
    MultiGPUContext<NUM_GPUS>& mcontext;

    size_t mpd_array_aligned_len, minput_aligned_len, mmerge_suffix_s12_aligned_len, mmerge_suffix_s0_aligned_len,
        mhalf_merge_suffix_s12_aligned_len, mhalf_merge_suffix_s0_aligned_len;
    size_t madditional_pd_space_size;
    size_t malloc_size, minput_offset, misa_offset;
    std::array<unsigned char*, NUM_GPUS> malloc_base;

    std::array<PDArrays, NUM_GPUS + 1> marrays_pd;
    std::array<PrepareS12Arrays, NUM_GPUS + 1> marrays_prepare_S12;
    std::array<PrepareS0Arrays, NUM_GPUS + 1> marrays_prepare_S0;
    std::array<MergeS12S0Arrays, NUM_GPUS + 1> marrays_merge_S12_S0;

#ifdef ENABLE_DUMPING
    unsigned char* mhost_alloc_base;
#endif
    unsigned char* mhost_temp_mem;
    sa_index_t* mh_result;

    PDArrays make_pd_arrays(unsigned char* base) const
    {
        ASSERT(8 * mpd_array_aligned_len * sizeof(sa_index_t) < misa_offset);

        PDArrays arr;

        arr.Isa = (sa_index_t*)(base + misa_offset);
        arr.Input = (base + minput_offset);
        arr.Sa_index = (sa_index_t*)(base);
        arr.Old_ranks = (sa_index_t*)(base + 1 * mpd_array_aligned_len * sizeof(sa_index_t));
        arr.Segment_heads = (sa_index_t*)(base + 2 * mpd_array_aligned_len * sizeof(sa_index_t));
        arr.Sa_rank = (sa_index_t*)(base + 3 * mpd_array_aligned_len * sizeof(sa_index_t));
        arr.Temp1 = (sa_index_t*)(base + 4 * mpd_array_aligned_len * sizeof(sa_index_t));
        arr.Temp2 = (sa_index_t*)(base + 5 * mpd_array_aligned_len * sizeof(sa_index_t));
        arr.Temp3 = (sa_index_t*)(base + 6 * mpd_array_aligned_len * sizeof(sa_index_t));
        arr.Temp4 = (sa_index_t*)(base + 7 * mpd_array_aligned_len * sizeof(sa_index_t));
        return arr;
    }

    PrepareS12Arrays make_prepare_S12_arrays(unsigned char* base) const
    {
        PrepareS12Arrays arr;
        ASSERT(3 * sizeof(MergeStageSuffix) * mmerge_suffix_s12_aligned_len <= misa_offset);

        arr.Isa = (sa_index_t*)(base + misa_offset);
        arr.Input = (base + minput_offset);
        arr.S12_result = (MergeStageSuffix*)base;
        arr.S12_buffer1 = (MergeStageSuffixS12HalfKey*)(base + sizeof(MergeStageSuffix) * mmerge_suffix_s12_aligned_len);
        arr.S12_buffer2 = (MergeStageSuffixS12HalfKey*)(base + 2 * sizeof(MergeStageSuffix) * mmerge_suffix_s12_aligned_len);
        size_t half_offset = mhalf_merge_suffix_s12_aligned_len * HALF_MERGE_STAGE_SUFFIX_SIZE;
        arr.S12_result_half = (MergeStageSuffixS12HalfValue*)(base + half_offset);
        arr.S12_buffer1_half = (MergeStageSuffixS12HalfValue*)(reinterpret_cast<unsigned char*>(arr.S12_buffer1) + half_offset);
        arr.S12_buffer2_half = (MergeStageSuffixS12HalfValue*)(reinterpret_cast<unsigned char*>(arr.S12_buffer2) + half_offset);
        return arr;
    }

    PrepareS0Arrays make_prepare_S0_arrays(unsigned char* base) const
    {
        PrepareS0Arrays arr;
        ASSERT(sizeof(MergeStageSuffix) * mmerge_suffix_s12_aligned_len + 3 * sizeof(MergeStageSuffix) * mmerge_suffix_s0_aligned_len <= misa_offset);

        unsigned char* S0_base = base + sizeof(MergeStageSuffix) * mmerge_suffix_s12_aligned_len;

        arr.Isa = (sa_index_t*)(base + misa_offset);
        arr.Input = (base + minput_offset);
        arr.S12_result = (MergeStageSuffixS12*)base;
        arr.S0_result = (MergeStageSuffix*)(S0_base);
        arr.S0_buffer1_keys = (MergeStageSuffixS0HalfKey*)(S0_base + 1 * sizeof(MergeStageSuffix) * mmerge_suffix_s0_aligned_len);
        arr.S0_buffer2_keys = (MergeStageSuffixS0HalfKey*)(S0_base + 2 * sizeof(MergeStageSuffix) * mmerge_suffix_s0_aligned_len);

        size_t half_offset = mhalf_merge_suffix_s0_aligned_len * HALF_MERGE_STAGE_SUFFIX_SIZE;

        arr.S0_result_2nd_half = (MergeStageSuffixS0HalfValue*)(S0_base + half_offset);
        arr.S0_buffer1_values = (MergeStageSuffixS0HalfValue*)(reinterpret_cast<unsigned char*>(arr.S0_buffer1_keys) + half_offset);
        arr.S0_buffer2_values = (MergeStageSuffixS0HalfValue*)(reinterpret_cast<unsigned char*>(arr.S0_buffer2_keys) + half_offset);

        return arr;
    }

    MergeS12S0Arrays make_merge_S12_S0_arrays(unsigned char* base) const
    {
        MergeS12S0Arrays arr;
        size_t total_bytes = 2 * (mmerge_suffix_s12_aligned_len + mmerge_suffix_s0_aligned_len) * sizeof(MergeStageSuffix);
        ASSERT(total_bytes <= malloc_size);

        arr.S12_result = (MergeStageSuffix*)base;
        arr.S0_result = (MergeStageSuffix*)(base + sizeof(MergeStageSuffix) * mmerge_suffix_s12_aligned_len);
        arr.buffer = (MergeStageSuffix*)(base + sizeof(MergeStageSuffix) * (mmerge_suffix_s12_aligned_len + mmerge_suffix_s0_aligned_len));
        arr.remaining_storage = base + total_bytes;
        arr.remaining_storage_size = malloc_size - total_bytes;
        return arr;
    }

    static inline size_t align_len(size_t no_elems, size_t elem_size)
    {
        ASSERT(ALIGN_BYTES % elem_size == 0);
        const size_t align = ALIGN_BYTES / elem_size;
        return SDIV(no_elems, align) * align;
    }

    static inline size_t align_down(size_t offset)
    {
        return (offset / ALIGN_BYTES) * ALIGN_BYTES;
    }
};

#endif // SUFFIXARRAYMEMORYMANAGER_H
