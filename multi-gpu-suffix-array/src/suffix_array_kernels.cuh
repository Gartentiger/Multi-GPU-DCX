#ifndef SA_TEST_KERNELS_CUH_
#define SA_TEST_KERNELS_CUH_

#include "suffix_types.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace kernels {

    __global__ void write_ranks_diff(const sa_index_t* Input_data, sa_index_t* Output_rank, sa_index_t base, sa_index_t def_value, size_t N);

    __global__ void write_if_eq(const sa_index_t* Input, sa_index_t* Output, sa_index_t value_neq, sa_index_t write_value, sa_index_t N);

    __global__ void write_compact_flags_multi(const sa_index_t* Ranks, const sa_index_t* Last_rank_prev,
        const sa_index_t* First_rank_next, sa_index_t* flags,
        size_t N);
    __global__ void write_sa_index_adding_h(const sa_index_t* Index, sa_index_t h, sa_index_t max_index,
        sa_index_t* Index_out, sa_index_t* Org_index_out, size_t N);

    __global__ void sub_value(const sa_index_t* values, sa_index_t* Out,
        sa_index_t sub, sa_index_t N);

    __global__ void fetch_isa_multi(const sa_index_t* Sa_idx, const sa_index_t* ISA, sa_index_t base_offset,
        sa_index_t* Rank_out, size_t N, size_t ISA_LEN);

    __global__ void write_to_rank(const sa_index_t* Input_ranks, const sa_index_t* Target_index, sa_index_t* Output_ranks,
        size_t N);

    __global__ void write_to_isa_2(const sa_index_t* Input_ranks, const sa_index_t* Target_index,
        sa_index_t* Isa, sa_index_t N, size_t ISA_N);

    __global__ void write_to_isa_sub_offset(const sa_index_t* Input_ranks, const sa_index_t* Target_index,
        sa_index_t* Isa, sa_index_t Isa_base_offset, sa_index_t N, size_t ISA_N);

    __global__ void prepare_isa_transform(const sa_index_t* Isa, sa_index_t base_offset,
        sa_index_t* Ranks_out, sa_index_t* Index_out, size_t N);

    __global__ void produce_index_kmer_tuples_12(const char* Input, sa_index_t start_index, sa_index_t* Output_index,
        sa_index_t* Output_kmers, size_t N);

    __global__ void produce_index_kmer_tuples_12_64_dc7(const char* Input, sa_index_t start_index, sa_index_t* Output_index,
        ulong1* Output_kmers, size_t N);

    __global__ void produce_index_kmer_tuples_12_64(const char* Input, sa_index_t start_index, sa_index_t* Output_index,
        ulong1* Output_kmers, size_t N);

    __global__ void fixup_last_two_12_kmers(sa_index_t* address);

    __global__ void fixup_last_four_12_kmers_64(ulong1* address);


    __global__ void prepare_S12(const sa_index_t* Isa, const unsigned char* Input,
        const sa_index_t* next_Isa, const unsigned char* next_input,
        sa_index_t node_pd_offset, MergeStageSuffixS12* out, size_t N);

    __global__ void prepare_S12_ind(const sa_index_t* indices, const sa_index_t* Isa, const unsigned char* Input,
        const sa_index_t* next_Isa, const unsigned char* next_Input,
        sa_index_t offset, size_t num_chars,
        MergeStageSuffixS12* out, size_t N);

    __global__ void prepare_S12_ind_kv(const sa_index_t* indices, const sa_index_t* Isa, const unsigned char* Input,
        const sa_index_t* next_Isa, const unsigned char* next_Input,
        sa_index_t offset, size_t num_chars, size_t pd_per_gpu,
        MergeStageSuffixS12HalfKey* out_keys, MergeStageSuffixS12HalfValue* out_values, size_t N);

    __global__ void write_indices(sa_index_t* Out, size_t N);

    __global__ void write_S12_back(const MergeStageSuffixS12* inp, MergeStageSuffix* outp, size_t base_offset, size_t N);

    __global__ void prepare_S0(const sa_index_t* Isa, const unsigned char* Input,
        sa_index_t node_offset, size_t no_chars, size_t Isa_size, bool last,
        MergeStageSuffixS0HalfKey* Out_key,
        MergeStageSuffixS0HalfValue* Out_value, size_t N);

    __global__ void combine_S0_kv(const MergeStageSuffixS0HalfKey* Keys,
        const MergeStageSuffixS0HalfValue* Values,
        MergeStageSuffix* Out, size_t N);

    __global__ void combine_S12_kv(const MergeStageSuffixS12HalfKey* Keys,
        const MergeStageSuffixS12HalfValue* Values,
        MergeStageSuffix* Out, size_t N);

    __global__ void combine_S12_kv_non_coalesced(const MergeStageSuffixS12HalfKey* Keys,
        const MergeStageSuffixS12HalfValue* Values,
        MergeStageSuffix* outp, size_t N);

    __global__ void from_merge_suffix_to_index(const MergeStageSuffix* Merge_suffixes, sa_index_t* Out, size_t N);

}

#endif /* SA_TEST_KERNELS_CUH_ */
