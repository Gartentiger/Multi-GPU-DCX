#ifndef MERGE_TYPES_HPP
#define MERGE_TYPES_HPP

#include <vector>
#include "suffix_types.h"
#include "../gossip/auxiliary.cuh"

namespace crossGPUReMerge {

    template <typename _key_t, typename _value_t>
    struct mergeTypes {
        using key_t = _key_t;
        using value_t = _value_t;
    };


    // All index ranges here are one-past-the-end, the node indices are, however, not one-past-the-end but
    // inclusive.

    struct MergePosition {
        uint node;
        sa_index_t index;
    };

    struct sa_index_range_t {
        sa_index_t start, end;
    };

    struct MergeRange {
        MergePosition start, end;
    };

    struct MergePartitionSource {
        uint node;
        sa_index_range_t range;
        sa_index_t dest_offset;
    };

    struct MergePartition {
        uint dest_node;
        sa_index_range_t dest_range;
        std::vector<MergePartitionSource> sources_1, sources_2;
        size_t size_from_1, size_from_2;
    };

    struct TwoWayPartitioningSearch {
        uint node_1, node_2, search_on;
        sa_index_range_t node1_range;
        sa_index_range_t node2_range;
        sa_index_t cross_diagonal;
        sa_index_t cross_diagonal_index;
        int64_t result;
        int64_t* d_result_ptr;
        int64_t* h_result_ptr;
    };

    // Multi merge
    struct MultiWayPartitioningSearch {
        MultiWayPartitioningSearch(const std::vector<MergeRange>& ranges_)
            : ranges(ranges_)
        {
            results.reserve(ranges.size());
        }

        //    MultiWayPartitioningSearch(const MultiWayPartitioningSearch& other) = delete;

        const std::vector<MergeRange>& ranges;
        size_t split_index;

        std::vector<int64_t> results;
        uint scheduled_on, range_to_take_one_more;
        int64_t* d_result_ptr;
        int64_t* h_result_ptr;
    };

    struct MultiWayMergePartition {
        uint dest_node;
        sa_index_range_t dest_range;
        std::vector<MergePartitionSource> sources;
        std::vector<sa_index_range_t> dest_micro_ranges;
    };

    // /Multi merge

    template <class mtypes>
    struct MergeNodeInfo {
        using key_t = typename mtypes::key_t;
        using value_t = typename mtypes::value_t;
        size_t num_elements; // gpu.working_len
        size_t detour_buffer_size; // gpu.working_len
        uint index; // gpu_index
        key_t* keys; // reinterpret_cast<uint64_t*>(gpu.Old_ranks)
        value_t* values; // gpu.Sa_index
        key_t* key_buffer; // reinterpret_cast<uint64_t*>(gpu.Sa_rank)
        value_t* value_buffer; // gpu.Isa
        key_t* key_detour_buffer; // reinterpret_cast<uint64_t*>(gpu.Temp2)
        value_t* value_detour_buffer; // gpu.Temp4
    };

    struct MergeNodeScheduledWork {
        std::vector<TwoWayPartitioningSearch*> searches;
        std::vector<MergePartition*> merge_partitions;

        std::vector<MultiWayPartitioningSearch*> multi_searches;
        std::vector<MultiWayMergePartition*> multi_merge_partitions;

    };

    template <class mtypes>
    struct MergeNode {
        MergeNode()
            : info{ 0, 0, 0, nullptr, nullptr,nullptr,nullptr,nullptr,nullptr }
        {
        }

        MergeNode(const MergeNodeInfo<mtypes>& info_)
            : info(info_) {
        }

        MergeNodeInfo<mtypes> info;
        MergeNodeScheduledWork scheduled_work;
    };

    using InterNodeCopy = InterNodeCopy<sa_index_t>;

}

#endif // MERGE_TYPES_HPP
