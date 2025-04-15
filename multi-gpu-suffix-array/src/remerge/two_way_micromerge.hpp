#ifndef TWO_WAY_MICROMERGE_HPP
#define TWO_WAY_MICROMERGE_HPP

#include <iostream>

#include "merge_types.hpp"
#include "mergenodeutils.hpp"

namespace crossGPUReMerge {

/** A 2 way merge between 2 sequences the keys and values of which may be distributed across several GPUs. */
class TwoWayMicroMerge {
    public:

        TwoWayMicroMerge(const MergeNodeUtils& node_utils, uint first_range, uint second_range)
            : mnode_utils(node_utils), mfirst_range(first_range), msecond_range(second_range)
        {}

        template <class MergeTopologyHelper>
        void schedule_searches_for_micro_merge(const MergeRange& r1, const MergeRange& r2,
                                               MergeTopologyHelper& topology_helper) {

            mcross_diagonal_searches.clear();

            size_t num_cross_diagonals = (r1.end.node - r1.start.node + 1) +
                                         (r2.end.node - r2.start.node + 1) - 1;

            mcross_diagonal_searches.resize(num_cross_diagonals);

            bool alternate = false;

            for (uint node1 = r1.start.node; node1 <= r1.end.node; ++node1) {
                for (uint node2 = r2.start.node; node2 <= r2.end.node; ++node2) {
                    ASSERT(node1 < node2);
                    sa_index_t start_1 = node1 == r1.start.node ? r1.start.index : 0;
                    sa_index_t end_1 = node1 == r1.end.node ? r1.end.index : get_node_num_elements(node1);
                    sa_index_t start_2 = node2 == r2.start.node ? r2.start.index : 0;
                    sa_index_t end_2 = node2 == r2.end.node ? r2.end.index : get_node_num_elements(node2);

                    sa_index_t size_1 = end_1 - start_1;
//                    sa_index_t size_2 = end_2 - start_2;

                    // Note that it will only work this way if only the first sequence of r1 and the last of
                    // r2 are not going over complete nodes, i.e. we need squares everywhere except for the
                    // first column and last row in the Merge-Path picture, if we draw r1 right and r2
                    // down. This is the case normally, as we merge over contiguous sequences, so all elements
                    // in the middle will be "full width".

//                    uint scheduled_on = decide_search_location(node1, node2, size_1, size_2);
//                    uint scheduled_on = alternate ? node2 : node1;
                    uint scheduled_on = topology_helper.get_node_to_schedule_two_way_micro_merge_search_on(node1, node2,
                                                                                                     alternate);
                    // node1 + node2 gives the cross-diagonal index we participate in
                    uint cross_diagonal_index = node1-r1.start.node + node2-r2.start.node;

                    mcross_diagonal_searches.at(cross_diagonal_index).push_back({ node1, node2, scheduled_on,
                                                                                { start_1, end_1 }, { start_2, end_2},
                                                                                 size_1, cross_diagonal_index,
                                                                                 -1});
                    alternate = !alternate;
                }
                alternate = !alternate;
            }
        }

        std::vector<std::pair<MergePosition, MergePosition>> create_partition_points_from_search_results() const {

            std::vector<std::pair<MergePosition, MergePosition>> partition_points;
            partition_points.reserve(mcross_diagonal_searches.size());

            for (const auto& searches : mcross_diagonal_searches) {
                const TwoWayPartitioningSearch* hit_search = nullptr;
                ASSERT(!searches.empty());
                for (uint search_idx = 0; search_idx < searches.size(); ++search_idx) {
                    const auto& search = searches[search_idx];
                    const TwoWayPartitioningSearch* next_search = (search_idx + 1 < searches.size()) ?
                                                &searches[search_idx+1] : nullptr;

                    // If there are several searches per cross-diagonal, take the one producing the actual
                    // result.
                    if ((search_idx == 0 && search.result == 0) ||
                        (search.result > 0 && search.result < search.cross_diagonal) ||
                        (search.result  == search.cross_diagonal && (next_search == nullptr || next_search->result == 0))) {
                        hit_search = &search;
                        break;
                    }

                }

                ASSERT(hit_search);

                MergePosition a { hit_search->node_1, hit_search->node1_range.start + (sa_index_t)hit_search->result };

                // Note this -1 here might wrap around the unsigned index type, but this will reversed in
                // safe_increment with another wrap-around anyway... :/
                MergePosition b { hit_search->node_2, hit_search->node2_range.start +
                                                      hit_search->cross_diagonal - (sa_index_t)hit_search->result - 1 };
                partition_points.push_back({ a, b });
            }
            return partition_points;
        }

        void output_partition(const MergeRange& r1, const MergeRange& r2,
                              const std::pair<uint, sa_index_range_t>& dest_range) {
            mpartitions.emplace_back(MergePartition({ dest_range.first, dest_range.second}));
            MergePartition& part = mpartitions.back();
            part.sources_1.reserve(r1.end.node - r1.start.node + 1);
            part.sources_2.reserve(r2.end.node - r2.start.node + 1);

            part.size_from_1 = part.size_from_2 = 0;

            mnode_utils.convert_to_per_node_ranges(r1, [&part] (uint node, sa_index_range_t r) {
                size_t size = r.end - r.start;
                if (size > 0) {
                    part.sources_1.push_back({ node, r, (sa_index_t) part.size_from_1 });
                    part.size_from_1 += size;
                }
            });

            mnode_utils.convert_to_per_node_ranges(r2, [&part] (uint node, sa_index_range_t r) {
                size_t size = r.end - r.start;
                if (size > 0) {
                    part.sources_2.push_back({ node, r, (sa_index_t) part.size_from_2 });
                    part.size_from_2 += size;
                }
            });
        }

        std::vector<std::pair<uint, sa_index_range_t>> get_dest_ranges(const MergeRange& input1,
                                                                       const MergeRange& input2) {
            std::vector<std::pair<uint, sa_index_range_t>> dest_ranges;
            dest_ranges.reserve(input1.end.node - input1.start.node + input2.end.node - input2.start.node + 2);

            mnode_utils.convert_to_per_node_ranges(mnode_utils.combine_ranges(input1, input2),
                                                   [&dest_ranges] (uint node, sa_index_range_t r) {
                dest_ranges.push_back({node, r});
            });;

            return dest_ranges;
        }

        void create_partitions_from_search_results(const MergeRange& input1,
                                                   const MergeRange& input2) {
            const std::vector<std::pair<MergePosition, MergePosition>> partition_points =
                    create_partition_points_from_search_results();
//            mpartition_points = partition_points; // DEBUG

            const std::vector<std::pair<uint, sa_index_range_t>> dest_ranges = get_dest_ranges(input1, input2);
            ASSERT(dest_ranges.size() == partition_points.size()+1);

            MergeRange r1, r2;

            r1.start = input1.start;
            r2.start = input2.start;

            uint dest_range_idx = 0;

            for (const std::pair<MergePosition, MergePosition>& split_point : partition_points) {
                MergePosition split2 = mnode_utils.safe_increment(split_point.second, input2);
                r1.end = split_point.first;
                r2.end = split2;

                output_partition(r1, r2, dest_ranges[dest_range_idx++]);

                r1.start = split_point.first;

                r2.start = split2;
            }

            r1.end = input1.end;
            r2.end = input2.end;
            output_partition(r1, r2, dest_ranges[dest_range_idx++]);
        }

        const std::vector<std::vector<TwoWayPartitioningSearch>>& cross_diagonal_searches() const { return mcross_diagonal_searches; }
        std::vector<std::vector<TwoWayPartitioningSearch>>& cross_diagonal_searches() { return mcross_diagonal_searches; }

        uint first_range() const { return mfirst_range; }
        uint second_range() const { return msecond_range; }

        const std::vector<MergePartition>& partitions() const { return mpartitions; }
        std::vector<MergePartition>& partitions() { return mpartitions; }

        void debug_print() const {
            auto& log = std::cerr;
            log << "\n\nRanges: " << first_range() << ", " << second_range() << "\n";
            uint cdidx = 0;
            for (const auto& cds : mcross_diagonal_searches) {
                log << "\nSearches over cross diagonal " << cdidx << ":\n";
                for (const TwoWayPartitioningSearch& ps : cds) {
                    log << "Nodes: " << ps.node_1 << ", " << ps.node_2 << ", active node is: "<< ps.search_on <<"\n";
                    log << "Range on node " << ps.node_1 << ": " << ps.node1_range.start << ", " << ps.node1_range.end;
                    log << ", on node " << ps.node_2 << ": " << ps.node2_range.start << ", " << ps.node2_range.end;
                    log << ", cross_diagonal " << ps.cross_diagonal << ", index: " << ps.cross_diagonal_index << "\n";
                    log << "--> Result: " << ps.result << "\n";
                }
                ++cdidx;
            }
            // Debug
//            uint ppidx = 0;
//            log << "\nPartition points: \n";
//            for (const std::pair<MergePosition, MergePosition>& pos :mpartition_points) {
//                log << ppidx << ". Node " << pos.first.node << ", index: " << pos.first.index << ",  ";
//                log << "Node: " << pos.second.node << ", index: " << pos.second.index << "\n";
//                ++ppidx;
//            }

            uint pidx = 0;
            log << "\nPartitions: \n";
            for (const MergePartition& part : mpartitions) {
                log << "\nPartition " << pidx << " (dest: " << part.dest_node
                          << ", from " << part.dest_range.start << " to " << part.dest_range.end << "):\n";
                log << ":\nRange 1 (" << part.size_from_1 << " elements):\n";
                uint psidx = 0;
                for (const MergePartitionSource& source : part.sources_1) {
                    log << psidx << ". Node " << source.node << ", from "
                              << source.range.start << " to " << source.range.end
                              << " offset: " << source.dest_offset << "\n";
                    ++psidx;
                }
                log << "Range 2 (" << part.size_from_2 << " elements): \n";
                psidx = 0;
                for (const MergePartitionSource& source : part.sources_2) {
                    log << psidx << ". Node " << source.node << ", from "
                              << source.range.start << " to " << source.range.end
                              << " offset: " << source.dest_offset << "\n";
                    ++psidx;
                }
                ++pidx;
            }
        }

    private:

        sa_index_t get_node_num_elements(uint node) const {
            return mnode_utils.get_node_num_elements(node);
        }

        const MergeNodeUtils& mnode_utils;
        uint mfirst_range, msecond_range;
        std::vector<std::vector<TwoWayPartitioningSearch>> mcross_diagonal_searches;
        std::vector<MergePartition> mpartitions;
        // DEBUG:
//        std::vector<std::pair<MergePosition, MergePosition>> mpartition_points;

};
}

#endif // TWO_WAY_MICROMERGE_HPP
