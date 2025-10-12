#ifndef MULTI_WAY_MICROMERGE_HPP
#define MULTI_WAY_MICROMERGE_HPP

#include "merge_types.hpp"
#include "mergenodeutils.hpp"

namespace crossGPUReMerge {

    class MultiWayMicroMerge {
    public:
        MultiWayMicroMerge(const MergeNodeUtils& node_utils,
            const std::vector<MergeRange>& micro_source_ranges, uint start_range_index,
            uint end_range_index)
            : mnode_utils(node_utils), mmicro_source_ranges(micro_source_ranges),
            mstart_range_index(start_range_index), mend_range_index(end_range_index)
        {
            ASSERT(micro_source_ranges.size() > 2);
            ASSERT(micro_source_ranges.size() == end_range_index - start_range_index + 1);
            for (const auto& r : micro_source_ranges) {
                ASSERT(r.start.node == r.end.node);
            }
        }

        void schedule_partitioning_searches() {
            msearches.clear();
            size_t N = mmicro_source_ranges.size();
            msearches.reserve(N);
            size_t elems_offset = 0;
            for (uint i = 0; i < N - 1; ++i) {
                elems_offset += mmicro_source_ranges[i].end.index - mmicro_source_ranges[i].start.index;
                msearches.emplace_back(MultiWayPartitioningSearch{ mmicro_source_ranges });
                // TODO: better placements? We can place the search on any node here since nodes are
                //       interconnected.
                msearches.back().scheduled_on = mmicro_source_ranges[i].start.node;
                msearches.back().split_index = elems_offset;
            }
        }

        void create_partitions_from_search_results() {
            const size_t NO_RANGES = mmicro_source_ranges.size();

            ASSERT(msearches.size() == NO_RANGES - 1);
            std::vector<size_t> last_split_offs(NO_RANGES);

            mpartitions.reserve(NO_RANGES);
            mpartitions.clear();

            size_t range_to_take_one_more = 0;

            for (size_t i = 0; i < NO_RANGES; ++i) {
                last_split_offs[i] = mmicro_source_ranges[i].start.index;
            }

            for (size_t curr_range_index = 0; curr_range_index < NO_RANGES; ++curr_range_index) {
                std::vector<size_t> current_split_offs(NO_RANGES);
                const MergeRange& current_range = mmicro_source_ranges[curr_range_index];
                const MultiWayPartitioningSearch& current_search = msearches[curr_range_index];

                uint current_node = current_range.start.node;

                if (curr_range_index < NO_RANGES - 1) {
                    for (uint i = 0; i < NO_RANGES; ++i) {
                        current_split_offs[i] = mmicro_source_ranges[i].start.index + current_search.results[i];
                    }
                    range_to_take_one_more = current_search.range_to_take_one_more;
                    ASSERT(range_to_take_one_more < NO_RANGES);
                }
                else {
                    for (size_t i = 0; i < NO_RANGES; ++i) {
                        current_split_offs[i] = mmicro_source_ranges[i].end.index;
                    }
                }
                size_t dest_offset = 0;

                mpartitions.emplace_back(MultiWayMergePartition());

                MultiWayMergePartition& p = mpartitions.back();
                p.dest_node = current_node;
                p.dest_range.start = current_range.start.index;
                p.dest_range.end = current_range.end.index;
                p.sources.reserve(NO_RANGES);
                p.dest_micro_ranges.reserve(NO_RANGES);

                const size_t dest_size = p.dest_range.end - p.dest_range.start;

                size_t source_sum = 0;
                for (uint source_range = 0; source_range < NO_RANGES; ++source_range) {
                    source_sum += current_split_offs[source_range] - last_split_offs[source_range];
                    p.sources.push_back({ mmicro_source_ranges[source_range].start.node,
                                         { (sa_index_t)last_split_offs[source_range],
                                           (sa_index_t)current_split_offs[source_range] },
                                         (sa_index_t)dest_offset });
                    const sa_index_t next_offset = dest_offset + current_split_offs[source_range] - last_split_offs[source_range];
                    //                    printf("\nRange %u: %u, %u\n", source_range, mmicro_source_ranges[source_range].start.index + (sa_index_t) dest_offset,
                    //                           mmicro_source_ranges[source_range].start.index + (sa_index_t) next_offset);
                    p.dest_micro_ranges.push_back({ current_range.start.index + (sa_index_t)dest_offset,
                                                    current_range.start.index + (sa_index_t)next_offset });
                    dest_offset = next_offset;
                }
                ASSERT(source_sum == dest_size);

                if (dest_offset < dest_size) {
                    current_split_offs[range_to_take_one_more]++;
                    p.sources[range_to_take_one_more].range.end++;
                    p.dest_micro_ranges[range_to_take_one_more].end++;
                    if (range_to_take_one_more + 1 < NO_RANGES)
                        p.dest_micro_ranges[range_to_take_one_more + 1].start++;
                }
                last_split_offs = current_split_offs;
            }
            // debug_print();
            // exit(0);
        }

        void debug_print() const {
            auto& log = std::cerr;
            log << "\n\nRanges: " << mstart_range_index << " to " << mend_range_index << "\n";
            uint sidx = 0;
            for (const MultiWayPartitioningSearch& ps : msearches) {
                log << "Split index: " << ps.split_index;
                log << ", scheduled on: " << ps.scheduled_on << "\nResults:\n";
                for (uint i = 0; i < ps.results.size(); ++i) {
                    log << ps.results[i] << ", ";
                }
                log << "range to take one more: " << ps.range_to_take_one_more << "\n";

                ++sidx;
            }
            uint pidx = 0;
            log << "\nPartitions: \n";
            for (const MultiWayMergePartition& part : mpartitions) {
                log << "\nPartition " << pidx << " (dest: " << part.dest_node
                    << ", from " << part.dest_range.start << " to " << part.dest_range.end << "):\n";

                uint psidx = 0;
                for (const MergePartitionSource& source : part.sources) {
                    log << psidx << ". Node " << source.node << ", from "
                        << source.range.start << " to " << source.range.end
                        << " offset: " << source.dest_offset << "\n";
                    ++psidx;
                }
                ++pidx;
            }
        }

        std::vector<MultiWayMergePartition>& partitions() { return mpartitions; }
        std::vector<MultiWayPartitioningSearch>& searches() { return msearches; }

        uint start_range_index() const { return mstart_range_index; }
        uint end_range_index() const { return mend_range_index; }

    private:
        const MergeNodeUtils& mnode_utils;
        uint mstart_range_index, mend_range_index;
        std::vector<MergeRange> mmicro_source_ranges;
        std::vector<MultiWayPartitioningSearch> msearches;
        std::vector<MultiWayMergePartition> mpartitions;
    };


}

#endif // MULTI_WAY_MICROMERGE_HPP
