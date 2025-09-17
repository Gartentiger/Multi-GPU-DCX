#ifndef REMERGEMANAGER_H
#define REMERGEMANAGER_H

#include "gossip/context.cuh"
#include <vector>
#include <algorithm>

#include "merge_types.hpp"
#include "two_way_micromerge.hpp"
#include "multi_way_micromerge.hpp"
#include "mergeprocessor.hpp"

namespace crossGPUReMerge
{

    template <uint NUM_GPUS, class mtypes, template <size_t, class> class TopologyHelperT>
    class ReMergeManager
    {
    public:
        using MergeNode = MergeNode<mtypes>;
        using MergeProcessor = GPUMergeProcessorT<NUM_GPUS, mtypes, TopologyHelperT>;
        using TopologyHelper = TopologyHelperT<NUM_GPUS, mtypes>;

        ReMergeManager(MultiGPUContext<NUM_GPUS>& context,
            QDAllocator& host_temp_allocator)
            : mmerge_processor(context, host_temp_allocator),
            mmerge_nodes(mmerge_processor._get_nodes())
        {
        }

        ReMergeManager(const ReMergeManager&) = delete;
        ReMergeManager& operator=(const ReMergeManager&) = delete;

        void set_node_info(const std::array<MergeNodeInfo<mtypes>, NUM_GPUS>& merge_node_info)
        {
            mmerge_processor.init_nodes(merge_node_info);
        }

        // We expect the ranges to be non-overlapping, i.e. a GPU can be involved in two merges maximum.
        template <class comp_fun_t>
        void merge(const std::vector<MergeRange>& ranges, comp_fun_t comp,
            std::function<void()> debug_func = nullptr)
        {

            // auto& t = kamping::measurements::timer();
            // t.synchronize_and_start("prep_merges");
            init_node_utils();
            init_micro_ranges(ranges);
            // t.stop();
            //printf("init %lu\n", world_rank());
            // t.synchronize_and_start("schedule_merges");
            // int i = 0;
            while (schedule_micro_merges() > 0)
            {
                // t.synchronize_and_start("iteration:" + std::to_string(i));
                // t.synchronize_and_start("schedule_partitioning_searches:" + std::to_string(i));
                schedule_partitioning_searches();
                // t.stop();
                // t.synchronize_and_start("do_searches:" + std::to_string(i));
                //printf("do search %lu\n", world_rank());
                // nvtxRangePush("do_searches");
                mmerge_processor.do_searches(comp);
                // nvtxRangePop();
                // t.stop();
                // printf("[%lu] do_searches done\n", world_rank());
                // t.synchronize_and_start("create_partitions_from_search_results:" + std::to_string(i));
                create_partitions_from_search_results();
                // t.stop();
                // printf("[%lu] create partitions done\n", world_rank());
                //                debug_print();
                // t.synchronize_and_start("do_copy_and_merge:" + std::to_string(i));
                // nvtxRangePush("do_copy_and_merge");
                mmerge_processor.do_copy_and_merge(comp, debug_func);
                // nvtxRangePop();
                // t.stop();
                // printf("[%lu] do copy and merge done\n", world_rank());
                // t.synchronize_and_start("combine_finished_microranges:" + std::to_string(i));
                combine_finished_microranges();
                // t.stop();
                // t.stop();
                //                std::cerr << "\n\nNew iteration... ---------------------------------------------\n";
            }
            mmerges.clear();
            // t.stop();

            // t.aggregate_and_print(
            //     kamping::measurements::SimpleJsonPrinter{ std::cout }
            // );
            // std::cout << std::endl;
            // t.aggregate_and_print(kamping::measurements::FlatPrinter{});
            // std::cout << std::endl;
            //            std::cerr << "\n\nFinished...\n";
            //            debug_print();
        }

        void debug_print() const
        {
            uint idx = 0;

            std::ostream& log = std::cerr;
            for (const Merge& merge : mmerges)
            {
                log << "\nMerge: " << idx;
                log << "\nMicro ranges: \n";
                uint mridx = 0;
                for (const MergeRange& mr : merge.micro_ranges)
                {
                    log << mridx << ". Start: Node " << mr.start.node << ", index: " << mr.start.index
                        << ", end: Node " << mr.end.node << ", index: " << mr.end.index << "\n";
                    ++mridx;
                }
                uint mmidx = 0;
                log << "\nActive two-way micro merges: \n";
                for (const TwoWayMicroMerge& mm : merge.active_micro_merges)
                {
                    log << "Two-way micro merge " << mmidx << ":\n";
                    mm.debug_print();
                    mmidx++;
                }
                mmidx = 0;
                log << "\nActive multi-way micro merges: \n";
                for (const MultiWayMicroMerge& mmm : merge.active_multi_micro_merges)
                {
                    log << "Multi-way micro merge " << mmidx << ":\n";
                    mmm.debug_print();
                    mmidx++;
                }
                ++idx;
            }
        }

    private:
        void init_micro_ranges(const std::vector<MergeRange>& ranges)
        {
            // At the beginning, we have one micro range per node.
            mmerges.clear();
            for (const auto& range : ranges)
            {
                // Basic sanity check, as I've done stupid things before.
                ASSERT(range.start.node <= range.end.node && range.end.node < NUM_GPUS);
                ASSERT(range.start.index < mnode_utils.get_node_num_elements(range.start.node));
                ASSERT(range.end.index <= mnode_utils.get_node_num_elements(range.end.node));
                mmerges.emplace_back(Merge());
                Merge& merge = mmerges.back();
                // mircro_ranges vector<MergeRange> 
                // MergeRange -> 2 MergePositions | MergePosition -> uint, sa_index==uint32_t
                // range == 0,0 3,num_elements(working_length(num_per_gpu))
                mnode_utils.convert_to_per_node_ranges(range, [&merge](uint node, sa_index_range_t r)
                    { merge.micro_ranges.push_back({ {node, r.start}, {node, r.end} }); });
            }
        }

        size_t schedule_micro_merges()
        {
            size_t scheduled = 0;
            TopologyHelper& topology_helper = mmerge_processor.topology_helper();
            for (Merge& merge : mmerges)
            {
                merge.active_multi_micro_merges.clear();
                merge.active_micro_merges.clear();

                bool each_on_one_different_node = true;
                ASSERT(!merge.micro_ranges.empty());
                uint last_node = 1u << 31;
                for (const auto& r : merge.micro_ranges)
                {
                    each_on_one_different_node &= (r.start.node == r.end.node) && last_node != r.start.node;
                    last_node = r.start.node;
                }

                if (merge.micro_ranges.size() > 2 && each_on_one_different_node)
                {
                    scheduled += topology_helper.schedule_multi_merge_equivalents(merge.micro_ranges,
                        mnode_utils,
                        merge.active_multi_micro_merges,
                        merge.active_micro_merges);
                }
                else if (merge.micro_ranges.size() > 1)
                {
                    for (uint i = 0; i < merge.micro_ranges.size() - 1; i += 2)
                    {
                        merge.active_micro_merges.push_back({ mnode_utils, i, i + 1 });
                        ++scheduled;
                    }
                }
            }
            return scheduled;
        }

        void schedule_partitioning_searches()
        {
            for (MergeNode& node : mmerge_nodes)
            {
                node.scheduled_work.searches.clear();
                node.scheduled_work.multi_searches.clear();
            }
            for (Merge& merge : mmerges)
            {
                for (TwoWayMicroMerge& mm : merge.active_micro_merges)
                {
                    const MergeRange& r1 = merge.micro_ranges[mm.first_range()];
                    const MergeRange& r2 = merge.micro_ranges[mm.second_range()];

                    mm.schedule_searches_for_micro_merge(r1, r2, mmerge_processor.topology_helper());
                    for (auto& cds : mm.cross_diagonal_searches())
                    {
                        for (TwoWayPartitioningSearch& s : cds)
                        {
                            mmerge_nodes[s.search_on].scheduled_work.searches.push_back(&s);
                        }
                    }
                }

                for (MultiWayMicroMerge& mmm : merge.active_multi_micro_merges)
                {
                    mmm.schedule_partitioning_searches();
                    for (MultiWayPartitioningSearch& s : mmm.searches())
                    {
                        mmerge_nodes[s.scheduled_on].scheduled_work.multi_searches.push_back(&s);
                    }
                }
            }
        }

        void create_partitions_from_search_results()
        {
            for (MergeNode& node : mmerge_nodes)
            {
                node.scheduled_work.merge_partitions.clear();
                node.scheduled_work.multi_merge_partitions.clear();
            }
            for (auto& merge : mmerges)
            {
                for (auto& mmm : merge.active_multi_micro_merges)
                {
                    mmm.create_partitions_from_search_results();
                    // Assign to dest nodes.
                    for (auto& p : mmm.partitions())
                    {
                        mmerge_nodes[p.dest_node].scheduled_work.multi_merge_partitions.push_back(&p);
                    }
                }
                for (auto& mm : merge.active_micro_merges)
                {
                    mm.create_partitions_from_search_results(merge.micro_ranges[mm.first_range()],
                        merge.micro_ranges[mm.second_range()]);
                    // Assign to dest nodes.
                    for (auto& p : mm.partitions())
                    {
                        mmerge_nodes[p.dest_node].scheduled_work.merge_partitions.push_back(&p);
                    }
                }
            }
        }

        void combine_finished_microranges()
        {
            for (Merge& merge : mmerges)
            {
                std::vector<MergeRange> consolidated_micro_ranges;
                consolidated_micro_ranges.reserve(merge.micro_ranges.size() / 2 + 1);

                for (MultiWayMicroMerge& multi_micro_merge : merge.active_multi_micro_merges)
                {
                    const MergeRange& r_start = merge.micro_ranges[multi_micro_merge.start_range_index()];
                    const MergeRange& r_end = merge.micro_ranges[multi_micro_merge.end_range_index()];
                    consolidated_micro_ranges.push_back({ r_start.start, r_end.end });
                }
                for (TwoWayMicroMerge& micro_merge : merge.active_micro_merges)
                {
                    MergeRange combined = combine_ranges(merge.micro_ranges[micro_merge.first_range()],
                        merge.micro_ranges[micro_merge.second_range()]);
                    // FIXME: may need sorting here, occurs only due to DGX-1 topology at the moment.
                    if (!consolidated_micro_ranges.empty() && consolidated_micro_ranges.back().start.node > combined.start.node)
                    {
                        consolidated_micro_ranges.insert(consolidated_micro_ranges.begin(), combined);
                    }
                    else
                    {
                        consolidated_micro_ranges.push_back(combined);
                    }
                }
                // Now add the ones not merged in this iteration; yes, this is a verbose implementation.
                for (uint mr_index = 0; mr_index < merge.micro_ranges.size(); ++mr_index)
                {
                    bool found = false;
                    for (MultiWayMicroMerge& multi_micro_merge : merge.active_multi_micro_merges)
                    {
                        if (mr_index >= multi_micro_merge.start_range_index() &&
                            mr_index <= multi_micro_merge.end_range_index())
                        {
                            found = true;
                            break;
                        }
                    }
                    for (TwoWayMicroMerge& micro_merge : merge.active_micro_merges)
                    {
                        if (micro_merge.first_range() == mr_index || micro_merge.second_range() == mr_index)
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        // FIXME: may need sorting here, occurs only due to DGX-1 topology at the moment.
                        const MergeRange& not_yet_merged = merge.micro_ranges[mr_index];
                        if (!consolidated_micro_ranges.empty() && consolidated_micro_ranges.back().start.node > not_yet_merged.start.node)
                        {
                            consolidated_micro_ranges.insert(consolidated_micro_ranges.begin(), not_yet_merged);
                        }
                        else
                        {
                            consolidated_micro_ranges.push_back(not_yet_merged);
                        }
                    }
                }
                merge.micro_ranges = consolidated_micro_ranges;
                //                fprintf(stderr, "Consolidated:\n");
                //                for (const MergeRange& mr : merge.micro_ranges) {
                //                    fprintf(stderr, "%u, %u to %u, %u\n", mr.start.node, mr.start.index, mr.end.node, mr.end.index);
                //                }
                merge.active_micro_merges.clear();
            }
        }

        static MergeRange combine_ranges(const MergeRange& r1, const MergeRange& r2)
        {
            return { r1.start, r2.end };
        }

        void init_node_utils()
        {
            std::vector<size_t> sizes;
            sizes.reserve(NUM_GPUS);
            for (const auto& node : mmerge_nodes)
            {
                sizes.push_back(node.info.num_elements);
            }

            mnode_utils.init(std::move(sizes));
        }

        struct Merge
        {
            std::vector<MergeRange> micro_ranges;
            std::vector<TwoWayMicroMerge> active_micro_merges;
            std::vector<MultiWayMicroMerge> active_multi_micro_merges;
        };

        MergeProcessor mmerge_processor;
        std::array<MergeNode, NUM_GPUS>& mmerge_nodes;
        std::vector<Merge> mmerges;
        MergeNodeUtils mnode_utils;
    };

}
#endif // MERGEMANAGER_H
