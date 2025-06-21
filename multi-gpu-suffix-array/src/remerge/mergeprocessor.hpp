#ifndef MERGEPROCESSOR_HPP
#define MERGEPROCESSOR_HPP

#include "merge_types.hpp"
#include <cstring>
#include <algorithm>
#include "gossip/context.cuh"

#include "moderngpu/kernel_merge.hxx"

#include "util.h"

#include "multi_way_partitioning_search.hpp"
#include "multi_way_micromerge_on_one_node.hpp"
#include "qdallocator.hpp"
__global__ void printArrays(uint32_t* key, size_t size, size_t rank, int spec)
{
    for (size_t i = 0; i < size; i++)
    {
        printf("[%lu] Key %d: %u\n", rank, spec, key[i]);
    }
    printf("---------------------------------------------------------------------------\n");
}
__global__ void printArrays(uint64_t* key, uint64_t* value, size_t size, size_t rank)
{
    for (size_t i = 0; i < size; i++) {

        printf("[%lu]: sa_rank 1: %lu, old_ranks 2: %lu\n", rank, key[i], value[i]);


    }
    printf("---------------------------------------------------------------------------\n");
}
template<typename ke>
__global__ void printArrays(ke* key, ke* value, size_t size, size_t rank)
{
    for (size_t i = 0; i < size; i++) {

        printf("[%lu]: sa_rank 1: %lu, old_ranks 2: %lu\n", rank, key[i], value[i]);


    }
    printf("---------------------------------------------------------------------------\n");
}
namespace crossGPUReMerge {


    enum MergePathBounds { bounds_lower, bounds_upper };

    // This function comes from ModernGPU.
    template<MergePathBounds bounds = bounds_lower, typename a_keys_it,
        typename b_keys_it, typename int_t, typename comp_t>
    HOST_DEVICE int_t merge_path(a_keys_it a_keys, int_t a_count, b_keys_it b_keys, int_t b_count, int_t diag,
        comp_t comp) {
        using type_t = typename std::iterator_traits<a_keys_it>::value_type;

        int_t begin = max(int_t(0), diag - b_count);
        int_t end = min(diag, a_count);

        while (begin < end) {
            int_t mid = (begin + end) / 2;
            type_t a_key = a_keys[mid];
            type_t b_key = b_keys[diag - 1 - mid];
            bool pred = (bounds_upper == bounds) ? comp(a_key, b_key) : !comp(b_key, a_key);

            if (pred)
                begin = mid + 1;
            else
                end = mid;
        }
        return begin;
    }

    template <typename a_keys_it, typename b_keys_it, typename comp_t>
    __global__ void run_partitioning_search(a_keys_it a_keys, int64_t a_count, b_keys_it b_keys, int64_t b_count,
        int64_t diag, comp_t comp, int64_t* store_result) {
        *store_result = crossGPUReMerge::merge_path(a_keys, a_count, b_keys, b_count, diag, comp);
    }

    template <size_t NUM_GPUS, class mtypes>
    std::array<std::vector<InterNodeCopy>, NUM_GPUS> partitions_to_copies(const std::array<MergeNode<mtypes>, NUM_GPUS>& nodes) {
        std::array<std::vector<InterNodeCopy>, NUM_GPUS> copies;

        for (const auto& node : nodes) {
            for (const MultiWayMergePartition* p : node.scheduled_work.multi_merge_partitions) {
                for (const MergePartitionSource& s : p->sources) {
                    copies[s.node].push_back({ s.node, p->dest_node,
                                              s.range.start,
                                              p->dest_range.start + s.dest_offset,
                                              s.range.end - s.range.start });
                }
            }

            for (const MergePartition* p : node.scheduled_work.merge_partitions) {
                sa_index_t dest1 = p->dest_range.start;
                sa_index_t dest2 = dest1 + p->size_from_1;
                for (const MergePartitionSource& s1 : p->sources_1) {
                    copies[s1.node].push_back({ s1.node, p->dest_node,
                                               s1.range.start,
                                               dest1 + s1.dest_offset,
                                               s1.range.end - s1.range.start });
                }
                for (const MergePartitionSource& s2 : p->sources_2) {
                    copies[s2.node].push_back({ s2.node, p->dest_node,
                                               s2.range.start,
                                               dest2 + s2.dest_offset,
                                               s2.range.end - s2.range.start });
                }
            }
        }
        return copies;
    }

    template <size_t NUM_GPUS, class mtypes, template <size_t, class> class TopologyHelperT>
    class GPUMergeProcessorT {
    public:
        using Context = MultiGPUContext<NUM_GPUS>;
        using MergeNode = MergeNode<mtypes>;

        using key_t = typename mtypes::key_t;
        using value_t = typename mtypes::value_t;

        using TopologyHelper = TopologyHelperT<NUM_GPUS, mtypes>;

        GPUMergeProcessorT(Context& context, QDAllocator& host_temp_allocator)
            : mcontext(context), mtopology_helper(context, mnodes),
            mhost_search_temp_allocator(host_temp_allocator)
        {
        }

        template <class comp_func_t>
        void do_searches(comp_func_t comp) {
            mhost_search_temp_allocator.reset();

            for (MergeNode node : mnodes) {
                if (node.scheduled_work.searches.size() > 0) {

                    printArrays << <1, 1, 0, mcontext.get_gpu_default_stream() >> > (node.info.keys, node.info.num_elements, (size_t)node.info.index, 1);
                }
                mcontext.sync_all_streams();
                comm_world().barrier();
            }

            for (MergeNode& node : mnodes) {

                const uint node_index = node.info.index;
                cudaSetDevice(mcontext.get_device_id(node_index)); CUERR;

                QDAllocator& d_alloc = mcontext.get_device_temp_allocator(node_index);

                for (auto s : node.scheduled_work.searches) {

                    uint other = (node_index == s->node_1) ? s->node_2 : s->node_1;
                    const cudaStream_t& stream = mcontext.get_streams(node_index).at(other);

                    key_t* start_1 = mnodes[s->node_1].info.keys + s->node1_range.start;
                    int64_t size_1 = s->node1_range.end - s->node1_range.start;

                    key_t* start_2 = mnodes[s->node_2].info.keys + s->node2_range.start;
                    int64_t size_2 = s->node2_range.end - s->node2_range.start;

                    s->d_result_ptr = d_alloc.get<int64_t>(1);
                    s->h_result_ptr = mhost_search_temp_allocator.get<int64_t>(1);
                    printf("[%u] size_1: %ld, size_2: %ld, cross_diagonal: %u\n", node_index, size_1, size_2, s->cross_diagonal);

                    printArrays << <1, 1, 0, stream >> > (start_1, start_1, size_1, (size_t)node_index);
                    printArrays << <1, 1, 0, stream >> > (start_2, start_2, size_2, (size_t)node_index + 10);

                    run_partitioning_search << <1, 1, 0, stream >> > (start_1, size_1, start_2, size_2, s->cross_diagonal,
                        comp, s->d_result_ptr);
                    CUERR;

                    cudaMemcpyAsync(s->h_result_ptr, s->d_result_ptr,
                        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);CUERR;
                }

                for (auto ms : node.scheduled_work.multi_searches) {
                    const size_t result_buffer_length = ms->ranges.size() + 1;
                    const cudaStream_t& stream = mcontext.get_gpu_default_stream(node.info.index);

                    ms->d_result_ptr = d_alloc.get<int64_t>(result_buffer_length);
                    ms->h_result_ptr = mhost_search_temp_allocator.get<int64_t>(result_buffer_length);

                    ArrayDescriptor<NUM_GPUS, key_t, int64_t> ad;
                    int i = 0;
                    for (const auto& r : ms->ranges) {
                        ad.keys[i] = mnodes[r.start.node].info.keys + r.start.index;
                        ad.lengths[i] = r.end.index - r.start.index;
                        i++;
                    }

                    multi_find_partition_points << <1, NUM_GPUS, 0, stream >> > (ad, (int64_t)ms->ranges.size(), (int64_t)ms->split_index,
                        comp,
                        (int64_t*)ms->d_result_ptr,
                        (uint*)(ms->d_result_ptr + result_buffer_length - 1)); CUERR;

                    cudaMemcpyAsync(ms->h_result_ptr, ms->d_result_ptr,
                        result_buffer_length * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);CUERR;
                }

            }

            mcontext.sync_all_streams();

            for (MergeNode& node : mnodes) {
                int id = 0;
                for (auto s : node.scheduled_work.searches) {
                    s->result = *s->h_result_ptr;
                    printf("[%lu] result: %ld, id: %d\n", node.info.index, s->result, id);
                }
                for (auto ms : node.scheduled_work.multi_searches) {
                    ms->results.resize(ms->ranges.size());
                    memcpy(ms->results.data(), ms->h_result_ptr, ms->ranges.size() * sizeof(int64_t));
                    ms->range_to_take_one_more = ms->h_result_ptr[ms->ranges.size()] & 0xffffffff;
                }
            }
        }

        void dump_array(const sa_index_t* idx, size_t size) {
            for (size_t i = 0; i < size; ++i) {
                std::cerr << idx[i] << ", ";
            }
        }

        template <class comp_fun_t>
        void do_copy_and_merge(comp_fun_t comp, std::function <void()> dbg_func) {
            (void)dbg_func;
            std::array<std::vector<InterNodeCopy>, NUM_GPUS> copies = partitions_to_copies<NUM_GPUS, mtypes>(mnodes);
            std::array<size_t, NUM_GPUS> detour_sizes;
            for (uint i = 0; i < NUM_GPUS; ++i)
                detour_sizes[i] = mnodes[i].info.detour_buffer_size;




            bool do_values = mnodes[0].info.values != nullptr;

            mtopology_helper.do_copies_async(copies, detour_sizes, do_values);



            std::vector<NodeMultiMerger<NUM_GPUS, mtypes, comp_fun_t>> multi_mergers;
            multi_mergers.reserve(mnodes.size() * mnodes.size()); // Essential because of pointer-init. in c'tor.
            for (const MergeNode& node : mnodes) {
                for (const MultiWayMergePartition* p : node.scheduled_work.multi_merge_partitions) {
                    multi_mergers.emplace_back(mcontext, node, *p, comp, do_values);
                }
            }
            //.......................................................
            mcontext.sync_all_streams();
            //

            //            if (dbg_func)
            //                dbg_func();

            for (const MergeNode& node : mnodes) {
                cudaSetDevice(mcontext.get_device_id(node.info.index));

                mgpu::my_mpgu_context_t& mgpu_context = mcontext.get_mgpu_default_context_for_device(node.info.index);

                for (const MergePartition* p : node.scheduled_work.merge_partitions) {
                    sa_index_t dest1 = p->dest_range.start;
                    sa_index_t dest2 = dest1 + p->size_from_1;
                    mgpu_context.reset_temp_memory();

                    //                    if (p->size_from_1 > 0 && p->size_from_2 > 0) {
                    if (do_values) {
                        mgpu::merge(node.info.key_buffer + dest1, node.info.value_buffer + dest1, (int)p->size_from_1,
                            node.info.key_buffer + dest2, node.info.value_buffer + dest2, (int)p->size_from_2,
                            node.info.keys + dest1, node.info.values + dest1, comp, mgpu_context);CUERR;
                    }
                    else {
                        mgpu::merge(node.info.key_buffer + dest1, (int)p->size_from_1,
                            node.info.key_buffer + dest2, (int)p->size_from_2,
                            node.info.keys + dest1, comp, mgpu_context);CUERR;
                    }
                    //                        printf("\nScheduled merge with sizes %d, %d, on device %u",
                    //                                (int)p->size_from_1, (int)p->size_from_2, node.info.index);
                    //                    }
                }
            }

            //            for (const MergeNode& node : mnodes) {
            //                cudaSetDevice(mcontext.get_device_id(node.info.index));

            //                for (const MultiWayMergePartition* p : node.scheduled_work.multi_merge_partitions) {
            //                // Iteratively merge multi-partitions, queueing all the launches and hoping for the best...
            //                    do_multi_merges(node, *p);
            //                }
            //            }
            // for (MergeNode mnode : mnodes) {
            //     mcontext.sync_gpu_default_stream(mnode.info.index);
            //     printArrays << <1, 1, 0, mcontext.get_gpu_default_stream(mnode.info.index) >> > (mnode.info.key_buffer, mnode.info.keys, mnode.info.num_elements, mnode.info.index);
            //     mcontext.sync_gpu_default_stream(mnode.info.index);
            // }
            // for (MergeNode mnode : mnodes) {
            //     mcontext.sync_gpu_default_stream(mnode.info.index);
            //     printArrays << <1, 1, 0, mcontext.get_gpu_default_stream(mnode.info.index) >> > (mnode.info.value_buffer, mnode.info.values, mnode.info.num_elements, mnode.info.index);
            //     mcontext.sync_gpu_default_stream(mnode.info.index);
            // }
            if (!multi_mergers.empty()) {
                bool finished = false;
                while (!finished) {
                    finished = true;

                    for (auto& merger : multi_mergers) {
                        finished &= merger.do_merge_step();
                    }
                    for (const MergeNode& node : mnodes) {
                        mcontext.get_device_temp_allocator(node.info.index).reset();
                    }
                    for (auto& merger : multi_mergers) {
                        merger.sync_used_streams();
                    }
                    //                    if (dbg_func)
                    //                        dbg_func();
                    //                    mcontext.sync_all_streams();
                }
            }
            mcontext.sync_all_streams();
        }

        void init_nodes(const std::array<MergeNodeInfo<mtypes>, NUM_GPUS>& merge_node_info) {
            for (uint n = 0; n < merge_node_info.size(); ++n) {
                mnodes[n] = MergeNode(merge_node_info[n]);
            }
        }

        std::array<MergeNode, NUM_GPUS>& _get_nodes() { return mnodes; }
        TopologyHelper& topology_helper() { return mtopology_helper; }

    private:
        Context& mcontext;
        std::array<MergeNode, NUM_GPUS> mnodes;
        TopologyHelper mtopology_helper;
        QDAllocator& mhost_search_temp_allocator;
    };

}

#endif // MERGEPROCESSOR_HPP
