#ifndef REMERGE_GPU_TOPOLOGY_HELPER_HPP
#define REMERGE_GPU_TOPOLOGY_HELPER_HPP

#include <vector>
#include <array>

#include "merge_types.hpp"
#include "mergenodeutils.hpp"
#include "gossip/context.cuh"
#include "multi_way_micromerge.hpp"
#include "two_way_micromerge.hpp"

#include "../merge_copy_detour_guide.hpp"

#include <span>
#include <kamping/p2p/recv.hpp>
#include <kamping/p2p/send.hpp>

namespace crossGPUReMerge {

    template <size_t NUM_GPUS, class mtypes>
    class DGX1TopologyHelper {
        static_assert(NUM_GPUS == 8, "DGX-1 topology requires 8 GPUs (merge).");

        using Context = MultiGPUContext<NUM_GPUS>;
        using MergeNode = MergeNode<mtypes>;

        Context& mcontext;
        std::array<MergeNode, NUM_GPUS>& mnodes;

        mergeCopying::MergeDGX1CopyDetourGuide mcopy_detour_guide;

    public:
        DGX1TopologyHelper(Context& context, std::array<MergeNode, NUM_GPUS>& nodes)
            : mcontext(context), mnodes(nodes), mcopy_detour_guide(context)
        {
        }

        inline uint get_quad(uint gpu) const {
            return mcontext.get_device_id(gpu) / 4;
        }

        // MultiMerges can only be scheduled if P2P access is available for all nodes involved.
        // Return: the number of merges (of either type) scheduled.
        size_t schedule_multi_merge_equivalents(const std::vector<MergeRange>& micro_ranges,
            const MergeNodeUtils& node_utils,
            std::vector<MultiWayMicroMerge>& mwmm,
            std::vector<TwoWayMicroMerge>& twmm) const {
            size_t scheduled = 0;
            // We can only do multi merges within each quad, but not across.
            uint quad = get_quad(micro_ranges[0].start.node);
            uint last_quad_range = 0;
            for (uint r = 1; r < micro_ranges.size(); ++r) {
                if (quad != get_quad(micro_ranges[r].start.node)) {
                    last_quad_range = r - 1;
                    break;
                }
            }

            if (last_quad_range > 1) {
                std::vector<MergeRange> spliced(&micro_ranges[0],
                    &micro_ranges[last_quad_range + 1]);
                mwmm.push_back({ node_utils, spliced,
                                 0, last_quad_range });
                ++scheduled;
            }
            else if (last_quad_range > 0) {
                twmm.push_back({ node_utils, 0, 1 });
                ++scheduled;
            }

            // Now, emit the remainder (of different quad) if we can...

            uint remaining = micro_ranges.size() - 1 - last_quad_range;
            if (remaining > 2) {
                std::vector<MergeRange> spliced(&micro_ranges[last_quad_range + 1],
                    &micro_ranges[(uint)micro_ranges.size()]);
                mwmm.push_back({ node_utils, spliced,
                                 last_quad_range + 1, (uint)micro_ranges.size() - 1 });
                ++scheduled;
            }
            else if (remaining == 2) {
                twmm.push_back({ node_utils, last_quad_range + 1, (uint)micro_ranges.size() - 1 });
                ++scheduled;
            }

            return scheduled;
        }

        unsigned int get_node_to_schedule_two_way_micro_merge_search_on(uint node1, uint node2,
            bool alternate) const {
            ASSERT(node1 != node2);
            if (mcontext.get_peer_status(node1, node2) == Context::PEER_STATUS_FAST) {
                return alternate ? node2 : node1;
            }
            else {
                // Take node inbetween.
                uint id1 = mcontext.get_device_id(node1);
                uint id2 = mcontext.get_device_id(node2);
                if (id1 > id2)
                    std::swap(id1, id2);
                ASSERT(id1 < 4 && id2 >= 4);
                node1 = mcopy_detour_guide.get_reverse_id(id1 + 4);
                node2 = mcopy_detour_guide.get_reverse_id(id2 - 4);
                return alternate ? node2 : node1; // FIXME: not quite right, but we don't care yet.
            }
        }

        void do_copies_async(const std::array<std::vector<InterNodeCopy>, NUM_GPUS>& copies,
            const std::array<size_t, NUM_GPUS>& detour_buffer_sizes,
            bool do_values) const {

            mcopy_detour_guide.copy_with_detours_async(copies, detour_buffer_sizes, do_values,
                [this](uint src_node, uint dest_node,
                    sa_index_t src_index, sa_index_t dest_index,
                    size_t len, int extra, mergeCopying::CopyMode mode,
                    bool do_values) {
                        using key_t = typename mtypes::key_t;
                        using value_t = typename mtypes::value_t;
                        (void)extra;
                        // Assume src device is active.
                        const key_t* src_k_buff;
                        key_t* dest_k_buff;
                        if (mode == mergeCopying::CopyMode::FromDetour) {
                            src_k_buff = mnodes[src_node].info.key_detour_buffer;
                            dest_k_buff = mnodes[dest_node].info.key_buffer;
                        }
                        else if (mode == mergeCopying::CopyMode::ToDetour) {
                            src_k_buff = mnodes[src_node].info.keys;
                            dest_k_buff = mnodes[dest_node].info.key_detour_buffer;
                        }
                        else { // Direct
                            src_k_buff = mnodes[src_node].info.keys;
                            dest_k_buff = mnodes[dest_node].info.key_buffer;
                        }

                        cudaMemcpyPeerAsync(dest_k_buff + dest_index, mcontext.get_device_id(dest_node),
                            src_k_buff + src_index, mcontext.get_device_id(src_node),
                            len * sizeof(key_t),
                            mcontext.get_streams(src_node)[dest_node]);CUERR;
                        if (do_values) {
                            const value_t* src_v_buff;
                            value_t* dest_v_buff;
                            if (mode == mergeCopying::CopyMode::FromDetour) {
                                src_v_buff = mnodes[src_node].info.value_detour_buffer;
                                dest_v_buff = mnodes[dest_node].info.value_buffer;
                            }
                            else if (mode == mergeCopying::CopyMode::ToDetour) {
                                src_v_buff = mnodes[src_node].info.values;
                                dest_v_buff = mnodes[dest_node].info.value_detour_buffer;
                            }
                            else { // Direct
                                src_v_buff = mnodes[src_node].info.values;
                                dest_v_buff = mnodes[dest_node].info.value_buffer;
                            }
                            cudaMemcpyPeerAsync(dest_v_buff + dest_index, mcontext.get_device_id(dest_node),
                                src_v_buff + src_index, mcontext.get_device_id(src_node),
                                len * sizeof(value_t),
                                mcontext.get_streams(src_node)[dest_node]);CUERR;
                        }
                });
        }
    };


    template <size_t NUM_GPUS, class mtypes>
    class MergeGPUAllConnectedTopologyHelper {
        using Context = MultiGPUContext<NUM_GPUS>;
        using MergeNode = MergeNode<mtypes>;

        Context& mcontext;
        std::array<MergeNode, NUM_GPUS>& mnodes;

    public:
        MergeGPUAllConnectedTopologyHelper(MultiGPUContext<NUM_GPUS>& context,
            std::array<MergeNode, NUM_GPUS>& nodes)
            : mcontext(context), mnodes(nodes)
        {
        }

        // MultiMerges can only be scheduled if P2P access is available for all nodes involved.
        // Return: the number of merges (of either type) scheduled.
        size_t schedule_multi_merge_equivalents(const std::vector<MergeRange>& micro_ranges,
            const MergeNodeUtils& node_utils,
            std::vector<MultiWayMicroMerge>& mwmm,
            std::vector<TwoWayMicroMerge>& twmm) const {
            (void)twmm;
            mwmm.push_back({ node_utils, micro_ranges,
                             0, (uint)micro_ranges.size() - 1 });
            return 1;
        }

        unsigned int get_node_to_schedule_two_way_micro_merge_search_on(uint node1, uint node2,
            bool alternate) const {
            return alternate ? node2 : node1;
        }

        void do_copies_async(const std::array<std::vector<InterNodeCopy>, NUM_GPUS>& copies,
            const std::array<size_t, NUM_GPUS>& detour_buffer_sizes,
            bool do_values) const {
            using key_t = typename mtypes::key_t;
            using value_t = typename mtypes::value_t;
            // make sure computations are done before copying
            mcontext.sync_all_streams();
            (void)detour_buffer_sizes;
            for (uint node = 0; node < NUM_GPUS; ++node) {
                cudaSetDevice(mcontext.get_device_id(node));CUERR;
                for (const InterNodeCopy& c : copies[node]) {
                    ASSERT(c.src_node == node);

                    // cudaMemcpyPeerAsync(dest_k_buff + c.dest_index, mcontext.get_device_id(c.dest_node),
                    //     src_k_buff + c.src_index, mcontext.get_device_id(c.src_node),
                    //     c.len * sizeof(typename mtypes::key_t),
                    //     mcontext.get_streams(node)[c.dest_node]);CUERR;
                    if (c.src_node == world_rank()) {
                        key_t* src_k_buff = mnodes[c.src_node].info.key + c.src_indexs;
                        std::span<uint64_t> sb(reinterpret_cast<uint64_t>(src_k_buff), c.len);
                        comm_world().send(send_buf(sb), send_count(c.len), destination((size_t)c.dest_node));
                    }
                    if (c.dest_node == world_rank()) {
                        key_t* dest_k_buff = mnodes[c.dest_node].info.key_buffer + c.dest_index;
                        std::span<uint64_t> rb(reinterpret_cast<uint64_t>(dest_k_buff), c.len);
                        comm_world().recv(recv_buf(rb), recv_count(c.len));
                    }
                    if (do_values) {
                        if (c.src_node == world_rank()) {
                            value_t* src_v_buff = mnodes[c.src_node].info.values + c.src_index;
                            std::span<uint32_t> sb(reinterpret_cast<uint32_t>(src_v_buff), c.len);
                            comm_world().send(send_buf(sb), send_count(c.len), destination((size_t)c.dest_node));
                        }
                        if (c.dest_node == world_rank()) {
                            value_t* dest_v_buff = mnodes[c.dest_node].info.value_buffer + c.dest_index;
                            std::span<uint32_t> rb(reinterpret_cast<uint32_t>(dest_v_buff), c.len);
                            comm_world().recv(recv_buf(rb), recv_count(c.len));
                        }
                        // cudaMemcpyPeerAsync(dest_v_buff + c.dest_index, mcontext.get_device_id(c.dest_node),
                        //     src_v_buff + c.src_index, mcontext.get_device_id(c.src_node),
                        //     c.len * sizeof(typename mtypes::value_t),
                        //     mcontext.get_streams(node)[c.dest_node]);CUERR;
                    }
                }

            }
        }
    };

}

#endif // REMERGE_GPU_TOPOLOGY_HPP
