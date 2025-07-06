#ifndef DISTRIB_MERGE_TOPOLOGY_HELPER_HPP
#define DISTRIB_MERGE_TOPOLOGY_HELPER_HPP

#include "gossip/context.cuh"
#include "../merge_copy_detour_guide.hpp"
#include "../gossip/auxiliary.cuh"

#include "distrib_merge_array.hpp"

#include <kamping/request_pool.hpp>
#include <kamping/p2p/irecv.hpp>
#include <kamping/p2p/isend.hpp>

namespace distrib_merge {

    template<typename key_t, typename value_t, typename index_t, size_t NUM_GPUS>
    class DistribMergeAllConnectedTopologyHelper {
        using Context = MultiGPUContext<NUM_GPUS>;
        using InterNodeCopy = InterNodeCopy<index_t>;
        using DistributedArray = distrib_merge::DistributedArray<key_t, value_t, index_t, NUM_GPUS>;

        Context& mcontext;

    public:
        DistribMergeAllConnectedTopologyHelper(MultiGPUContext<NUM_GPUS>& context)
            : mcontext(context)
        {
        }

        unsigned int get_node_to_schedule_search_on(uint node1, uint node2,
            const std::array<uint, NUM_GPUS>& searches_per_node) const {
            if (searches_per_node[node1] > searches_per_node[node2])
                return node2;
            else
                return node1;
        }


        void do_copies_async(const std::array<std::vector<InterNodeCopy>, NUM_GPUS>& copies,
            const DistributedArray& a, const DistributedArray& b,
            DistributedArray& out,
            bool do_values) const {
            RequestPool pool;
            int msgTag = 0;
            for (uint node = 0; node < NUM_GPUS; ++node) {
                //(mcontext.get_device_id(node));CUERR;
                for (const InterNodeCopy& c : copies[node]) {
                    ASSERT(c.src_node == node);

                    if (c.src_node == world_rank()) {
                        //const
                        key_t* src_k_buff = c.extra ? b[c.src_node].keys : a[c.src_node].keys;
                        //key_t* src_k_buff = mnodes[c.src_node].info.keys + c.src_index;
                        std::span<key_t> sb(src_k_buff + c.src_index, c.len);
                        comm_world().isend(send_buf(sb), send_count(c.len), tag(msgTag), destination((size_t)c.dest_node), request(pool.get_request()));
                        if (do_values) {
                            value_t* src_v_buff = c.extra ? b[c.src_node].values : a[c.src_node].values;
                            //value_t* src_v_buff = mnodes[c.src_node].info.values + c.src_index;
                            std::span<value_t> sb(src_v_buff + c.src_index, c.len);
                            comm_world().isend(send_buf(sb), send_count(c.len), tag(msgTag + 1), destination((size_t)c.dest_node), request(pool.get_request()));

                        }
                    }
                    if (c.dest_node == world_rank()) {
                        key_t* dest_k_buff = out[c.dest_node].keys_buffer + c.dest_index;
                        //key_t* dest_k_buff = mnodes[c.dest_node].info.key_buffer + c.dest_index;
                        std::span<key_t> rb(dest_k_buff, c.len);
                        comm_world().irecv(recv_buf(rb), tag(msgTag), recv_count(c.len), request(pool.get_request()));
                        if (do_values) {
                            value_t* dest_v_buff = out[c.dest_node].values_buffer + c.dest_index;
                            //value_t* dest_v_buff = mnodes[c.dest_node].info.value_buffer + c.dest_index;
                            std::span<value_t> rb(dest_v_buff, c.len);
                            comm_world().irecv(recv_buf(rb), tag(msgTag + 1), recv_count(c.len), request(pool.get_request()));
                        }
                    }
                    msgTag += 2;


                    // cudaMemcpyPeerAsync(dest_v_buff + c.dest_index, mcontext.get_device_id(c.dest_node),
                    //     src_v_buff + c.src_index, mcontext.get_device_id(c.src_node),
                    //     c.len * sizeof(typename mtypes::value_t),
                    //     mcontext.get_streams(node)[c.dest_node]);CUERR;

                // cudaMemcpyPeerAsync(dest_k_buff + c.dest_index, mcontext.get_device_id(c.dest_node),
                //     src_k_buff + c.src_index, mcontext.get_device_id(c.src_node),
                //     c.len * sizeof(key_t),
                //     mcontext.get_streams(node)[c.dest_node]);CUERR;
                // if (do_values) {
                    // cudaMemcpyPeerAsync(dest_v_buff + c.dest_index, mcontext.get_device_id(c.dest_node),
                    //     src_v_buff + c.src_index, mcontext.get_device_id(c.src_node),
                    //     c.len * sizeof(value_t),
                    //     mcontext.get_streams(node)[c.dest_node]);CUERR;
                // }
                }
            }
            pool.wait_all();
        }
    };

    template<typename key_t, typename value_t, typename index_t, size_t NUM_GPUS>
    class DGX1TopologyHelper {
        static_assert(NUM_GPUS == 8, "DGX-1 topology requires 8 GPUs (distrib-merge).");

        using Context = MultiGPUContext<NUM_GPUS>;
        using InterNodeCopy = InterNodeCopy<index_t>;
        using DistributedArray = distrib_merge::DistributedArray<key_t, value_t, index_t, NUM_GPUS>;

        Context& mcontext;
        mergeCopying::MergeDGX1CopyDetourGuide mcopy_detour_guide;

    public:
        DGX1TopologyHelper(Context& context)
            : mcontext(context), mcopy_detour_guide(context) {
        }

        unsigned int get_node_to_schedule_search_on(uint node1, uint node2,
            const std::array<uint, NUM_GPUS>& searches_per_node) const {
            ASSERT(node1 != node2);
            if (mcontext.get_peer_status(node1, node2) == Context::PEER_STATUS_FAST) {
                if (searches_per_node[node1] > searches_per_node[node2])
                    return node2;
                else
                    return node1;
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

                if (searches_per_node[node1] > searches_per_node[node2])
                    return node2;
                else
                    return node1;
            }
        }

        void do_copies_async(const std::array<std::vector<InterNodeCopy>, NUM_GPUS>& copies,
            const DistributedArray& a, const DistributedArray& b,
            DistributedArray& out, bool do_values) const {
            std::array<size_t, NUM_GPUS> detour_buffer_sizes;

            std::array<std::vector<InterNodeCopy>, NUM_GPUS> deferred_direct_copies;
            for (uint i = 0; i < NUM_GPUS; ++i) {
                detour_buffer_sizes[i] = out[i].count;
                deferred_direct_copies[i].reserve(2 * NUM_GPUS * (NUM_GPUS - 3));
            }

            mcopy_detour_guide.copy_with_detours_async(copies, detour_buffer_sizes, do_values,
                [this, &deferred_direct_copies, &out, &a, &b]
                (uint src_node, uint dest_node,
                    index_t src_index, index_t dest_index,
                    size_t len, int extra, mergeCopying::CopyMode mode,
                    bool do_values) {
                        // Assume src device is active.
                        key_t* src_k_buff;
                        key_t* dest_k_buff;
                        if (mode == mergeCopying::CopyMode::FromDetour) {
                            src_k_buff = out[src_node].keys_buffer;
                            dest_k_buff = out[dest_node].keys_buffer;
                        }
                        else if (mode == mergeCopying::CopyMode::ToDetour) {
                            src_k_buff = extra == 0 ? a[src_node].keys : b[src_node].keys;
                            dest_k_buff = out[dest_node].keys_buffer;
                        }
                        else { // Direct
                            deferred_direct_copies[src_node].push_back({ src_node, dest_node,
                                                                         src_index, dest_index,
                                                                         len, extra });
                            return;
                        }

                        cudaMemcpyPeerAsync(dest_k_buff + dest_index, mcontext.get_device_id(dest_node),
                            src_k_buff + src_index, mcontext.get_device_id(src_node),
                            len * sizeof(key_t),
                            mcontext.get_streams(src_node)[dest_node]);CUERR;
                        if (do_values) {
                            value_t* src_v_buff;
                            value_t* dest_v_buff;

                            if (mode == mergeCopying::CopyMode::FromDetour) {
                                src_v_buff = out[src_node].values;
                                dest_v_buff = out[src_node].values_buffer;
                            }
                            else if (mode == mergeCopying::CopyMode::ToDetour) {
                                src_v_buff = extra == 0 ? a[src_node].values : b[src_node].values;
                                dest_v_buff = out[dest_node].values_buffer;;
                            }
                            else { // Direct
                                ASSERT(false);
                            }
                            cudaMemcpyPeerAsync(dest_v_buff + dest_index, mcontext.get_device_id(dest_node),
                                src_v_buff + src_index, mcontext.get_device_id(src_node),
                                len * sizeof(value_t),
                                mcontext.get_streams(src_node)[dest_node]);CUERR;
                        }
                });

            mcontext.sync_all_streams();
            for (uint node = 0; node < NUM_GPUS; ++node) {
                //(mcontext.get_device_id(node));
                for (const InterNodeCopy& c : deferred_direct_copies[node]) {
                    key_t* src_k_buff = c.extra ? b[c.src_node].keys : a[c.src_node].keys;
                    key_t* dest_k_buff = out[c.dest_node].keys_buffer;

                    cudaMemcpyPeerAsync(dest_k_buff + c.dest_index, mcontext.get_device_id(c.dest_node),
                        src_k_buff + c.src_index, mcontext.get_device_id(c.src_node),
                        c.len * sizeof(key_t),
                        mcontext.get_streams(node)[c.dest_node]);CUERR;
                    if (do_values) {
                        value_t* src_v_buff = c.extra ? b[c.src_node].values : a[c.src_node].values;
                        value_t* dest_v_buff = out[c.dest_node].values_buffer;
                        cudaMemcpyPeerAsync(dest_v_buff + c.dest_index, mcontext.get_device_id(c.dest_node),
                            src_v_buff + c.src_index, mcontext.get_device_id(c.src_node),
                            c.len * sizeof(value_t),
                            mcontext.get_streams(node)[c.dest_node]);CUERR;
                    }

                }
            }
        }
    };

}

#endif // DISTRIB_MERGE_TOPOLOGY_HELPER_HPP
