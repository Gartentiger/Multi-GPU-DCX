#pragma once

#include <type_traits>
#include <array>
#include <vector>

#include "context.cuh"
#include "auxiliary.cuh"
#include "util.h"
#include <kamping/request_pool.hpp>
#include <kamping/p2p/irecv.hpp>
#include <kamping/p2p/isend.hpp>

namespace gossip {

    template<uint NUM_GPUS>
    class All2All {
        using Context = MultiGPUContext<NUM_GPUS>;
        using device_id_t = typename Context::device_id_t;
        Context& context;

    public:
        static const uint num_gpus = NUM_GPUS;

        All2All(Context& context_)
            : context(context_)
        {
        }

        template <typename key_t, typename value_t, typename index_t, typename table_t>
        bool execAsync(const std::array<All2AllNodeInfoT<key_t, value_t, index_t>, NUM_GPUS>& node_info,
            const split_table_tt<table_t, NUM_GPUS>& table) const {
            if (context.is_in_node()) {
                return execAsyncInNode(node_info, table);
            }

            // compute prefix sums over the partition table
            std::array<std::array<table_t, num_gpus + 1>, num_gpus> h_table = { {0} }; // horizontal scan
            std::array<std::array<table_t, num_gpus>, num_gpus + 1> v_table = { {0} }; // vertical scan

            ncclComm_t nccl_comm = context.get_nccl();
            // necessary sync because we cant use the stream for communication 
            context.sync_all_streams();

            ncclGroupStart();
            for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                for (uint dest_gpu = 0; dest_gpu < num_gpus; ++dest_gpu) {
                    h_table[src_gpu][dest_gpu + 1] = table[src_gpu][dest_gpu] + h_table[src_gpu][dest_gpu];
                    v_table[src_gpu + 1][dest_gpu] = table[src_gpu][dest_gpu] + v_table[src_gpu][dest_gpu];

                    // printf("[%lu][%u][%u]: h_table: %u, v_table: %u, table: %u, next h_table: %u, next v_table: %u\n", world_rank(), src_gpu, dest_gpu, h_table[src_gpu][dest_gpu], v_table[src_gpu][dest_gpu], table[src_gpu][dest_gpu], h_table[src_gpu][dest_gpu + 1], v_table[src_gpu][dest_gpu + 1]);

                    const table_t src_index = h_table[src_gpu][dest_gpu];
                    const table_t dest_index = v_table[src_gpu][dest_gpu];
                    const table_t len = table[src_gpu][dest_gpu];

                    if (src_gpu == world_rank()) {
                        key_t* from_k = node_info[src_gpu].src_keys + src_index;
                        if (context.get_peer_status(src_gpu, dest_gpu) >= 1) {
                            key_t* to_k = node_info[dest_gpu].dest_keys + dest_index;
                            cudaMemcpyPeerAsync(to_k, context.get_device_id(dest_gpu),
                                from_k, context.get_device_id(src_gpu),
                                len * sizeof(key_t),
                                context.get_streams(src_gpu)[dest_gpu]);
                        }
                        else {
                            ncclSend(from_k, sizeof(key_t) * len, ncclChar, dest_gpu, nccl_comm, context.get_streams(src_gpu)[dest_gpu]);
                        }
                    }
                    if (dest_gpu == world_rank() && context.get_peer_status(src_gpu, dest_gpu) < 1) {
                        key_t* to_k = node_info[dest_gpu].dest_keys + dest_index;

                        ncclRecv(to_k, sizeof(key_t) * len, ncclChar, src_gpu, nccl_comm, context.get_streams(dest_gpu)[src_gpu]);
                    }
                } CUERR;
            }
            ncclGroupEnd();

            return check_tables(node_info, h_table, v_table);
        }
        template <typename key_t, typename value_t, typename index_t, typename table_t>
        bool execAsyncInNode(const std::array<All2AllNodeInfoT<key_t, value_t, index_t>, NUM_GPUS>& node_info,
            const split_table_tt<table_t, NUM_GPUS>& table) const {
            // compute prefix sums over the partition table
            std::array<std::array<table_t, num_gpus + 1>, num_gpus> h_table = { {0} }; // horizontal scan
            std::array<std::array<table_t, num_gpus>, num_gpus + 1> v_table = { {0} }; // vertical scan

            // necessary sync because we cant use the stream for communication 
            context.sync_all_streams();

            for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                for (uint dest_gpu = 0; dest_gpu < num_gpus; ++dest_gpu) {
                    if (src_gpu == world_rank()) {
                        h_table[src_gpu][dest_gpu + 1] = table[src_gpu][dest_gpu] + h_table[src_gpu][dest_gpu];
                        v_table[src_gpu + 1][dest_gpu] = table[src_gpu][dest_gpu] + v_table[src_gpu][dest_gpu];

                        const table_t src_index = h_table[src_gpu][dest_gpu];
                        const table_t dest_index = v_table[src_gpu][dest_gpu];
                        const table_t len = table[src_gpu][dest_gpu];

                        key_t* from_k = node_info[src_gpu].src_keys + src_index;
                        key_t* to_k = node_info[dest_gpu].dest_keys + dest_index;
                        cudaMemcpyPeerAsync(to_k, context.get_device_id(dest_gpu),
                            from_k, context.get_device_id(src_gpu),
                            len * sizeof(key_t),
                            context.get_streams(src_gpu)[dest_gpu]);
                    }
                } CUERR;
            }
            return check_tables(node_info, h_table, v_table);
        }


        template <typename key_t, typename value_t, typename index_t, typename table_t>
        bool execKVAsync(const std::array<All2AllNodeInfoT<key_t, value_t, index_t>, NUM_GPUS>& node_info,
            const split_table_tt<table_t, NUM_GPUS>& table) const {  // [src_gpu, partition]

            if (context.is_in_node()) {
                return execKVAsyncInNode(node_info, table);
            }

            // compute prefix sums over the partition table
            std::array<std::array<table_t, num_gpus + 1>, num_gpus> h_table = { {0} }; // horizontal scan
            std::array<std::array<table_t, num_gpus>, num_gpus + 1> v_table = { {0} }; // vertical scan
            // necessary sync because we cant use the stream for communication 
            // context.sync_gpu_default_stream(world_rank());



            ncclComm_t nccl_comm = context.get_nccl();

            ncclGroupStart();
            // RequestPool pool;
            for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                for (uint dest_gpu = 0; dest_gpu < num_gpus; ++dest_gpu) {
                    h_table[src_gpu][dest_gpu + 1] = table[src_gpu][dest_gpu] + h_table[src_gpu][dest_gpu];
                    v_table[src_gpu + 1][dest_gpu] = table[src_gpu][dest_gpu] + v_table[src_gpu][dest_gpu];

                    const table_t len = table[src_gpu][dest_gpu];

                    if (src_gpu == world_rank()) {
                        const table_t src_index = h_table[src_gpu][dest_gpu];
                        key_t* from_k = node_info[src_gpu].src_keys + src_index;
                        value_t* from_v = node_info[src_gpu].src_values + src_index;

                        if (context.get_peer_status(src_gpu, dest_gpu) >= 1) {
                            const table_t dest_index = v_table[src_gpu][dest_gpu];
                            key_t* to_k = node_info[dest_gpu].dest_keys + dest_index;
                            value_t* to_v = node_info[dest_gpu].dest_values + dest_index;
                            cudaMemcpyPeerAsync(to_k, context.get_device_id(dest_gpu),
                                from_k, context.get_device_id(src_gpu),
                                len * sizeof(key_t),
                                context.get_streams(src_gpu)[dest_gpu]);

                            cudaMemcpyPeerAsync(to_v, context.get_device_id(dest_gpu),
                                from_v, context.get_device_id(src_gpu),
                                len * sizeof(value_t),
                                context.get_streams(src_gpu)[dest_gpu]);
                        }
                        else {

                            ncclSend(from_k, sizeof(key_t) * len, ncclChar, dest_gpu, nccl_comm, context.get_streams(src_gpu)[dest_gpu]);

                            ncclSend(from_v, sizeof(value_t) * len, ncclChar, dest_gpu, nccl_comm, context.get_streams(src_gpu)[dest_gpu]);
                        }
                    }
                    if (dest_gpu == world_rank() && context.get_peer_status(src_gpu, dest_gpu) < 1) {
                        const table_t dest_index = v_table[src_gpu][dest_gpu];
                        key_t* to_k = node_info[dest_gpu].dest_keys + dest_index;
                        value_t* to_v = node_info[dest_gpu].dest_values + dest_index;
                        ncclRecv(to_k, sizeof(key_t) * len, ncclChar, src_gpu, nccl_comm, context.get_streams(dest_gpu)[src_gpu]);

                        ncclRecv(to_v, sizeof(value_t) * len, ncclChar, src_gpu, nccl_comm, context.get_streams(dest_gpu)[src_gpu]);
                    }
                } CUERR;
            }
            ncclGroupEnd();
            return check_tables(node_info, h_table, v_table);
        }

        template <typename key_t, typename value_t, typename index_t, typename table_t>
        bool execKVAsyncInNode(const std::array<All2AllNodeInfoT<key_t, value_t, index_t>, NUM_GPUS>& node_info,
            const split_table_tt<table_t, NUM_GPUS>& table) const {  // [src_gpu, partition]
            // compute prefix sums over the partition table
            std::array<std::array<table_t, num_gpus + 1>, num_gpus> h_table = { {0} }; // horizontal scan
            std::array<std::array<table_t, num_gpus>, num_gpus + 1> v_table = { {0} }; // vertical scan
            // necessary sync because we cant use the stream for communication 
            // context.sync_gpu_default_stream(world_rank());

            for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                for (uint dest_gpu = 0; dest_gpu < num_gpus; ++dest_gpu) {
                    if (src_gpu == world_rank()) {
                        h_table[src_gpu][dest_gpu + 1] = table[src_gpu][dest_gpu] + h_table[src_gpu][dest_gpu];
                        v_table[src_gpu + 1][dest_gpu] = table[src_gpu][dest_gpu] + v_table[src_gpu][dest_gpu];

                        const table_t len = table[src_gpu][dest_gpu];

                        const table_t src_index = h_table[src_gpu][dest_gpu];
                        key_t* from_k = node_info[src_gpu].src_keys + src_index;
                        value_t* from_v = node_info[src_gpu].src_values + src_index;
                        const table_t dest_index = v_table[src_gpu][dest_gpu];
                        key_t* to_k = node_info[dest_gpu].dest_keys + dest_index;
                        value_t* to_v = node_info[dest_gpu].dest_values + dest_index;
                        cudaMemcpyPeerAsync(to_k, context.get_device_id(dest_gpu),
                            from_k, context.get_device_id(src_gpu),
                            len * sizeof(key_t),
                            context.get_streams(src_gpu)[dest_gpu]);
                        cudaMemcpyPeerAsync(to_v, context.get_device_id(dest_gpu),
                            from_v, context.get_device_id(src_gpu),
                            len * sizeof(value_t),
                            context.get_streams(src_gpu)[dest_gpu]);
                    }
                } CUERR;
            }
            return check_tables(node_info, h_table, v_table);
        }

        void print_connectivity_matrix() const noexcept {
            context.print_connectivity_matrix();
        }

        void sync() const noexcept {
            context.sync_all_streams();
        }

        void sync_hard() const noexcept {
            context.sync_hard();
        }

    private:

        template <typename key_t, typename value_t, typename index_t, typename table_t>
        bool check_tables(const std::array<All2AllNodeInfoT<key_t, value_t, index_t>, NUM_GPUS>& node_info,
            const std::array<std::array<table_t, num_gpus + 1>, num_gpus>& h_table,
            const std::array<std::array<table_t, num_gpus>, num_gpus + 1>& v_table) const
        {
            // check src_lens for compatibility
            bool valid_srcs_lens = true;
            for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                valid_srcs_lens &= (h_table[src_gpu][num_gpus] <= node_info[src_gpu].src_len);
            }

            if (!valid_srcs_lens) {
                error("srcs_lens not compatible with partition_table.");
            }

            // check dst_lens for compatibility
            bool valid_dsts_lens = true;
            for (uint dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
                // printf("nodeinfo[%u].dest_len: %u, v_table[%u][%u]: %u\n", dst_gpu, node_info[dst_gpu].dest_len, num_gpus, dst_gpu, v_table[num_gpus][dst_gpu]);
                valid_dsts_lens &= v_table[num_gpus][dst_gpu] <= node_info[dst_gpu].dest_len;
            }
            if (!valid_dsts_lens) {
                error("dsts_lens not compatible with partition_table.");
            }
            return true;
        }

    };

}
