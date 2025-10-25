#pragma once

#include <type_traits>
#include <array>
#include <vector>

#include "context.cuh"
#include "auxiliary.cuh"
#include "util.h"

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
            auto& t = kamping::measurements::timer();
            t.start("alltoall");
            // compute prefix sums over the partition table
            std::array<std::array<table_t, num_gpus + 1>, num_gpus> h_table = { {0} }; // horizontal scan
            std::array<std::array<table_t, num_gpus>, num_gpus + 1> v_table = { {0} }; // vertical scan

            for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                for (uint dest_gpu = 0; dest_gpu < num_gpus; ++dest_gpu) {
                    h_table[src_gpu][dest_gpu + 1] = table[src_gpu][dest_gpu] + h_table[src_gpu][dest_gpu];
                    v_table[src_gpu + 1][dest_gpu] = table[src_gpu][dest_gpu] + v_table[src_gpu][dest_gpu];
                    // printf("[%u][%u]: h_table: %u, v_table: %u, table: %u, next h_table: %u, next v_table: %u\n", src_gpu, dest_gpu, h_table[src_gpu][dest_gpu], v_table[src_gpu][dest_gpu], table[src_gpu][dest_gpu], h_table[src_gpu][dest_gpu + 1], v_table[src_gpu][dest_gpu + 1]);

                    const table_t src_index = h_table[src_gpu][dest_gpu];
                    const table_t dest_index = v_table[src_gpu][dest_gpu];
                    const table_t len = table[src_gpu][dest_gpu];

                    key_t* from_k = node_info[src_gpu].src_keys + src_index;
                    key_t* to_k = node_info[dest_gpu].dest_keys + dest_index;

                    cudaMemcpyPeerAsync(to_k, context.get_device_id(dest_gpu),
                        from_k, context.get_device_id(src_gpu),
                        len * sizeof(key_t),
                        context.get_streams(src_gpu)[dest_gpu]);
                } CUERR;
            }
            t.stop_and_add();
            return check_tables(node_info, h_table, v_table);
        }

        template <typename key_t, typename value_t, typename index_t, typename table_t>
        bool execKVAsync(const std::array<All2AllNodeInfoT<key_t, value_t, index_t>, NUM_GPUS>& node_info,
            const split_table_tt<table_t, NUM_GPUS>& table) const {  // [src_gpu, partition]
            auto& t = kamping::measurements::timer();
            t.start("alltoall");
            // compute prefix sums over the partition table
            std::array<std::array<table_t, num_gpus + 1>, num_gpus> h_table = { {0} }; // horizontal scan
            std::array<std::array<table_t, num_gpus>, num_gpus + 1> v_table = { {0} }; // vertical scan

            for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                for (uint dest_gpu = 0; dest_gpu < num_gpus; ++dest_gpu) {
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

                    value_t* from_v = node_info[src_gpu].src_values + src_index;
                    value_t* to_v = node_info[dest_gpu].dest_values + dest_index;

                    cudaMemcpyPeerAsync(to_v, context.get_device_id(dest_gpu),
                        from_v, context.get_device_id(src_gpu),
                        len * sizeof(value_t),
                        context.get_streams(src_gpu)[dest_gpu]);
                } CUERR;
            }
            t.stop_and_add();
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
