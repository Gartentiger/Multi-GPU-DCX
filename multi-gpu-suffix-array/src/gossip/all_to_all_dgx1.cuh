#ifndef ALL2ALLDGX1_H
#define ALL2ALLDGX1_H
#pragma once

#include <type_traits>
#include <array>
#include <vector>

#include "auxiliary.cuh"
#include "context.cuh"
#include "../merge_copy_detour_guide.hpp"

namespace gossip {

    namespace detail {

        using InterNodeCopy = InterNodeCopy<sa_index_t>;
        enum class CopyMode { CopyDirect, FromDetour, ToDetour };

        using CopyAsyncExecutor = std::function<void (uint,         // src node
            uint,         // dest node
            sa_index_t,   // src index
            sa_index_t,   // dest index
            size_t,       // length
            CopyMode,     // direct / from/to detour
            bool)>;       // include values

        class DGX1AllToAllDetourGuide {
            static const uint NUM_GPUS = 8;

            using Context = MultiGPUContext<NUM_GPUS>;

            struct DetourCopy {
                uint detour_node, src_node, dest_node;
                sa_index_t detour_index, src_index, dest_index;
                size_t len;
            };

            Context& mcontext;
            std::array<uint, 8> mreverse_ids;

            static constexpr std::array<uint, NUM_GPUS> consider_nodes_order() { return { 0, 1, 5, 4, 2, 3, 7, 6 }; }
            static const size_t MIN_AVAIL_SPLIT_THRESHOLD = 16 * 1024;

        public:

            DGX1AllToAllDetourGuide(Context& context) noexcept
                : mcontext(context)
            {
                init_reverse_ids_and_check_topology();
            }

            void copy_with_detours_async(const std::array<std::vector<InterNodeCopy>, NUM_GPUS>& copies,
                const std::array<size_t, NUM_GPUS>& detour_buffer_sizes,
                bool include_values, CopyAsyncExecutor async_copy_callback) const noexcept {

                std::array<std::vector<const InterNodeCopy*>, NUM_GPUS> deferred_transfers;

                DetourGuideState detour_state(detour_buffer_sizes, mreverse_ids, mcontext);

                for (uint idx = 0; idx < NUM_GPUS; ++idx) {
                    uint id = consider_nodes_order()[idx];
                    uint node = mreverse_ids[id];
                    //(id);
                    deferred_transfers[node].reserve(3);
                    for (const InterNodeCopy& c : copies[node]) {
                        ASSERT(c.src_node == node);
                        if (c.len == 0)
                            continue;

                        if (mcontext.get_peer_status(c.src_node, c.dest_node) == Context::PEER_STATUS_SLOW) {
                            deferred_transfers[node].push_back(&c);
                        }
                        else {
#ifdef LOAD_BALANCING_LOGS
                            printf("Fast: %u to %u, %zu K\n", mcontext.get_device_id(c.src_node),
                                mcontext.get_device_id(c.dest_node),
                                c.len / 1024);
#endif
                            async_copy_callback(c.src_node, c.dest_node, c.src_index, c.dest_index, c.len,
                                CopyMode::CopyDirect, include_values);
                        }
                    }
                }

                for (uint idx = 0; idx < NUM_GPUS; ++idx) {
                    uint id = consider_nodes_order()[idx];
                    uint node = mreverse_ids[id];
                    //(id);
                    for (const InterNodeCopy* c_ : deferred_transfers[node]) {
                        const InterNodeCopy& c = *c_;
                        ASSERT(get_quad(c.src_node) != get_quad(c.dest_node));
                        DetourGuideState::DetourAllocation detour = detour_state.get_detour_allocation(mcontext.get_device_id(c.src_node),
                            mcontext.get_device_id(c.dest_node),
                            c.len);
                        if (detour.len_through1 > 0) {
                            DetourCopy detour_copy1{ detour.detour_node1,
                                                      c.src_node,
                                                      c.dest_node,
                                                      detour_state.get_detour_offset(detour.detour_node1),
                                                      c.src_index,
                                                      c.dest_index,
                                                      detour.len_through1 };
                            detour_state.register_detour_copy(detour_copy1);
                            emit_copy_to_detour(detour_copy1, async_copy_callback, include_values);
                        }
                        if (detour.len_through2 > 0) {
                            DetourCopy detour_copy2{ detour.detour_node2,
                                                      c.src_node,
                                                      c.dest_node,
                                                      detour_state.get_detour_offset(detour.detour_node2),
                                                      c.src_index + sa_index_t(detour.len_through1),
                                                      c.dest_index + sa_index_t(detour.len_through1),
                                                      detour.len_through2 };
                            detour_state.register_detour_copy(detour_copy2);
                            emit_copy_to_detour(detour_copy2, async_copy_callback, include_values);
                        }
                    }
                }
#ifdef LOAD_BALANCING_LOGS
                detour_state.print_stats();
#endif
                mcontext.sync_all_streams();

                for (uint node = 0; node < NUM_GPUS; ++node) {
                    //(mcontext.get_device_id(node));CUERR;
                    for (const DetourCopy& dc : detour_state.detour_copies()[node]) {
                        emit_copy_from_detour(dc, async_copy_callback, include_values);
                    }
                }
            }

            uint get_reverse_id(uint id) const noexcept {
                return mreverse_ids[id];
            }

        private:

            void emit_copy_to_detour(const DetourCopy& c, const CopyAsyncExecutor& async_copy_callback,
                bool include_values) const noexcept {
                async_copy_callback(c.src_node, c.detour_node, c.src_index, c.detour_index, c.len,
                    CopyMode::ToDetour, include_values);
            }

            void emit_copy_from_detour(const DetourCopy& c, const CopyAsyncExecutor& async_copy_callback,
                bool include_values) const noexcept {
                async_copy_callback(c.detour_node, c.dest_node, c.detour_index, c.dest_index, c.len,
                    CopyMode::FromDetour, include_values);
            }

            inline uint get_quad(uint gpu) const noexcept {
                return mcontext.get_device_id(gpu) / 4;
            }

#define SRC_DEST(src, dest) ( (src)*NUM_GPUS+(dest) )

            void init_reverse_ids_and_check_topology() noexcept {
                for (uint gpu = 0; gpu < NUM_GPUS; ++gpu) {
                    uint id = mcontext.get_device_id(gpu);
                    ASSERT(id < NUM_GPUS);
                    mreverse_ids[id] = gpu;
                }

                bool smaller = mreverse_ids[0] < mreverse_ids[4];

                // Check if the topology is what we expect to avoid stupid mistakes.
                uint FAST = Context::PEER_STATUS_FAST;
                for (uint i = 0; i < 4; ++i) {
                    for (uint j = 0; j < i; ++j) {
                        // Within quads should be fast.
                        ASSERT(mcontext.get_peer_status(mreverse_ids[i], mreverse_ids[j]) == FAST);
                        ASSERT(mcontext.get_peer_status(mreverse_ids[4 + i], mreverse_ids[4 + j]) == FAST);
                    }
                    // Inter-quad connections.
                    ASSERT(mcontext.get_peer_status(mreverse_ids[i], mreverse_ids[i + 4]) == FAST);
                    // Quads grouped together.
                    ASSERT(smaller ? mreverse_ids[i] < mreverse_ids[4 + i] : mreverse_ids[i] > mreverse_ids[4 + i]);
                }
            }

            class DetourGuideState {
            public:
                DetourGuideState(const std::array<size_t, NUM_GPUS>& detour_buffer_sizes_,
                    const std::array<uint, 8>& reverse_ids,
                    const Context& context /* debugging */) noexcept
                    : mdetour_offsets{ 0 },
                    mdetour_buffer_sizes(detour_buffer_sizes_), mcontext(context),
                    mreverse_ids(reverse_ids)
                {
                    for (size_t i = 0; i < NUM_GPUS; ++i) {
                        mdetour_copies[i].reserve(4);
                    }
                }

                void register_detour_copy(const DetourCopy& copy) noexcept {
                    mdetour_copies[copy.detour_node].push_back(copy);
                    mdetour_offsets[copy.detour_node] += copy.len;
                }

                struct DetourAllocation {
                    uint detour_node1;
                    uint detour_node2;
                    size_t len_through1, len_through2;
                };

                inline DetourAllocation get_detour_allocation(uint src_id, uint dest_id, size_t len) noexcept {
                    uint detour_node1 = (src_id + 4) % NUM_GPUS;
                    uint detour_node2 = (dest_id + 4) % NUM_GPUS;

                    uint detour_node1_tr = mreverse_ids[detour_node1];
                    uint detour_node2_tr = mreverse_ids[detour_node2];

                    size_t avail_detour_node1 = get_available_detour_space(detour_node1_tr);
                    size_t avail_detour_node2 = get_available_detour_space(detour_node2_tr);
                    size_t max_len1 = std::min(len, avail_detour_node1);
                    size_t max_len2 = std::min(len, avail_detour_node2);
                    bool first = true;

                    if (src_id == 2) {
                        if (dest_id == 5 || dest_id == 7)
                            first = false;
                    }

                    if (src_id == 3) {
                        if (dest_id == 4 || dest_id == 6)
                            first = false;
                    }

                    if (src_id == 6) {
                        if (dest_id == 1 || dest_id == 3)
                            first = false;
                    }

                    if (src_id == 7) {
                        if (dest_id == 0 || dest_id == 2)
                            first = false;
                    }

                    if (first) {
                        return DetourAllocation{ detour_node1_tr, detour_node2_tr,
                                                  max_len1, len - max_len1 };
                    }
                    else {
                        return DetourAllocation{ detour_node1_tr, detour_node2_tr,
                                                  len - max_len2, max_len2 };
                    }
                }

                size_t get_available_detour_space(uint node_index) const noexcept {
                    return mdetour_buffer_sizes[node_index] - mdetour_offsets[node_index];
                }

                sa_index_t get_detour_offset(uint node_index) const noexcept {
                    return mdetour_offsets[node_index];
                }

                const std::array<std::vector<DetourCopy>, NUM_GPUS>& detour_copies() const noexcept {
                    return mdetour_copies;
                }

            private:

                const std::array<size_t, NUM_GPUS>& mdetour_buffer_sizes;
                const Context& mcontext;
                const std::array<uint, 8>& mreverse_ids;

                std::array<sa_index_t, NUM_GPUS> mdetour_offsets;
                std::array<std::vector<DetourCopy>, NUM_GPUS> mdetour_copies;
            };
        };

    }


    template<uint NUM_GPUS, bool THROW_EXCEPTIONS = true>
    class All2AllDGX1 {
        using Context = MultiGPUContext<NUM_GPUS>;
        using device_id_t = typename Context::device_id_t;
        Context& context;
        detail::DGX1AllToAllDetourGuide mcopy_detour_guide;

    public:
        static const uint num_gpus = NUM_GPUS;

        All2AllDGX1(Context& context_)
            : context(context_), mcopy_detour_guide(context_)
        {
        }

        template <typename key_t, typename value_t, typename index_t, typename table_t>
        bool execAsync(const std::array<All2AllNodeInfoT<key_t, value_t, index_t>, NUM_GPUS>& node_info,
            const split_table_tt<table_t, NUM_GPUS>& table) const noexcept {

            using InterNodeCopy = InterNodeCopy<index_t>;
            std::array<std::vector<InterNodeCopy>, NUM_GPUS> copies;

            bool valid = check_and_schedule_copies(node_info, table, copies);
            if (!valid)
                return false;


            std::array<size_t, num_gpus> detour_buffer_sizes;
            for (uint i = 0; i < num_gpus; ++i) {
                detour_buffer_sizes[i] = node_info[i].temp_len;
            }

            mcopy_detour_guide.copy_with_detours_async(copies, detour_buffer_sizes, false,
                [&](uint src_node, uint dest_node,
                    index_t src_index, index_t dest_index,
                    size_t len, detail::CopyMode mode,
                    bool do_values) noexcept {
                        // Assume src device is active.
                        key_t* src_buff;
                        key_t* dest_buff;
                        if (mode == detail::CopyMode::FromDetour) {
                            src_buff = node_info[src_node].temp_keys;
                            dest_buff = node_info[dest_node].dest_keys;
                        }
                        else if (mode == detail::CopyMode::ToDetour) {
                            src_buff = node_info[src_node].src_keys;
                            dest_buff = node_info[dest_node].temp_keys;
                        }
                        else { // Direct
                            src_buff = node_info[src_node].src_keys;
                            dest_buff = node_info[dest_node].dest_keys;
                        }

                        cudaMemcpyPeerAsync(dest_buff + dest_index, context.get_device_id(dest_node),
                            src_buff + src_index, context.get_device_id(src_node),
                            len * sizeof(key_t),
                            context.get_streams(src_node)[dest_node]);CUERR;
                        ASSERT(!do_values);
                });


            return true;
        }

        template <typename key_t, typename value_t, typename index_t, typename table_t>
        bool execKVAsync(const std::array<All2AllNodeInfoT<key_t, value_t, index_t>, NUM_GPUS>& node_info,
            const split_table_tt<table_t, NUM_GPUS>& table) const noexcept {  // [src_gpu, partition]

            using InterNodeCopy = InterNodeCopy<index_t>;
            std::array<std::vector<InterNodeCopy>, NUM_GPUS> copies;

            bool valid = check_and_schedule_copies(node_info, table, copies);
            if (!valid)
                return false;

            std::array<size_t, num_gpus> detour_buffer_sizes;
            for (uint i = 0; i < num_gpus; ++i)
                detour_buffer_sizes[i] = node_info[i].temp_len;

            mcopy_detour_guide.copy_with_detours_async(copies, detour_buffer_sizes, true,
                [&](uint src_node, uint dest_node,
                    index_t src_index, index_t dest_index,
                    size_t len, detail::CopyMode mode,
                    bool do_values) noexcept {
                        // Assume src device is active.
                        key_t* src_buff;
                        key_t* dest_buff;
                        if (mode == detail::CopyMode::FromDetour) {
                            src_buff = node_info[src_node].temp_keys;
                            dest_buff = node_info[dest_node].dest_keys;
                        }
                        else if (mode == detail::CopyMode::ToDetour) {
                            src_buff = node_info[src_node].src_keys;
                            dest_buff = node_info[dest_node].temp_keys;
                        }
                        else { // Direct
                            src_buff = node_info[src_node].src_keys;
                            dest_buff = node_info[dest_node].dest_keys;
                        }

                        cudaMemcpyPeerAsync(dest_buff + dest_index, context.get_device_id(dest_node),
                            src_buff + src_index, context.get_device_id(src_node),
                            len * sizeof(key_t),
                            context.get_streams(src_node)[dest_node]);CUERR;
                        ASSERT(do_values);
                        {
                            value_t* src_buff;
                            value_t* dest_buff;
                            if (mode == detail::CopyMode::FromDetour) {
                                src_buff = node_info[src_node].temp_values;
                                dest_buff = node_info[dest_node].dest_values;
                            }
                            else if (mode == detail::CopyMode::ToDetour) {
                                src_buff = node_info[src_node].src_values;
                                dest_buff = node_info[dest_node].temp_values;
                            }
                            else { // Direct
                                src_buff = node_info[src_node].src_values;
                                dest_buff = node_info[dest_node].dest_values;
                            }
                            cudaMemcpyPeerAsync(dest_buff + dest_index, context.get_device_id(dest_node),
                                src_buff + src_index, context.get_device_id(src_node),
                                len * sizeof(value_t),
                                context.get_streams(src_node)[dest_node]);CUERR;
                        }
                });
            return true;
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
        bool check_and_schedule_copies(const std::array<All2AllNodeInfoT<key_t, value_t, index_t>, NUM_GPUS>& node_info,
            const split_table_tt<table_t, NUM_GPUS>& table,
            std::array<std::vector<InterNodeCopy<index_t>>, NUM_GPUS>& out_copies) const noexcept
        {
            // compute prefix sums over the partition table
            std::array<std::array<table_t, num_gpus + 1>, num_gpus> h_table = { {0} }; // horizontal scan
            std::array<std::array<table_t, num_gpus>, num_gpus + 1> v_table = { {0} }; // vertical scan

            for (uint gpu = 0; gpu < num_gpus; ++gpu) {
                out_copies[gpu].reserve(num_gpus);
                for (uint part = 0; part < num_gpus; ++part) {
                    h_table[gpu][part + 1] = table[gpu][part] + h_table[gpu][part];
                    v_table[gpu + 1][part] = table[gpu][part] + v_table[gpu][part];
                    out_copies[gpu].push_back({ gpu, part,
                                                h_table[gpu][part],
                                                v_table[gpu][part],
                                                table[gpu][part] });
                }
            }

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
                valid_dsts_lens &= v_table[num_gpus][dst_gpu] <= node_info[dst_gpu].dest_len;
            }
            if (!valid_dsts_lens) {
                error("dsts_lens not compatible with partition_table.");
            }
            return true;
        }
    };

}
#endif // ALL2ALLDGX1_H
