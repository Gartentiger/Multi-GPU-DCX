#ifndef GPU_TOPOLOGY_HPP
#define GPU_TOPOLOGY_HPP

#include <functional>
#include <algorithm>

#include "gossip/context.cuh"
#include "suffix_types.h"

//#define LOAD_BALANCING_LOGS

namespace mergeCopying {

    using InterNodeCopy = InterNodeCopy<sa_index_t>;
    enum class CopyMode { CopyDirect, FromDetour, ToDetour };

    using CopyAsyncExecutor = std::function<void (uint,         // src node
        uint,         // dest node
        sa_index_t,   // src index
        sa_index_t,   // dest index
        size_t,       // length
        int,          // extra information (from InterNodeCopy)
        CopyMode,     // direct / from/to detour
        bool)>;       // include values

    // If fully connected.
    template <uint NUM_GPUS>
    struct MergeDummyCopyDetourGuide {
    public:

        MergeDummyCopyDetourGuide(const MultiGPUContext<NUM_GPUS>& context)
        {
            (void)context;
        }

        void copy_with_detours_async(const std::array<std::vector<InterNodeCopy>, NUM_GPUS>& copies,
            const std::array<size_t, NUM_GPUS>& detour_buffer_sizes,
            bool include_values, CopyAsyncExecutor async_copy_callback) const;
    };


    class MergeDGX1CopyDetourGuide {
        static const uint NUM_GPUS = 8;

        using Context = MultiGPUContext<NUM_GPUS>;

        struct DetourCopy {
            uint detour_node, src_node, dest_node;
            sa_index_t detour_index, src_index, dest_index;
            size_t len;
            int extra;
        };

        Context& mcontext;
        std::array<uint, 8> mreverse_ids;

        static constexpr std::array<uint, NUM_GPUS> consider_nodes_order() { return { 0, 1, 5, 4, 2, 3, 7, 6 }; }
        static const size_t MIN_AVAIL_SPLIT_THRESHOLD = 16 * 1024;

    public:

        MergeDGX1CopyDetourGuide(Context& context)
            : mcontext(context)
        {
            init_reverse_ids_and_check_topology();
        }

        void copy_with_detours_async(const std::array<std::vector<InterNodeCopy>, NUM_GPUS>& copies,
            const std::array<size_t, NUM_GPUS>& detour_buffer_sizes,
            bool include_values, CopyAsyncExecutor async_copy_callback) const {

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
                        detour_state.register_copy_for_stats(c);
                        async_copy_callback(c.src_node, c.dest_node, c.src_index, c.dest_index, c.len, c.extra,
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

        uint get_reverse_id(uint id) const {
            return mreverse_ids[id];
        }

    private:

        void emit_copy_to_detour(const DetourCopy& c, const CopyAsyncExecutor& async_copy_callback,
            bool include_values) const {
            async_copy_callback(c.src_node, c.detour_node, c.src_index, c.detour_index, c.len, c.extra,
                CopyMode::ToDetour, include_values);
        }

        void emit_copy_from_detour(const DetourCopy& c, const CopyAsyncExecutor& async_copy_callback,
            bool include_values) const {
            async_copy_callback(c.detour_node, c.dest_node, c.detour_index, c.dest_index, c.len, c.extra,
                CopyMode::FromDetour, include_values);
        }

        inline uint get_pair_node_from_other_quad(uint node) const {
            return mreverse_ids[(mcontext.get_device_id(node) + 4) % NUM_GPUS];
        }

        inline uint get_quad(uint gpu) const {
            return mcontext.get_device_id(gpu) / 4;
        }

#define SRC_DEST(src, dest) ( (src)*NUM_GPUS+(dest) )

        void init_reverse_ids_and_check_topology() {
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
                const Context& context /* debugging */)
                : mdetour_offsets{ 0 },
                mdetour_buffer_sizes(detour_buffer_sizes_), mcontext(context),
                mreverse_ids(reverse_ids), max_second_transfer_load(0)
            {
                for (size_t i = 0; i < NUM_GPUS; ++i) {
                    mdetour_copies[i].reserve(4);

                    if (i < 4) {
                        mfirst_transfer_stats[i] = 0;
                        msecond_transfer_stats[i] = 0;
                    }

                    for (size_t j = 0; j < NUM_GPUS; ++j)
                        mstats[i][j] = { 0, 0, 0 };

                }
            }

            void register_copy_for_stats(const InterNodeCopy& cpy) {
                uint src_id = mcontext.get_device_id(cpy.src_node);
                uint dest_id = mcontext.get_device_id(cpy.dest_node);

                mstats[src_id][dest_id].direct += cpy.len;

                uint transfer_index = inter_quad_transfer_index(src_id, dest_id);
                if (transfer_index < NUM_GPUS) {
                    ASSERT(transfer_index < NUM_GPUS / 2);
                    mfirst_transfer_stats[transfer_index] += cpy.len;
                }
            }

            void register_detour_copy(const DetourCopy& copy) {
                mdetour_copies[copy.detour_node].push_back(copy);
                mdetour_offsets[copy.detour_node] += copy.len;

                uint src_id = mcontext.get_device_id(copy.src_node);
                uint detour_id = mcontext.get_device_id(copy.detour_node);
                uint dest_id = mcontext.get_device_id(copy.dest_node);

                mstats[src_id][detour_id].to_detour += copy.len;
                mstats[detour_id][dest_id].from_detour += copy.len;

                uint first_transfer_index = inter_quad_transfer_index(src_id, detour_id);
                uint second_transfer_index = inter_quad_transfer_index(detour_id, dest_id);

                if (first_transfer_index < NUM_GPUS) {
                    ASSERT(first_transfer_index < 4);
                    mfirst_transfer_stats[first_transfer_index] += copy.len;
                    size_t for_max = mfirst_transfer_stats[first_transfer_index];
                    if (first_transfer_index >= 2)
                        for_max *= 2;
                }
                if (second_transfer_index < NUM_GPUS) {
                    ASSERT(second_transfer_index < 4);
                    msecond_transfer_stats[second_transfer_index] += copy.len;
                    size_t for_max = msecond_transfer_stats[second_transfer_index];
                    if (second_transfer_index >= 2)
                        for_max *= 2;
                    max_second_transfer_load = std::max(max_second_transfer_load, for_max);
                }
            }

            static bool does_id_belong_to_first_row(uint id) {
                return id == 0 || id == 1 || id == 4 || id == 5;
            }

            struct DetourAllocation {
                uint detour_node1;
                uint detour_node2;
                size_t len_through1, len_through2;
            };
            inline DetourAllocation get_detour_allocation(uint src_id, uint dest_id, size_t len) {
                uint detour_node1 = (src_id + 4) % NUM_GPUS;
                uint detour_node2 = (dest_id + 4) % NUM_GPUS;

                uint detour_node1_tr = mreverse_ids[detour_node1];
                uint detour_node2_tr = mreverse_ids[detour_node2];

                size_t avail_detour_node1 = get_available_detour_space(detour_node1_tr);
                size_t avail_detour_node2 = get_available_detour_space(detour_node2_tr);

                uint first_transfer_index = inter_quad_transfer_index(src_id, detour_node1);
                uint second_transfer_index = inter_quad_transfer_index(detour_node2, dest_id);

                ASSERT(first_transfer_index < 4 && second_transfer_index < 4);

                size_t detour1_load = mfirst_transfer_stats[first_transfer_index];
                size_t detour2_load = msecond_transfer_stats[second_transfer_index];

                size_t corrected_detour1_load = detour1_load;
                size_t corrected_detour2_load = detour2_load;

                if (first_transfer_index >= 2)
                    corrected_detour1_load *= 2;

                if (second_transfer_index >= 2)
                    corrected_detour2_load *= 2;

                size_t max_len1 = std::min(len, avail_detour_node1);
                size_t max_len2 = std::min(len, avail_detour_node2);

                //                    bool first = true;

                //                    if (src_id == 2) {
                //                        if (dest_id == 5 || dest_id == 7)
                //                            first = false;
                //                    }

                //                    if (src_id == 3) {
                //                        if (dest_id == 4 || dest_id == 6)
                //                            first = false;
                //                    }

                //                    if (src_id == 6) {
                //                        if (dest_id == 1 || dest_id == 3)
                //                            first = false;
                //                    }

                //                    if (src_id == 7) {
                //                        if (dest_id == 0 || dest_id == 2)
                //                            first = false;
                //                    }

                //                    if (first) {
                //                        return DetourAllocation { detour_node1_tr, detour_node2_tr,
                //                                                  max_len1, len-max_len1 };
                //                    }
                //                    else {
                //                        return DetourAllocation { detour_node1_tr, detour_node2_tr,
                //                                                  len - max_len2, max_len2 };
                //                    }


#ifdef LOAD_BALANCING_LOGS
                printf("Transfer %7zu K from %u to %u, routing through %u (load %7zu, corr: %7zu) or %u (load %7zu, corr: %7zu).",
                    len / 1024, src_id, dest_id,
                    detour_node1, detour1_load / 1024, corrected_detour1_load / 1024,
                    detour_node2, detour2_load / 1024, corrected_detour2_load / 1024);
#endif

                if (corrected_detour2_load < corrected_detour1_load) {
                    // Push all available to 2; if 2637 detour, limit through max of existing transfers
                    // when transferring from 0154.
                    ASSERT(len - max_len2 <= max_len1);
                    size_t limit_len = max_len2;
                    if (does_id_belong_to_first_row(src_id) && second_transfer_index >= 2) {
                        limit_len = max_second_transfer_load - corrected_detour2_load;
                        limit_len /= 2;
                        limit_len = std::min(limit_len, len);
#ifdef LOAD_BALANCING_LOGS
                        printf("Want to transfer %zu K over %u. Detour corrected load is %zu K with max transfer load %zu K. limit len is %zu K\n",
                            max_len2 / 1024, detour_node2, corrected_detour2_load / 1024, max_second_transfer_load / 1024,
                            limit_len / 1024);
#endif
                        limit_len = std::max(limit_len, len - max_len1);
                    }

#ifdef LOAD_BALANCING_LOGS
                    printf(" Routing %zu K through %u and %zu K through %u.\n",
                        (len - limit_len) / 1024, detour_node1, limit_len / 1024, detour_node2);
#endif

                    return DetourAllocation{ detour_node1_tr, detour_node2_tr,
                                              len - limit_len, limit_len };
                }
                else {
                    // Push all available to 1.
                    ASSERT(len - max_len1 <= max_len2);

#ifdef LOAD_BALANCING_LOGS
                    printf(" Routing %zu K through %u and %zu K through %u.\n",
                        max_len1 / 1024, detour_node1, (len - max_len1) / 1024, detour_node2);
#endif
                    return DetourAllocation{ detour_node1_tr, detour_node2_tr,
                                              max_len1, len - max_len1 };
                }

            }

            inline uint inter_quad_transfer_index(uint src_id, uint dest_id) const {
                if (dest_id == ((src_id + 4) % NUM_GPUS)) {
                    if (dest_id < 4)
                        return dest_id;
                    else
                        return src_id;
                }
                return NUM_GPUS;
            }

            size_t get_available_detour_space(uint node_index) const {
                return mdetour_buffer_sizes[node_index] - mdetour_offsets[node_index];
            }

            sa_index_t get_detour_offset(uint node_index) const {
                return mdetour_offsets[node_index];
            }

            void print_stats() const {
                const std::vector<std::pair<uint, uint>> SINGLE_CONNS{ {0, 1}, {0, 2}, {1, 3},
                                                                        {4, 5}, {4, 6}, {5, 7},
                                                                        {2, 6}, {3, 7} };
                size_t balanced_max_1 = 0, balanced_max_2 = 0;
                for (uint i = 0; i < NUM_GPUS; ++i) {
                    for (uint j = i + 1; j < NUM_GPUS; ++j) {
                        bool single = std::find(SINGLE_CONNS.begin(), SINGLE_CONNS.end(), std::make_pair(i, j)) != SINGLE_CONNS.end();
                        const TransferStat& c1 = mstats[i][j];
                        const TransferStat& c2 = mstats[j][i];
                        const TransferStat conn{ c1.direct + c2.direct,
                                                  c1.to_detour + c2.to_detour,
                                                  c1.from_detour + c2.from_detour };
                        printf("Over %u-%u (%s): %7zu K direct, %7zu K to detour -> %7zu K first, %7zu K from detour.\n",
                            i, j, single ? "single" : "double",
                            conn.direct / 1024, conn.to_detour / 1024,
                            (conn.direct + conn.to_detour) / 1024,
                            conn.from_detour / 1024);
                        size_t phase_1 = conn.direct + conn.to_detour;
                        size_t phase_2 = conn.from_detour;
                        if (!single) {
                            phase_1 /= 2;
                            phase_2 /= 2;
                        }
                        balanced_max_1 = std::max(balanced_max_1, phase_1);
                        balanced_max_2 = std::max(balanced_max_2, phase_2);
                    }
                }
                printf("Balanced max: phase 1: %zu K, phase 2: %zu K\n", balanced_max_1 / 1024,
                    balanced_max_2 / 1024);
            }

            const std::array<std::vector<DetourCopy>, NUM_GPUS>& detour_copies() const {
                return mdetour_copies;
            }

        private:

            const std::array<size_t, NUM_GPUS>& mdetour_buffer_sizes;
            const Context& mcontext;
            const std::array<uint, 8>& mreverse_ids;

            std::array<sa_index_t, NUM_GPUS> mdetour_offsets;
            std::array<std::vector<DetourCopy>, NUM_GPUS> mdetour_copies;


            struct TransferStat {
                size_t direct;
                size_t to_detour;
                size_t from_detour;
            };

            std::array<std::array<TransferStat, NUM_GPUS>, NUM_GPUS> mstats;
            std::array<size_t, NUM_GPUS / 2> mfirst_transfer_stats; // 0: 0-4, 1: 1-5, 2: 2-6, 3: 37
            std::array<size_t, NUM_GPUS / 2> msecond_transfer_stats;
            size_t max_second_transfer_load;
        };
    };


}

#endif // GPU_TOPOLOGY_HPP
