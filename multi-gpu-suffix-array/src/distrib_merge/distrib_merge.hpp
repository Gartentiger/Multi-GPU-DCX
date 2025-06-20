#ifndef DISTRIB_MERGE_HPP
#define DISTRIB_MERGE_HPP

#include <iostream>

#include <array>
#include <vector>
#include<algorithm>
#include <cassert>
#include <moderngpu/kernel_merge.hxx>

#include "../gossip/auxiliary.cuh"
#include "../gossip/context.cuh"

#include "distrib_merge_array.hpp"
#include "distrib_merge_topology_helper.hpp"
#include "util.h"
#include <kamping/data_buffer.hpp>
#include <kamping/checking_casts.hpp>
#include <kamping/named_parameters.hpp>
#include <kamping/collectives/allgather.hpp>
#include <span>

namespace distrib_merge {

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
        *store_result = merge_path(a_keys, a_count, b_keys, b_count, diag, comp);
    }


    /* Class for merging two arrays that are distributed across multiple GPUs to an array
     * that is also distributed across multiple GPUs. The number of output elements on each GPU
     * is independent from the number of input elements on each GPU, however, the sum of of all
     * input elements must match the number of output elements.
     *
     * You have to supply a temporary bufffer alongside the output buffer that is of equal size
     * for each node. Input and output buffers may point to the same memory, but the temporary
     * buffer must not.
     *
     * You also have to initialize the context's temporary allocators to another, separate
     * buffer that is used by ModernGPU's merge for storing partioning information.
     * It can be much smaller.
     *
     * TODO: separate APIs for key and key+value merge.
     */
    template <typename key_t, typename value_t, typename index_t, size_t NUM_NODES,
        template<typename, typename, typename, size_t> class TopologyHelperT>
    class DistributedMerge {
    public:
        using DistrArray = DistributedArray<key_t, value_t, index_t, NUM_NODES>;
    private:
        using Context = MultiGPUContext<NUM_NODES>;
        using TopologyHelper = TopologyHelperT<key_t, value_t, index_t, NUM_NODES>;

        Context& mcontext;
        TopologyHelper mtopology_helper;
        QDAllocator& mhost_search_temp_allocator;

        const DistrArray& minp_a;
        const DistrArray& minp_b;
        DistrArray& mout;

    public:

        template <class comp_f>
        static void merge_async(const DistrArray& a, const DistrArray& b, DistrArray& out, comp_f comp,
            bool do_values, Context& context, QDAllocator& host_search_temp_allocator) {
            DistributedMerge dm(context, host_search_temp_allocator, a, b, out);
            dm.merge(comp, do_values);

        }

    private:
        using idx_t = int64_t;
        struct Coord3 {
            idx_t inp_a_start, inp_b_start;
            idx_t output_start;
        };

        struct Search {
            uint node_a;
            uint node_b;
            int64_t cross_diagonal;
            uint scheduled_on;
            int64_t result;
            int64_t* d_result_ptr;
            int64_t* h_result_ptr;
        };

        struct MergePosition {
            uint node;
            index_t index;
        };

        struct MergePartition {
            uint dest_node;
            size_t size_from_a, size_from_b;
            struct MergePartitionSource {
                uint src_node;
                index_t src_index;
                index_t count;
                index_t dest_offset;
            };
            std::vector<MergePartitionSource> a_sources, b_sources;
        };

        DistributedMerge(Context& context, QDAllocator& host_search_temp_allocator,
            const DistrArray& a, const DistrArray& b, DistrArray& out)
            : mcontext(context), mtopology_helper(context),
            mhost_search_temp_allocator(host_search_temp_allocator),
            minp_a(a), minp_b(b), mout(out)
        {
        }

        template <class comp_f>
        void merge(comp_f comp, bool do_values) {
            std::array<std::vector<Search>, NUM_NODES - 1> searches = plan_searches();

            execute_searches(searches, comp);

            std::array<std::pair<MergePosition, MergePosition>, NUM_NODES - 1> partition_points =
                create_partition_points_from_search_results(searches);

            std::array<MergePartition, NUM_NODES> partitions = make_partitions(searches);

            std::array<std::vector<InterNodeCopy<index_t>>, NUM_NODES> copies;

            for (uint i = 0; i < NUM_NODES; ++i)
                copies[i].reserve(2 * NUM_NODES);

            for (const MergePartition& p : partitions) {
                for (const typename MergePartition::MergePartitionSource& s : p.a_sources) {
                    if (s.count > 0)
                        copies[s.src_node].push_back({ s.src_node, p.dest_node, s.src_index, s.dest_offset,
                                                       s.count, 0 });
                }
                for (const typename MergePartition::MergePartitionSource& s : p.b_sources) {
                    if (s.count > 0)
                        copies[s.src_node].push_back({ s.src_node, p.dest_node, s.src_index, (index_t)p.size_from_a + s.dest_offset,
                                                       s.count, 1 });
                }
            }

            mtopology_helper.do_copies_async(copies, minp_a, minp_b, mout, do_values);

            mcontext.sync_all_streams();

            execute_merges_async(partitions, comp, do_values);
        }


        template <class comp_f>
        void execute_searches(std::array<std::vector<Search>, NUM_NODES - 1>& searches, comp_f comp) {
            // Assign to nodes
            std::array<std::vector<Search*>, NUM_NODES> searches_on_nodes;
            for (auto& cd_searches : searches) {
                for (Search& search : cd_searches) {
                    searches_on_nodes[search.scheduled_on].push_back(&search);
                }
            }
            // printf("[%lu]---------------------------------------------------------------\n", world_rank());
            for (uint node = 0; node < NUM_NODES; ++node)
            {
                //uint node = world_rank();
                QDAllocator& d_alloc = mcontext.get_device_temp_allocator(node);

                if (node != world_rank()) {
                    int i = 0;
                    for (Search* s : searches_on_nodes[node]) {
                        ASSERT(!(s->node_a == world_rank() && s->node_b == world_rank()));
                        if (s->node_a == world_rank()) {
                            const auto& node_a = minp_a[s->node_a];
                            std::span<key_t> sb(node_a.keys, size_t(node_a.count));
                            comm_world().isend(send_buf(sb), send_count(size_t(node_a.count)), tag(i), destination((size_t)s->node_b));
                        }
                        else if (s->node_b == world_rank()) {
                            const auto& node_b = minp_b[s->node_b];
                            std::span<key_t> sb(node_b.keys, size_t(node_b.count));
                            comm_world().isend(send_buf(sb), send_count(size_t(node_b.count)), tag(i), destination((size_t)s->node_a));
                        }
                        i++;
                    }
                    // printf("[%lu] sends done\n", world_rank());
                }
                else {

                    // cudaSetDevice(mcontext.get_device_id(node));
                    int i = 0;
                    for (Search* s : searches_on_nodes[node])
                    {
                        ASSERT(s->node_a == world_rank() || s->node_b == world_rank());

                        uint other = (node == s->node_a) ? s->node_b : s->node_a;
                        const cudaStream_t& stream = mcontext.get_streams(node).at(other);

                        const auto& node_a = minp_a[s->node_a];
                        const auto& node_b = minp_b[s->node_b];

                        s->d_result_ptr = d_alloc.get<int64_t>(1);


                        key_t* temp;
                        if (node == s->node_a) {
                            // cudaMalloc could be needed here
                            // temp = (key_t*)d_alloc.get_raw(sizeof(key_t) * size_t(node_b.count));
                            cudaMalloc(&temp, sizeof(key_t) * size_t(node_b.count));

                            std::span<key_t> rb(temp, size_t(node_b.count));
                            comm_world().recv(recv_buf(rb), tag(i), recv_count(size_t(node_b.count)));
                            run_partitioning_search << <1, 1, 0, stream >> > (node_a.keys,
                                int64_t(node_a.count),
                                temp,
                                int64_t(node_b.count),
                                s->cross_diagonal,
                                comp,
                                s->d_result_ptr);
                            CUERR;
                        }
                        else {
                            // cudaMalloc could be needed here
                            // temp = (key_t*)d_alloc.get_raw(sizeof(key_t) * size_t(node_a.count));
                            cudaMalloc(&temp, sizeof(key_t) * size_t(node_a.count));

                            std::span<key_t> rb(temp, size_t(node_a.count));
                            comm_world().recv(recv_buf(rb), tag(i), recv_count(size_t(node_a.count)));
                            run_partitioning_search << <1, 1, 0, stream >> > (temp,
                                int64_t(node_a.count),
                                node_b.keys,
                                int64_t(node_b.count),
                                s->cross_diagonal,
                                comp,
                                s->d_result_ptr);
                            CUERR;
                        }
                        // printf("[%lu] recv, i: %d \n", world_rank(), i);
                        s->h_result_ptr = mhost_search_temp_allocator.get<int64_t>(1);

                        cudaMemcpyAsync(s->h_result_ptr, s->d_result_ptr,
                            sizeof(int64_t), cudaMemcpyDeviceToHost, stream);CUERR;
                        cudaFreeAsync(temp, stream);
                        i++;
                    }
                    // printf("[%lu] recv done\n", world_rank());
                }
            }
            mcontext.sync_all_streams();
            std::vector<int64_t> hResultsIn;
            hResultsIn.clear();
            for (Search* s : searches_on_nodes[world_rank()]) {
                hResultsIn.push_back(*s->h_result_ptr);
            }
            std::vector<int64_t> hResultsOut;
            hResultsOut.clear();
            comm_world().allgatherv(send_buf(hResultsIn), send_count(hResultsIn.size()), recv_buf<resize_to_fit>(hResultsOut));
            // printf("[%lu] allgatherv distributed_merge done\n", world_rank());
            int i = 0;
            for (uint node = 0; node < NUM_NODES; ++node) {
                for (Search* s : searches_on_nodes[node]) {
                    s->result = hResultsOut[i++];
                }
            }
        }

        template <class comp_f>
        void execute_merges_async(const std::array<MergePartition, NUM_NODES>& partitions, comp_f comp,
            bool do_values) {
            //for (uint node = 0; node < NUM_NODES; ++node) 
                {
                    uint node = world_rank();
                    const MergePartition& p = partitions[node];
                    ASSERT(node == p.dest_node);
                    //                printf("Merging %zu from A, %zu from B on node %u.\n", p.size_from_a, p.size_from_b, node);
                    cudaSetDevice(mcontext.get_device_id(node));
                    if (do_values) {
                        mgpu::merge(mout[node].keys_buffer, mout[node].values_buffer, p.size_from_a,
                            mout[node].keys_buffer + p.size_from_a, mout[node].values_buffer + p.size_from_a, p.size_from_b,
                            mout[node].keys, mout[node].values, comp, mcontext.get_mgpu_default_context_for_device(node));
                        CUERR;

                    }
                    else {
                        mgpu::merge(mout[node].keys_buffer, p.size_from_a,
                            mout[node].keys_buffer + p.size_from_a, p.size_from_b,
                            mout[node].keys, comp, mcontext.get_mgpu_default_context_for_device(node));
                        CUERR;
                    }
                }
        }

        std::array<std::vector<Search>, NUM_NODES - 1> plan_searches() {
            std::array<std::vector<Search>, NUM_NODES - 1> searches;
            std::array<uint, NUM_NODES> searches_per_node;

            std::array<Coord3, NUM_NODES> mcoord;
            idx_t a_size;
            idx_t b_size;
            idx_t output_size;

            mcoord[0].inp_a_start = mcoord[0].inp_b_start = mcoord[0].output_start = 0;

            for (uint i = 1; i < NUM_NODES; ++i) {
                searches[i - 1].reserve(NUM_NODES);
                searches_per_node[i - 1] = 0;
                mcoord[i].inp_a_start = mcoord[i - 1].inp_a_start + minp_a[i - 1].count;
                mcoord[i].inp_b_start = mcoord[i - 1].inp_b_start + minp_b[i - 1].count;
                mcoord[i].output_start = mcoord[i - 1].output_start + mout[i - 1].count;
            }
            a_size = mcoord.back().inp_a_start + minp_a.back().count;
            b_size = mcoord.back().inp_b_start + minp_b.back().count;
            output_size = mcoord.back().output_start + mout.back().count;
            searches_per_node.back() = 0;
            ASSERT(a_size + b_size == output_size);

            for (uint i = 1; i < NUM_NODES; ++i) {
                idx_t cross_diagonal = mcoord[i].output_start; //mcoord[i].output;
                //                printf("Cross diagonal %ld ...\n", cross_diagonal);

                idx_t begin = std::max(idx_t(0), cross_diagonal - b_size);
                idx_t end = std::min(cross_diagonal, a_size);
                idx_t p_a = begin;

                while (p_a < end) {
                    idx_t p_b = cross_diagonal - p_a;
                    auto r = find_bordering_rect(mcoord, p_a, p_b);

                    idx_t a = p_a - mcoord[r.first].inp_a_start;
                    idx_t b = p_b - mcoord[r.second].inp_b_start;
                    idx_t a_end = r.first + 1 < NUM_NODES ? mcoord[r.first + 1].inp_a_start : a_size;
                    idx_t inv_a = a_end - p_a;

                    uint schedule_on = mtopology_helper.get_node_to_schedule_search_on(r.first, r.second,
                        searches_per_node);
                    //                    printf("At %ld, %ld, hit rect: %u, %u  -> coords %ld, %ld, search on cross-diagonal %zu, scheduled on %u.\n",
                    //                           p_a, p_b, r.first, r.second, a, b, a+b, schedule_on);
                    ++searches_per_node[schedule_on];

                    searches[i - 1].push_back({ r.first, r.second, a + b, schedule_on, 0 });

                    idx_t steps = std::min(inv_a, b);
                    p_a += steps;
                }
            }
            return searches;
        }

        // TODO: Can be optimized by considering the current rect, of course.
        static std::pair<uint, uint> find_bordering_rect(const std::array<Coord3, NUM_NODES>& nodes,
            idx_t a, idx_t b) {
            uint rect_a = 0, rect_b = 0;
            for (uint i = 0; i < NUM_NODES; ++i) {
                if (nodes[i].inp_a_start <= a)
                    rect_a = i;
                if (nodes[i].inp_b_start < b)
                    rect_b = i;
            }
            return std::make_pair(rect_a, rect_b);
        }

        std::array<std::pair<MergePosition, MergePosition>, NUM_NODES - 1>
            create_partition_points_from_search_results(const std::array<std::vector<Search>, NUM_NODES - 1> all_searches) const {

            std::array<std::pair<MergePosition, MergePosition>, NUM_NODES - 1> partition_points;

            uint n = 0;
            for (const auto& searches : all_searches) {
                const Search* hit_search = nullptr;
                ASSERT(!searches.empty());
                for (uint search_idx = 0; search_idx < searches.size(); ++search_idx) {
                    const auto& search = searches[search_idx];
                    const Search* next_search = (search_idx + 1 < searches.size()) ?
                        &searches[search_idx + 1] : nullptr;

                    // If there are several searches per cross-diagonal, take the one producing the actual
                    // result.

                    idx_t begin = std::max(idx_t(0), search.cross_diagonal - idx_t(minp_b[search.node_b].count));
                    idx_t next_search_begin = 0;

                    if (next_search)
                        next_search_begin = std::max(idx_t(0), next_search->cross_diagonal - idx_t(minp_b[next_search->node_b].count));

                    idx_t end = std::min(search.cross_diagonal, idx_t(minp_a[search.node_a].count));

                    //                    printf("For search on %u, %u, cd %ld, begin/end are %ld, %ld\n",
                    //                           search.node_a, search.node_b, search.cross_diagonal, begin, end);

                    if ((search_idx == 0 && search.result == begin) ||
                        (search.result > begin && search.result < end) ||
                        (search.result == end && (next_search == nullptr || next_search->result == next_search_begin))) {
                        hit_search = &search;
                        break;
                    }
                }

                ASSERT(hit_search);

                MergePosition a{ hit_search->node_a, (index_t)hit_search->result };

                // Note this -1 here might wrap around the unsigned index type, but this will reversed in
                // safe_increment with another wrap-around anyway... :/
                MergePosition b{ hit_search->node_b, (index_t)hit_search->cross_diagonal - (index_t)hit_search->result - 1 };
                partition_points[n++] = { a, b };
            }
            return partition_points;
        }

        template <class emit_node_items_func>
        static void convert_to_per_node_ranges(const MergePosition& start, const MergePosition& end,
            const DistrArray& node_info,
            emit_node_items_func f) {
            ASSERT(start.node <= end.node);
            if (start.node == end.node) {
                f(start.node, start.index, end.index);
            }
            else {
                f(start.node, start.index, node_info[start.node].count);
                for (uint node = start.node + 1; node < end.node; ++node) {
                    f(node, 0, node_info[node].count);
                }
                f(end.node, 0, end.index);
            }
        }

        MergePartition output_partition(const MergePosition& last_a, const MergePosition& current_a,
            const MergePosition& last_b, const MergePosition& current_b,
            const DistributedArrayNode<key_t, value_t, index_t>& dest) const {
            MergePartition p{ dest.node_index, 0, 0 };
            index_t offset = 0;
            p.a_sources.reserve(current_a.node - last_a.node + 1);
            p.b_sources.reserve(current_b.node - last_b.node + 1);

            convert_to_per_node_ranges(last_a, current_a, minp_a, [&p, &offset](uint node, index_t start, index_t end) {
                if (end - start > 0)
                    p.a_sources.push_back({ node, start, end - start, offset });
                offset += end - start;
                });
            p.size_from_a = offset;

            offset = 0;
            convert_to_per_node_ranges(last_b, current_b, minp_b, [&p, &offset](uint node, index_t start, index_t end) {
                if (end - start > 0)
                    p.b_sources.push_back({ node, start, end - start, offset });
                offset += end - start;
                });
            p.size_from_b = offset;

            return p;
        }

        std::array<MergePartition, NUM_NODES> make_partitions(const std::array<std::vector<Search>, NUM_NODES - 1>& searches) const {

            std::array<std::pair<MergePosition, MergePosition>, NUM_NODES - 1> partition_points
                = create_partition_points_from_search_results(searches);

            MergePosition a_pos{ 0 }, b_pos{ 0 };

            std::array<MergePartition, NUM_NODES> partitions;

            uint p_index;
            for (p_index = 0; p_index < NUM_NODES - 1; ++p_index) {
                const std::pair<MergePosition, MergePosition>& split_point = partition_points[p_index];
                const MergePosition& split_a = split_point.first;
                MergePosition split_b = safe_increment(split_point.second, minp_b);
                partitions[p_index] = output_partition(a_pos, split_point.first, b_pos, split_b, mout[p_index]);
                ASSERT(partitions[p_index].size_from_a + partitions[p_index].size_from_b == mout[p_index].count);
                a_pos = split_a;
                b_pos = split_b;
            }

            MergePosition a_end{ NUM_NODES - 1, (index_t)minp_a[NUM_NODES - 1].count };
            MergePosition b_end{ NUM_NODES - 1, (index_t)minp_b[NUM_NODES - 1].count };

            partitions.back() = output_partition(a_pos, a_end, b_pos, b_end, mout.back());
            ASSERT(partitions.back().size_from_a + partitions.back().size_from_b == mout.back().count);

            return partitions;
        }

        static MergePosition safe_increment(MergePosition p, const DistrArray& arr) {
            ++p.index;
            // We allow pointing past the very last element without switching to the next node.
            if (p.index == arr[p.node].count && p.node != NUM_NODES - 1) {
                p.index = 0;
                ++p.node;
                ASSERT(p.node <= NUM_NODES);
            }
            ASSERT(p.index <= arr[p.node].count);
            return p;
        }

    };

}
#endif // DISTRIB_MERGE_HPP
