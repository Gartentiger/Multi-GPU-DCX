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

#include <kamping/collectives/allgather.hpp>
#include <kamping/data_buffer.hpp>
#include "kamping/collectives/bcast.hpp"
#include <kamping/named_parameters.hpp>
#include <kamping/checking_casts.hpp>
#include <kamping/p2p/send.hpp>
#include <kamping/p2p/isend.hpp>
#include "kamping/measurements/printer.hpp"
#include "kamping/measurements/timer.hpp"

#include <kamping/p2p/recv.hpp>

#include <queue>
#include <span>
#include <fstream>
#include <cstdio>

namespace crossGPUReMerge
{
    template<typename ke>
    __global__ void printArrays(ke* key, size_t size, size_t rank, int spec)
    {
        for (size_t i = 0; i < size; i++)
        {
            printf("[%lu] Key %d: %lu\n", rank, spec, key[i]);
        }
        printf("---------------------------------------------------------------------------\n");
    }
    __global__ void printArrays(uint64_t* key, uint64_t* value, size_t size, size_t rank)
    {
        for (size_t i = 0; i < size; i++)
        {

            printf("[%lu]: sa_rank 1: %lu, old_ranks 2: %lu\n", rank, key[i], value[i]);
        }
        printf("---------------------------------------------------------------------------\n");
    }
    template <typename ke>
    __global__ void printArrays(ke* key, ke* value, size_t size, size_t rank)
    {
        for (size_t i = 0; i < size; i++)
        {

            printf("[%lu]: sa_rank 1: %lu, old_ranks 2: %lu\n", rank, key[i], value[i]);
        }
        printf("---------------------------------------------------------------------------\n");
    }

    enum MergePathBounds
    {
        bounds_lower,
        bounds_upper
    };

    // This function comes from ModernGPU.
    template <MergePathBounds bounds = bounds_lower, typename a_keys_it,
        typename b_keys_it, typename int_t, typename comp_t>
    HOST_DEVICE int_t merge_path(a_keys_it a_keys, int_t a_count, b_keys_it b_keys, int_t b_count, int_t diag,
        comp_t comp)
    {
        using type_t = typename std::iterator_traits<a_keys_it>::value_type;

        int_t begin = max(int_t(0), diag - b_count);
        int_t end = min(diag, a_count);
        // printf("fff %u", a_keys[4]);
        while (begin < end)
        {
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
        int64_t diag, comp_t comp, int64_t* store_result)
    {
        *store_result = crossGPUReMerge::merge_path(a_keys, a_count, b_keys, b_count, diag, comp);
    }

    template <size_t NUM_GPUS, class mtypes>
    std::array<std::vector<InterNodeCopy>, NUM_GPUS> partitions_to_copies(const std::array<MergeNode<mtypes>, NUM_GPUS>& nodes)
    {
        std::array<std::vector<InterNodeCopy>, NUM_GPUS> copies;

        for (const auto& node : nodes)
        {
            for (const MultiWayMergePartition* p : node.scheduled_work.multi_merge_partitions)
            {
                for (const MergePartitionSource& s : p->sources)
                {
                    copies[s.node].push_back({ s.node, p->dest_node,
                                              s.range.start,
                                              p->dest_range.start + s.dest_offset,
                                              s.range.end - s.range.start });
                }
            }

            for (const MergePartition* p : node.scheduled_work.merge_partitions)
            {
                sa_index_t dest1 = p->dest_range.start;
                sa_index_t dest2 = dest1 + p->size_from_1;
                for (const MergePartitionSource& s1 : p->sources_1)
                {
                    copies[s1.node].push_back({ s1.node, p->dest_node,
                                               s1.range.start,
                                               dest1 + s1.dest_offset,
                                               s1.range.end - s1.range.start });
                }
                for (const MergePartitionSource& s2 : p->sources_2)
                {
                    copies[s2.node].push_back({ s2.node, p->dest_node,
                                               s2.range.start,
                                               dest2 + s2.dest_offset,
                                               s2.range.end - s2.range.start });
                }
            }
        }
        return copies;
    }

    template<typename key_t, typename int_t, class comp_fun_t, size_t MAX_GPUS>
    std::tuple<uint, int_t, key_t> multi_way_k_selectHost(ArrayDescriptor<MAX_GPUS, key_t, int_t> arr_descr, int_t M, int_t k, comp_fun_t comp, Communicator<> comm) {

        int_t mid_index[MAX_GPUS];
        key_t mid_values[MAX_GPUS];
        int_t starts[MAX_GPUS];
        int_t ends[MAX_GPUS];
        int before_mid_count = 0;
        size_t total_size = 0;
        // Initialize
        // M = ranges.size()
        for (uint i = 0; i < M; ++i) {
            starts[i] = 0;
            ends[i] = arr_descr.lengths[i];
            uint a = (starts[i] + ends[i]) / 2;

            mid_index[i] = (starts[i] + ends[i]) / 2;
            if (world_rank() == i) {
                cudaMemcpy(mid_values + i, arr_descr.keys[i] + mid_index[i], sizeof(key_t), cudaMemcpyDeviceToHost);
                // printf("[%lu] cpying done: %u\n", world_rank(), mid_values[i]);
            }
            std::span<key_t> srb(mid_values + i, 1);
            comm.bcast(send_recv_buf(srb), send_recv_count(1), root((size_t)i));
            // printf("[%lu] received: %u\n", world_rank(), mid_values[i]);

            before_mid_count += mid_index[i];
            total_size += arr_descr.lengths[i];
        }


        assert(k < total_size);

        bool done = false;
        key_t result_value;
        int_t result_index = 0;
        int_t result_list_index;

        while (!done) {
            key_t min_value, max_value;
            //        memset(&min_value , 0xff, sizeof(key_t));
            //        memset(&max_value , 0x00, sizeof(key_t));

            //        key_t min_value = std::numeric_limits<key_t>::max();  // Does silently fail for unknown types... :/
            //        key_t max_value = std::numeric_limits<key_t>::min();
            int_t max_index = -1, min_index = -2;

            for (int i = 0; i < M; ++i) {
                if (starts[i] < ends[i]) {
                    // Pick the min index in a range of equal values.
                    if (comp(mid_values[i], min_value) || min_index < 0) { // <
                        min_value = mid_values[i];
                        min_index = i;
                    }
                    // Pick the max index in a range of equal values.
                    if (!comp(mid_values[i], max_value) || max_index < 0) { // >=
                        max_value = mid_values[i];
                        max_index = i;
                    }
                }
                //            printf("Multi-way partioning search: %u from %u to %u.\n", i, starts[i], ends[i]);
                //            std::cout << i << ". From " << starts[i] << " to " << ends[i] << ", mid: " << mid_index[i]
                //                      << ", value: " << mid_values[i] << std::endl;
            }

            //        std::cout << "min index: " << min_index << ", value: " << min_value << std::endl;
            //        std::cout << "max index: " << max_index << ", value: " << max_value;

            //        std::cout << "\nbefore mid count: " << before_mid_count << ", k: " << k << std::endl;
            if (min_index == max_index && before_mid_count == k) {
                result_value = mid_values[min_index];
                result_index = mid_index[min_index];
                result_list_index = min_index;
                break;
            }
            if (before_mid_count < k) {
                //            std::cout << "Adjusting min..." << min_index << std::endl;
                int_t old_mid = mid_index[min_index];
                if (starts[min_index] == mid_index[min_index]) {
                    starts[min_index] = mid_index[min_index] + 1;
                }
                else {
                    starts[min_index] = mid_index[min_index];
                }
                mid_index[min_index] = (starts[min_index] + ends[min_index]) / 2;
                if (world_rank() == min_index) {
                    cudaMemcpy(mid_values + min_index, arr_descr.keys[min_index] + mid_index[min_index], sizeof(key_t), cudaMemcpyDeviceToHost);
                    // printf("[%lu]cpying 2\n", world_rank());
                }
                std::span<key_t> srb(mid_values + min_index, 1);
                comm.bcast(send_recv_buf(srb), send_recv_count(1), root((size_t)min_index));
                // UPDATE_MID_INDEX_AND_VALUE(min_index);
                before_mid_count += mid_index[min_index] - old_mid;
            }
            else {
                //            std::cout << "Adjusting max... " << max_index << std::endl;
                ends[max_index] = mid_index[max_index];
                int_t old_mid = mid_index[max_index];

                mid_index[max_index] = (starts[max_index] + ends[max_index]) / 2;
                if (world_rank() == max_index) {
                    cudaMemcpy(mid_values + max_index, arr_descr.keys[max_index] + mid_index[max_index], sizeof(key_t), cudaMemcpyDeviceToHost);
                    // printf("[%lu]cpying 3\n", world_rank());
                }
                std::span<key_t> srb(mid_values + max_index, 1);
                comm.bcast(send_recv_buf(srb), send_recv_count(1), root((size_t)max_index));
                // UPDATE_MID_INDEX_AND_VALUE(max_index);
                before_mid_count -= old_mid - mid_index[max_index];
            }
            //        std::cout << std::endl;
        }

        //    std::cout << "Needed " << count << " iterations with total size " << total_size << ".\n";

        //    printf("Multi-way partioning search exited: result list index %u, result index %u.\n", result_list_index, result_index);

        return std::make_tuple(result_list_index, result_index, result_value);
    }


    template <size_t NUM_GPUS, class mtypes, template <size_t, class> class TopologyHelperT>
    class GPUMergeProcessorT
    {
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
        void do_searches(comp_func_t comp)
        {
            auto& t = kamping::measurements::timer();
            // t.synchronize_and_start("search");
            mhost_search_temp_allocator.reset();
            mcontext.sync_all_streams();
            // printf("[%lu] do_searches\n", world_rank());

            std::vector<SearchGPU<NUM_GPUS, key_t, int64_t>> searchesGPU;
            searchesGPU.clear();
            QDAllocator& dAlloc = mcontext.get_device_temp_allocator(world_rank());
            // needed for ipc shared mem if we have merges in the same node
            comm_world().barrier();

            // for (int i = 0; i < 4; i++) {
            //     printArrays << <1, 1, 0, mcontext.get_gpu_default_stream(world_rank()) >> > (mnodes[i].info.keys, 7, world_rank(), i);
            // }
            // mcontext.sync_all_streams();
            // comm_world().barrier();
            // t.synchronize_and_start("in_node_find");

            // check for all merges that are in one node. They can be executed normally
            for (MergeNode& node : mnodes)
            {
                for (auto ms : node.scheduled_work.multi_searches)
                {
                    ms->in_node_merge = true;
                    ms->used = false;
                    for (const auto& r : ms->ranges)
                    {
                        if (world_rank() == r.start.node) {
                            assert(!ms->used);
                            ms->used = true;
                        }
                        ms->in_node_merge &= mcontext.get_peer_status(node.info.index, r.start.node) >= 1;
                    }
                    // printf("[%lu] multi search in node: %s\n", world_rank(), ms->in_node_merge ? "true" : "false");
                    if (ms->in_node_merge && world_rank() == node.info.index) {
                        ArrayDescriptor<NUM_GPUS, key_t, int64_t> ad;
                        int i = 0;
                        for (const auto& r : ms->ranges)
                        {
                            ad.lengths[i] = r.end.index - r.start.index;
                            ad.keys[i] = mnodes[r.start.node].info.keys + r.start.index;
                            // printf("[%lu] i: %d, node: %u, index; %u, length: %lu\n", world_rank(), i, r.start.node, r.start.index, ad.lengths[i]);
                            i++;
                        }
                        const size_t result_buffer_length = ms->ranges.size() + 1;

                        const cudaStream_t& stream = mcontext.get_gpu_default_stream(world_rank());

                        ms->d_result_ptr = dAlloc.get<int64_t>(result_buffer_length);
                        ms->h_result_ptr = mhost_search_temp_allocator.get<int64_t>(result_buffer_length);
                        // printf("[%lu] ranges.size(): %ld, split_index: %ld\n", world_rank(), (int64_t)ms->ranges.size(), (int64_t)ms->split_index);
                        multi_find_partition_points << <1, NUM_GPUS, 0, stream >> > (ad, (int64_t)ms->ranges.size(), (int64_t)ms->split_index,
                            comp,
                            (int64_t*)ms->d_result_ptr,
                            (uint*)(ms->d_result_ptr + result_buffer_length - 1));

                        cudaMemcpyAsync(ms->h_result_ptr, ms->d_result_ptr,
                            result_buffer_length * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);

                        // mcontext.sync_all_streams();
                        // for (int i = 0; i < ms->ranges.size() + 1; i++) {
                        //     printf("[%lu] results[%d]: %ld\n", world_rank(), i, ms->h_result_ptr[i]);
                        // }
                    }
                }
            }
            // t.stop();

            // t.synchronize_and_start("mult_way_k_Host");

            // printf("[%lu] queue\n", world_rank());
            // queue work for the gpu associated with this process
            for (MergeNode& node : mnodes)
            {
                for (auto ms : node.scheduled_work.multi_searches)
                {
                    if (ms->in_node_merge) {
                        continue;
                    }

                    if (!ms->used) {
                        continue;
                    }

                    const size_t result_buffer_length = ms->ranges.size() + 1;
                    ms->h_result_ptr = mhost_search_temp_allocator.get<int64_t>(result_buffer_length);

                    // printf("[%lu] not in node\n", world_rank());
                    SearchGPU<NUM_GPUS, key_t, int64_t> sgpu;
                    ArrayDescriptor<NUM_GPUS, key_t, int64_t> ad;

                    std::vector<int> ranks(ms->ranges.size());

                    int i = 0;
                    for (const auto& r : ms->ranges)
                    {
                        ranks[i] = r.start.node;
                        if (r.start.node == world_rank()) {
                            sgpu.startIndex = r.start.index;
                        }
                        ad.lengths[i] = r.end.index - r.start.index;
                        ad.keys[i] = mnodes[r.start.node].info.keys + r.start.index;
                        i++;
                    }


                    Communicator c = comm_world().create_subcommunicators(ranks);
                    // could be multi threaded

                    std::tuple<size_t, size_t, key_t> ksmallest = multi_way_k_selectHost(ad, (int64_t)ms->ranges.size(), (int64_t)ms->split_index, comp, c);
                    *(reinterpret_cast<uint*>(ms->h_result_ptr + result_buffer_length - 1)) = (uint)std::get<0>(ksmallest);

                    sgpu.M = (int64_t)ms->ranges.size();
                    for (int i = 0; i < (int64_t)ms->ranges.size(); i++) {
                        sgpu.lengths[i] = ad.lengths[i];
                    }
                    sgpu.ksmallest = ksmallest;
                    searchesGPU.push_back(sgpu);
                }
            }
            // t.stop();

            // for (auto searches : searchesGPU) {
                //     printf("[%lu] ksmallest: %lu, %lu, %u\n", world_rank(), std::get<0>(searches.ksmallest), std::get<1>(searches.ksmallest), std::get<2>(searches.ksmallest));
                // }
                //printf("[%lu] searchesGPU.size(): %lu\n", world_rank(), searchesGPU.size());

                //auto resultPtrHost = mhost_search_temp_allocator.get<int64_t>(searchesGPU.size());
            // t.synchronize_and_start("not_in_node_find");
            std::vector<int64_t> resultHost(searchesGPU.size());
            if (searchesGPU.size() > 0)
            {
                // printf("[%lu] result HOst\n", world_rank());
                auto resultPtrDevice = dAlloc.get<int64_t>(searchesGPU.size());
                find_partition_points << <1, searchesGPU.size(), 0, mcontext.get_gpu_default_stream(world_rank()) >> > (mnodes[world_rank()].info.keys, comp, (uint)world_rank(), resultPtrDevice, searchesGPU.data());

                cudaMemcpyAsync(resultHost.data(), resultPtrDevice,
                    searchesGPU.size() * sizeof(int64_t), cudaMemcpyDeviceToHost, mcontext.get_gpu_default_stream(world_rank()));
            }
            // t.stop();
            // sync again for next communication
            // comm_world().barrier();
            // printf("[%lu] multi partition find done\n", world_rank());

            // t.synchronize_and_start("send_data");
            ncclGroupStart();
            for (MergeNode& node : mnodes) {
                int msgTag = 0;
                for (auto s : node.scheduled_work.searches)
                {
                    assert(node.info.index == s->node_1 || node.info.index == s->node_2);
                    // we have peer access no need to send/recv data
                    if (mcontext.get_peer_status(s->node_1, s->node_2) >= 1) {
                        msgTag++;
                        continue;
                    }

                    if (node.info.index != world_rank())
                    {
                        if (s->node_1 == world_rank())
                        {
                            key_t* start_1 = mnodes[s->node_1].info.keys + s->node1_range.start;

                            int64_t size_1 = s->node1_range.end - s->node1_range.start;
                            // std::span<key_t> sb(start_1, size_1);
                            ncclSend(start_1, size_1 * sizeof(key_t), ncclChar, s->node_2, mcontext.get_nccl(), mcontext.get_streams(s->node_1)[s->node_2]);

                            // comm_world().isend(send_buf(sb), tag(msgTag), send_count(size_1), destination((size_t)node.info.index));
                        }
                        else if (s->node_2 == world_rank())
                        {
                            key_t* start_2 = mnodes[s->node_2].info.keys + s->node2_range.start;
                            int64_t size_2 = s->node2_range.end - s->node2_range.start;
                            ncclSend(start_2, size_2 * sizeof(key_t), ncclChar, s->node_1, mcontext.get_nccl(), mcontext.get_streams(s->node_2)[s->node_1]);
                            // std::span<key_t> sb(start_2, size_2);
                            // comm_world().isend(send_buf(sb), tag(msgTag), send_count(size_2), destination((size_t)node.info.index));
                        }
                    }
                    else
                    {
                        if (s->node_1 == world_rank())
                        {
                            int64_t size_2 = s->node2_range.end - s->node2_range.start;

                            cudaMallocAsync(&(mnodes[s->node_2].info.keys), sizeof(key_t) * size_2, mcontext.get_streams(s->node_1)[s->node_2]);

                            // std::span<key_t> rb(start_2, size_2);
                            // comm_world().recv(recv_buf(rb), tag(msgTag++), recv_count(size_2));
                            ncclRecv(mnodes[s->node_2].info.keys, size_2 * sizeof(key_t), ncclChar, s->node_2, mcontext.get_nccl(), mcontext.get_streams(s->node_1)[s->node_2]);

                        }
                        else {
                            int64_t size_1 = s->node1_range.end - s->node1_range.start;

                            cudaMallocAsync(&(mnodes[s->node_1].info.keys), sizeof(key_t) * size_1, mcontext.get_streams(s->node_2)[s->node_1]);

                            // std::span<key_t> rb(start_1, size_1);
                            // comm_world().recv(recv_buf(rb), tag(msgTag++), recv_count(size_1));
                            ncclRecv(mnodes[s->node_1].info.keys, size_1 * sizeof(key_t), ncclChar, s->node_1, mcontext.get_nccl(), mcontext.get_streams(s->node_2)[s->node_1]);
                        }
                        // std::span<key_t> rb(otherKey, otherSize);
                    }
                    msgTag++;
                }
            }
            ncclGroupEnd();

            // t.stop();

            // t.synchronize_and_start("partition_search_two");
            for (MergeNode& node : mnodes)
            {
                const uint node_index = node.info.index;
                // //(mcontext.get_device_id(node_index));
                // CUERR;

                if (world_rank() == node_index)
                {
                    int msgTag = 0;


                    for (auto s : node.scheduled_work.searches)
                    {
                        uint other = (node_index == s->node_1) ? s->node_2 : s->node_1;
                        const cudaStream_t& stream = mcontext.get_streams(node_index).at(other);

                        s->d_result_ptr = dAlloc.get<int64_t>(1);
                        s->h_result_ptr = mhost_search_temp_allocator.get<int64_t>(1);
                        int64_t size_1 = s->node1_range.end - s->node1_range.start;
                        int64_t size_2 = s->node2_range.end - s->node2_range.start;
                        key_t* start_1;
                        key_t* start_2;
                        key_t* tempRef = nullptr;


                        if (other == node.info.index || mcontext.get_peer_status(world_rank(), other) >= 1)
                        {
                            start_1 = mnodes[s->node_1].info.keys + s->node1_range.start;
                            start_2 = mnodes[s->node_2].info.keys + s->node2_range.start;
                        }
                        else
                        {
                            // printf("[%lu] receiving search, no peer access\n", world_rank());
                            if (s->node_1 == world_rank())
                            {
                                start_1 = mnodes[s->node_1].info.keys + s->node1_range.start;
                                start_2 = mnodes[s->node_2].info.keys;
                                tempRef = start_2;
                                // cudaMalloc(&start_2, sizeof(key_t) * size_2);
                                // tempRef = start_2;
                                // std::span<key_t> rb(start_2, size_2);
                                // comm_world().recv(recv_buf(rb), tag(msgTag++), recv_count(size_2));
                            }
                            else
                            {
                                start_1 = mnodes[s->node_1].info.keys;
                                start_2 = mnodes[s->node_2].info.keys + s->node2_range.start;
                                tempRef = start_1;
                                // cudaMalloc(&start_1, sizeof(key_t) * size_1);
                                // tempRef = start_1;
                                // std::span<key_t> rb(start_1, size_1);
                                // comm_world().recv(recv_buf(rb), tag(msgTag++), recv_count(size_1));

                            }
                        }

                        run_partitioning_search << <1, 1, 0, stream >> > (start_1, size_1, start_2, size_2, s->cross_diagonal,
                            comp, s->d_result_ptr);
                        CUERR;
                        cudaMemcpyAsync(s->h_result_ptr, s->d_result_ptr,
                            sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
                        CUERR;
                        if (tempRef != nullptr)
                            cudaFreeAsync(tempRef, stream);
                    }
                }
            }
            // t.stop();
            mcontext.sync_all_streams();
            // printf("[%lu] partition search working done\n", world_rank());

            // t.synchronize_and_start("allgather_splitter_multi_split");
            std::vector<std::queue<int64_t>> resultSplitted;
            std::vector<int64_t> resultSplitIdx;
            auto [recvCountsOut] = comm_world().allgatherv(send_buf(resultHost), recv_buf<resize_to_fit>(resultSplitIdx), recv_counts_out());
            int totalIdx = 0;
            for (auto recv : recvCountsOut) {
                std::queue<int64_t> q;
                for (int i = 0; i < recv; i++) {
                    q.push(resultSplitIdx[totalIdx++]);
                }
                resultSplitted.push_back(q);
            }
            // t.stop();


            // t.synchronize_and_start("bcast_result_multi_split");
            // printf("[%lu] allgatg\n", world_rank());
            for (MergeNode& node : mnodes)
            {
                const uint node_index = node.info.index;
                // QDAllocator& d_alloc = mcontext.get_device_temp_allocator(node_index);
                // int adCount = 0;
                for (auto ms : node.scheduled_work.multi_searches)
                {
                    // ArrayDescriptor<NUM_GPUS, key_t, int64_t> ad;
                    // int i = 0;
                    // for (const auto& r : ms->ranges)
                    // {
                    //     ad.keys[i] = mnodes[r.start.node].info.keys + r.start.index;
                    //     ad.lengths[i] = r.end.index - r.start.index;
                    //     i++;
                    // }
                    //std::tuple<size_t, size_t, key_t> ksmallest = multi_way_k_selectHost(ad, (int64_t)ms->ranges.size(), (int64_t)ms->split_index, comp);
                    //if (world_rank() == node_index) {
                    //const size_t result_buffer_length = ms->ranges.size() + 1;

                    //    const cudaStream_t& stream = mcontext.get_gpu_default_stream(node_index);

                    //ms->d_result_ptr = d_alloc.get<int64_t>(result_buffer_length);
                    //ms->h_result_ptr = mhost_search_temp_allocator.get<int64_t>(result_buffer_length);

                    // multi_find_partition_points << <1, NUM_GPUS, 0, stream >> > (ad, (int64_t)ms->ranges.size(), (int64_t)ms->split_index,
                    //     comp,
                    //     (int64_t*)ms->d_result_ptr,
                    //     (uint*)(ms->d_result_ptr + result_buffer_length - 1), ksmallest);

                    //   cudaMemcpyAsync(ms->h_result_ptr, ms->d_result_ptr,
                    //       result_buffer_length * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);                    
                    //}
                    const size_t result_buffer_length = ms->ranges.size() + 1;

                    if (ms->in_node_merge) {
                        if (world_rank() != node.info.index) {
                            ms->h_result_ptr = mhost_search_temp_allocator.get<int64_t>(result_buffer_length);
                        }
                        comm_world().bcast(send_recv_buf(std::span<int64_t>(ms->h_result_ptr, ms->ranges.size() + 1)), root((size_t)node.info.index));
                        continue;
                    }
                    // printf("[%lu] after bcast\n", world_rank());

                    if (!ms->used) {
                        ms->h_result_ptr = mhost_search_temp_allocator.get<int64_t>(result_buffer_length);
                    }

                    int i = 0;
                    for (auto r : ms->ranges) {
                        ms->h_result_ptr[i] = resultSplitted[r.start.node].front();
                        resultSplitted[r.start.node].pop();
                        i++;
                    }
                }
            }
            // t.stop();

            // t.synchronize_and_start("allgather_two_search");
            MergeNode mergeNode = mnodes[world_rank()];
            // printf("[%lu] do search kernel phase done, size multi: %lu\n", world_rank(), mergeNode.scheduled_work.multi_searches.size());
            size_t send_size_sum = 0;
            for (MergeNode node : mnodes) {
                send_size_sum += node.scheduled_work.searches.size();
            }
            std::vector<int64_t> recv_search_result;
            if (send_size_sum > 0) {
                size_t send_size = mergeNode.scheduled_work.searches.size();
                std::vector<int64_t> send_search_result(send_size);
                send_search_result.clear();
                for (auto s : mergeNode.scheduled_work.searches)
                {
                    // printf("[%lu] result before communication %u\n", world_rank(), *s->h_result_ptr);
                    send_search_result.push_back(*s->h_result_ptr);
                }
                // printf("[%lu] before allgather\n", world_rank());


                comm_world().allgatherv(send_buf(send_search_result), recv_buf<resize_to_fit>(recv_search_result));
            }

            // t.stop();
            // t.synchronize_and_start("memcpys");

            // printf("Allgather %lu\n", world_rank());
            int enumer = 0;
            for (int i = 0; i < comm_world().size(); i++)
            {
                for (auto s : mnodes[i].scheduled_work.searches)
                {
                    if (world_rank() != i) {
                        s->h_result_ptr = mhost_search_temp_allocator.get<int64_t>(1);
                        memcpy(s->h_result_ptr, recv_search_result.data() + enumer, sizeof(int64_t));
                        // *s->h_result_ptr = recv_search_result[enumer];
                    }
                    enumer++;
                }
            }
            // printf("[%lu] after allgather\n", world_rank());
            //}
            // printf("Searches done %lu\n", world_rank());

            // size_t mulit_search_size = mergeNode.scheduled_work.multi_searches.size();
            // std::vector<int64_t> send_multi_search_result;
            // send_multi_search_result.reserve(mulit_search_size);
            // send_multi_search_result.clear();
            // for (auto ms : mergeNode.scheduled_work.multi_searches)
            // {
            //     size_t size = ms->ranges.size() + 1;
            //     for (size_t j = 0; j < size; j++)
            //     {
            //         // printf("ms->h_result_ptr %ld, rank %lu\n", ms->h_result_ptr[j], world_rank());
            //         send_multi_search_result.push_back(ms->h_result_ptr[j]);
            //     }
            // }

            // std::vector<int64_t> recv_multi_search_result;
            // auto [multi_search_output_counts] = comm_world().allgatherv(send_buf(send_multi_search_result), recv_buf<resize_to_fit>(recv_multi_search_result), recv_counts_out());
            // printf("Multi searches %lu, counts.size() %lu\n", world_rank(), multi_search_output_counts.size());
            // for (int64_t ah : recv_multi_search_result)
            // {
            //     printf("[%lu] received multi search results %ld\n", world_rank(), ah);
            // }
            // printf("[%lu] after allgather 2\n", world_rank());

            // int totalIdx = 0;
            // for (int i = 0; i < comm_world().size(); i++)
            // {
            //     ASSERT(mnodes[i].info.index == i);
            //     for (auto ms : mnodes[i].scheduled_work.multi_searches)
            //     {
            //         int size = ms->ranges.size() + 1;
            //         if (world_rank() != mnodes[i].info.index) {
            //             ms->h_result_ptr = mhost_search_temp_allocator.get<int64_t>(size);
            //             memcpy(ms->h_result_ptr, recv_multi_search_result.data() + totalIdx, size * sizeof(int64_t));
            //         }
            //         totalIdx += size;
            //     }
            // }
            // for (MergeNode node : mnodes) {
            //     for (auto ms : node.scheduled_work.multi_searches)
            //     {
            //         for (int i = 0; i < ms->ranges.size() + 1; i++) {
            //             printf("[%lu] ms->h_result_ptr[%d]: %ld, rank: %lu\n", world_rank(), i, ms->h_result_ptr[i]);
            //         }

            //     }
            // }
            // printf("[%lu] searches done copying back\n", world_rank());
            for (MergeNode& node : mnodes)
            {

                for (auto s : node.scheduled_work.searches)
                {
                    s->result = *s->h_result_ptr;
                    // printf("[%lu] search result: %ld\n", world_rank(), s->result);
                }

                for (auto ms : node.scheduled_work.multi_searches)
                {
                    ms->results.resize(ms->ranges.size());
                    memcpy(ms->results.data(), ms->h_result_ptr, ms->ranges.size() * sizeof(int64_t));
                    // for (int i = 0; i < ms->ranges.size(); i++) {
                        // printf("[%lu] results[%d] 2: %ld\n", world_rank(), i, ms->results[i]);
                    // }

                    ms->range_to_take_one_more = ms->h_result_ptr[ms->ranges.size()] & 0xffffffff;
                    // printf("[%lu] range_to_take_one_more: %ld\n", world_rank(), ms->range_to_take_one_more);

                }
            }
            // t.stop();
            // t.stop();
        }

        void
            dump_array(const sa_index_t* idx, size_t size)
        {
            for (size_t i = 0; i < size; ++i)
            {
                std::cerr << idx[i] << ", ";
            }
        }

        template <class comp_fun_t>
        void do_copy_and_merge(comp_fun_t comp, std::function<void()> dbg_func)
        {
            // auto& t = kamping::measurements::timer();
            // t.synchronize_and_start("do_copy_and_merge");
            (void)dbg_func;
            std::array<std::vector<InterNodeCopy>, NUM_GPUS> copies = partitions_to_copies<NUM_GPUS, mtypes>(mnodes);
            std::array<size_t, NUM_GPUS> detour_sizes;
            for (uint i = 0; i < NUM_GPUS; ++i)
                detour_sizes[i] = mnodes[i].info.detour_buffer_size;

            bool do_values = mnodes[0].info.values != nullptr;
            // t.synchronize_and_start("bcast_do_values");
            comm_world().bcast_single(send_recv_buf(do_values), root(0));
            // t.stop();
            // t.synchronize_and_start("do_copies_async");
            mtopology_helper.do_copies_async(copies, detour_sizes, do_values);
            // t.stop();
            // t.synchronize_and_start("multi_mergers");
            std::vector<NodeMultiMerger<NUM_GPUS, mtypes, comp_fun_t>> multi_mergers;
            multi_mergers.reserve(mnodes.size() * mnodes.size()); // Essential because of pointer-init. in c'tor.
            for (const MergeNode& node : mnodes)
            {
                if (node.info.index == world_rank())
                {
                    for (const MultiWayMergePartition* p : node.scheduled_work.multi_merge_partitions)
                    {
                        multi_mergers.emplace_back(mcontext, node, *p, comp, do_values);
                    }
                }
            }
            // t.stop();
            // t.synchronize_and_start("sync_do_copies_async");
            mcontext.sync_all_streams();
            // t.stop();
            //            if (dbg_func)
            //                dbg_func();
            // t.synchronize_and_start("mgpu::merge");
            for (const MergeNode& node : mnodes)
            {
                // //(mcontext.get_device_id(node.info.index));

                mgpu::my_mpgu_context_t& mgpu_context = mcontext.get_mgpu_default_context_for_device(node.info.index);
                if (node.info.index == world_rank())
                {

                    for (const MergePartition* p : node.scheduled_work.merge_partitions)
                    {
                        sa_index_t dest1 = p->dest_range.start;
                        sa_index_t dest2 = dest1 + p->size_from_1;
                        mgpu_context.reset_temp_memory();

                        //                    if (p->size_from_1 > 0 && p->size_from_2 > 0) {
                        if (do_values)
                        {
                            mgpu::merge(node.info.key_buffer + dest1, node.info.value_buffer + dest1, (int)p->size_from_1,
                                node.info.key_buffer + dest2, node.info.value_buffer + dest2, (int)p->size_from_2,
                                node.info.keys + dest1, node.info.values + dest1, comp, mgpu_context);
                            CUERR;
                        }
                        else
                        {
                            mgpu::merge(node.info.key_buffer + dest1, (int)p->size_from_1,
                                node.info.key_buffer + dest2, (int)p->size_from_2,
                                node.info.keys + dest1, comp, mgpu_context);
                            CUERR;
                        }
                        //                        printf("\nScheduled merge with sizes %d, %d, on device %u",
                        //                                (int)p->size_from_1, (int)p->size_from_2, node.info.index);
                        //                    }
                    }
                }
            }
            //            for (const MergeNode& node : mnodes) {
            //                //(mcontext.get_device_id(node.info.index));

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
            // mcontext.sync_all_streams();
            // t.stop();
            // t.synchronize_and_start("do_merge_step");
            // printf("[%lu] before multi merge\n", world_rank());
            if (!multi_mergers.empty())
            {
                bool finished = false;
                while (!finished)
                {
                    finished = true;

                    for (auto& merger : multi_mergers)
                    {
                        finished &= merger.do_merge_step();
                    }
                    for (const MergeNode& node : mnodes)
                    {
                        if (world_rank() == node.info.index)
                            mcontext.get_device_temp_allocator(node.info.index).reset();
                    }
                    for (auto& merger : multi_mergers)
                    {
                        merger.sync_used_streams();
                    }
                    //                    if (dbg_func)
                    //                        dbg_func();
                    //                    mcontext.sync_all_streams();
                }
            }
            mcontext.sync_all_streams();
            // t.stop();
            // t.stop();

        }

        void init_nodes(const std::array<MergeNodeInfo<mtypes>, NUM_GPUS>& merge_node_info)
        {
            // first call merge_node_info.size() == NUM_GPUS
            for (uint n = 0; n < merge_node_info.size(); ++n)
            {
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
