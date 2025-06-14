#pragma once

#include "context.cuh"
#include "auxiliary.cuh"

#include "multisplit/dispatch_multisplit.cuh"

template <uint NUM_GPUS>
class MultiSplit
{
    static const uint num_gpus = NUM_GPUS;
    using DispatchMultisplit = DispatchMultisplit<NUM_GPUS>;

    MultiGPUContext<NUM_GPUS>& context;

    const uint32_t effective_buckets;

    uint32_t* h_offsets[NUM_GPUS];

public:
    MultiSplit(MultiGPUContext<NUM_GPUS>& context_)
        : context(context_),
        effective_buckets(DispatchMultisplit::effective_num_buckets())
    {
        for (uint gpu = 0; gpu < NUM_GPUS; ++gpu)
        {
            if (world_rank() == gpu)
            {
                cudaSetDevice(0);
                cudaMallocHost(&h_offsets[gpu], sizeof(uint32_t) * NUM_GPUS);
            }
        }
        CUERR;
    }

    ~MultiSplit()
    {
        for (uint gpu = 0; gpu < num_gpus; ++gpu)
        {
            cudaSetDevice(context.get_device_id(gpu));
            cudaFreeHost(h_offsets[gpu]);
        }
        CUERR;
    }

    template <typename key_t, typename value_t, typename index_t, typename table_t, typename funct_t>
    bool execAsync(const std::array<MultiSplitNodeInfoT<key_t, value_t, index_t>, NUM_GPUS>& node_info,
        split_table_tt<table_t, NUM_GPUS>& split_table,
        std::array<table_t, NUM_GPUS>& src_lens,
        std::array<table_t, NUM_GPUS>& dest_lens,
        funct_t functor) const noexcept
    {
        return doExecAsync<key_t, value_t, index_t, table_t, funct_t, false>(node_info, split_table, src_lens, dest_lens, functor);
    }

    template <typename key_t, typename value_t, typename index_t, typename table_t, typename funct_t>
    bool execKVAsync(const std::array<MultiSplitNodeInfoT<key_t, value_t, index_t>, NUM_GPUS>& node_info,
        split_table_tt<table_t, NUM_GPUS>& split_table,
        std::array<table_t, NUM_GPUS>& src_lens,
        std::array<table_t, NUM_GPUS>& dest_lens,
        funct_t functor) const noexcept
    {
        return doExecAsync<key_t, value_t, index_t, table_t, funct_t, true>(node_info, split_table, src_lens, dest_lens, functor);
    }

    template <typename key_t, typename value_t, typename index_t, typename table_t, typename funct_t,
        bool include_values>
    bool doExecAsync(const std::array<MultiSplitNodeInfoT<key_t, value_t, index_t>, NUM_GPUS>& node_info,
        split_table_tt<table_t, NUM_GPUS>& split_table,
        std::array<table_t, NUM_GPUS>& src_lens,
        std::array<table_t, NUM_GPUS>& dest_lens,
        funct_t functor) const noexcept
    {
        using MultiSplitNodeInfo = MultiSplitNodeInfoT<key_t, value_t, index_t>;

        for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
        {
            if (node_info[src_gpu].src_len > node_info[src_gpu].dest_len)
            {
                error("dsts_lens too small for given srcs_lens.");
            }
        }

        uint32_t* d_offsets[effective_buckets];

        for (uint gpu = 0; gpu < num_gpus; ++gpu)
        {
            cudaSetDevice(context.get_device_id(gpu));
            const MultiSplitNodeInfo& info = node_info[gpu];
            if (info.src_len > 0)
            {
                if (world_rank == gpu) {
                    //only the process with this gpu should allocate gpu memory 
                    QDAllocator& d_alloc = context.get_device_temp_allocator(gpu);
                    d_offsets[gpu] = d_alloc.get<uint32_t>(effective_buckets);

                    if (include_values)
                    {
                        DispatchMultisplit::multisplit_kv(info.src_keys, info.dest_keys, info.src_values, info.dest_values,
                            d_offsets[gpu], functor, info.src_len,
                            context.get_device_temp_allocator(gpu),
                            context.get_gpu_default_stream(gpu));
                    }
                    else
                    {
                        DispatchMultisplit::multisplit(info.src_keys, info.dest_keys, d_offsets[gpu], functor,
                            info.src_len, context.get_device_temp_allocator(gpu),
                            context.get_gpu_default_stream(gpu));
                    }

                    cudaMemcpyAsync(h_offsets[gpu],
                        d_offsets[gpu],
                        sizeof(uint32_t) * num_gpus, cudaMemcpyDeviceToHost,
                        context.get_gpu_default_stream(gpu));
                    CUERR;
                }
            }
            else
            {
                memset(h_offsets[gpu], 0, sizeof(uint32_t) * num_gpus);
            }
            dest_lens[gpu] = 0;
        }
        // this sync is mandatory
        context.sync_default_streams();

        // world_rank() == num_gpus
        std::span<uint32_t> sb(h_offsets[world_rank()], num_gpus * sizeof(uint32_t));
        std::vector<uint32_t> recv;
        recv.reserve(num_gpus * world_size());
        comm_world().allgather(send_buf(sb), recv_buf<no_resize>(recv));
        for (int i = 0; i < world_size(); i++) {
            if (i == world_rank()) {
                continue;
            }
            memcpy(h_offsets[i], recv.data() + i * num_gpus, num_gpus * sizeof(uint32_t));
        }

        for (int i = 0; i < world_size(); i++) {
            for (int j = 0; j < num_gpus; j++) {
                printf("h_offsets[%d]: %u, rank: %lu\n", i, h_offsets[i][j], world_rank());
            }
        }

        // recover the partition table from accumulated counters
        for (uint gpu = 0; gpu < num_gpus; ++gpu)
        {
            for (uint64_t part = 0; part < num_gpus; ++part)
            {
                split_table[gpu][part] = part + 1 < num_gpus ? h_offsets[gpu][part + 1] - h_offsets[gpu][part]
                    : node_info[gpu].src_len - h_offsets[gpu][part];
                dest_lens[part] += split_table[gpu][part];
            }
            src_lens[gpu] = node_info[gpu].src_len;
        }

        return true;
    }
};
