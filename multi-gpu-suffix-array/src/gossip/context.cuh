#pragma once

#include <array>
#include <iostream>
#include "../cuda_helpers.h"
#include <mpi.h>

#include <kamping/checking_casts.hpp>
#include <kamping/communicator.hpp>
#include <kamping/collectives/bcast.hpp>
#include <kamping/environment.hpp>
#include <span>
#include <nccl.h>

#include "../my_mgpu_context.hxx"

#include "qdallocator.hpp"

using uint = unsigned int;
using namespace kamping;

template <uint NUM_GPUS,
    bool THROW_EXCEPTIONS = true,
    uint PEER_STATUS_SLOW_ = 0,
    uint PEER_STATUS_FAST_ = 1,
    uint PEER_STATUS_DIAG_ = 2>
class MultiGPUContext
{
public:
    using device_id_t = uint;
    enum PeerStatus
    {
        PEER_STATUS_SLOW = PEER_STATUS_SLOW_,
        PEER_STATUS_FAST = PEER_STATUS_FAST_,
        PEER_STATUS_DIAG = PEER_STATUS_DIAG_
    };

private:
    std::array<std::array<cudaStream_t, NUM_GPUS>, NUM_GPUS> streams;
    std::array<device_id_t, NUM_GPUS> device_ids;


    std::array<std::array<uint, NUM_GPUS>, NUM_GPUS> peer_status;

    std::array<std::array<mgpu::my_mpgu_context_t*, NUM_GPUS>, NUM_GPUS> mpgu_contexts;
    std::array<QDAllocator, NUM_GPUS> mdevice_temp_allocators;
    bool in_node;
    ncclComm_t nccl_comm;

public:
    static const uint num_gpus = NUM_GPUS;
    MultiGPUContext(const MultiGPUContext&) = delete;
    MultiGPUContext& operator=(const MultiGPUContext&) = delete;

    MultiGPUContext(ncclComm_t comm, const std::array<device_id_t, NUM_GPUS>* device_ids_ = nullptr, uint num_per_node = 4)
    {
        uint node_id = uint(world_rank() / num_per_node);


        nccl_comm = comm;
        // Copy num_gpus many device identifiers

        for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
        {
            device_ids[src_gpu] = device_ids_ ? (*device_ids_)[src_gpu] : (src_gpu % num_per_node);
        }

        // Create num_gpus^2 streams where streams[gpu*num_gpus+part]
        // denotes the stream to be used for GPU gpu and partition part.

        for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
        {
            if (world_rank() == src_gpu)
            {
                cudaDeviceSynchronize();
            }
            for (uint part = 0; part < num_gpus; ++part)
            {
                if (world_rank() == src_gpu)
                {
                    cudaStreamCreate(&streams[src_gpu][part]);
                }
                mpgu_contexts[src_gpu][part] = new mgpu::my_mpgu_context_t(streams[src_gpu][part],
                    mdevice_temp_allocators[src_gpu]);
            }
        }
        CUERR;

        return;
        // compute the connectivity matrix

        uint src = get_device_id(world_rank());
        for (uint dst_gpu = 0; dst_gpu < num_gpus; dst_gpu++) {
            if (dst_gpu == world_rank()) {
                peer_status[world_rank()][dst_gpu] = PEER_STATUS_DIAG;
                continue;
            }

            // inter node gpus are not directly connected
            if (node_id != dst_gpu / num_per_node) {
                peer_status[world_rank()][dst_gpu] = PEER_STATUS_SLOW;
                continue;
            }

            uint dst = get_device_id(dst_gpu);
            int status;
            cudaDeviceCanAccessPeer(&status, src, dst);
            if (!status) {
                std::cerr << "Could not enable peer access between " << world_rank() << " and " << dst_gpu << std::endl;
                exit(1);
            }
            else {
                peer_status[world_rank()][dst_gpu] = PEER_STATUS_FAST;
                printf("[%lu] peer access to [%u] activated: %u\n", world_rank(), dst_gpu, peer_status[world_rank()][dst_gpu]);
            }
        }


        CUERR;

        //for (uint src_gpu = 0; src_gpu < NUM_GPUS_PER_NODE; ++src_gpu)
        {
            uint src_gpu = world_rank();
            const device_id_t src = get_device_id(src_gpu);
            // //(src);
            for (uint dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu)
            {
                device_id_t dst = get_device_id(dst_gpu);

                // in node ids should be unique
                if (src_gpu != dst_gpu && node_id == dst_gpu / num_per_node)
                {
                    if (THROW_EXCEPTIONS)
                    {
                        if (src == dst) {
                            continue;
                            throw std::invalid_argument("Device identifiers are not unique inside a node.");
                        }
                    }
                }

                if (peer_status[src_gpu][dst_gpu] == PEER_STATUS_FAST)
                {
                    cudaDeviceEnablePeerAccess(dst, 0);

                    // Consume error for rendundant peer access initialization.
                    const cudaError_t cuerr = cudaGetLastError();

                    if (cuerr == cudaErrorPeerAccessAlreadyEnabled)
                    {
                        std::cout << "STATUS: redundant enabling of peer access from GPU " << src_gpu
                            << " to GPU " << dst << " attempted." << std::endl;
                    }
                    else if (cuerr)
                    {
                        std::cout << "CUDA error: "
                            << cudaGetErrorString(cuerr) << " : "
                            << __FILE__ << ", line "
                            << __LINE__ << std::endl;
                    }
                }
            }
        }
        CUERR;

        for (size_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
        {
            std::span<uint> srb(&peer_status[src_gpu][0], num_gpus);
            comm_world().bcast(send_recv_buf(srb), send_recv_count(num_gpus), root(src_gpu));
        }

        std::array<bool, num_gpus> in_nodes;
        for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
            for (uint dest_gpu = 0; dest_gpu < num_gpus; ++dest_gpu) {
                if (peer_status[src_gpu][dest_gpu] >= PEER_STATUS_FAST) {
                    in_nodes[src_gpu] = true;
                }
                else {
                    in_nodes[src_gpu] = false;
                    break;
                }
            }
        }
        in_node = true;
        for (auto b : in_nodes) {
            in_node &= b;
        }
    }

    ~MultiGPUContext()
    {

        // Synchronize and destroy streams
        // for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
        {
            uint src_gpu = world_rank();
            // //(get_device_id(src_gpu));
            cudaDeviceSynchronize();
            CUERR;
            for (uint part = 0; part < num_gpus; ++part)
            {
                cudaStreamSynchronize(get_streams(src_gpu)[part]);
                CUERR;
                delete mpgu_contexts[src_gpu][part];
                cudaStreamDestroy(get_streams(src_gpu)[part]);
                CUERR;
            }
        }
        CUERR;

        // disable peer access
        //for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
        {
            uint src_gpu = world_rank();
            device_id_t src = get_device_id(src_gpu);
            // //(src);
            for (uint dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu)
            {
                device_id_t dst = get_device_id(dst_gpu);

                if (peer_status[src_gpu][dst_gpu] == PEER_STATUS_FAST)
                {
                    cudaDeviceDisablePeerAccess(dst);

                    // consume error for rendundant
                    // peer access deactivation
                    const cudaError_t cuerr = cudaGetLastError();
                    if (cuerr == cudaErrorPeerAccessNotEnabled)
                    {
                        std::cout << "STATUS: redundant disabling of peer access from GPU " << src_gpu
                            << " to GPU " << dst << " attempted." << std::endl;
                    }
                    else if (cuerr)
                    {
                        std::cout << "CUDA error: " << cudaGetErrorString(cuerr) << " : "
                            << __FILE__ << ", line " << __LINE__ << std::endl;
                    }
                }
            }
        }
        CUERR
    }

    bool is_in_node() {
        return in_node;
    }

    ncclComm_t get_nccl() {
        return nccl_comm;
    }

    device_id_t get_device_id(uint gpu) const noexcept
    {
        // return the actual device identifier of GPU gpu
        return device_ids[gpu];
    }
    const std::array<cudaStream_t, NUM_GPUS>& get_streams(uint gpu) const noexcept
    {
        return streams[gpu];
    }

    const cudaStream_t& get_gpu_default_stream(uint gpu) const noexcept
    {
        return streams[gpu][0];
    }

    mgpu::my_mpgu_context_t& get_mgpu_default_context_for_device(uint gpu) const noexcept
    {
        return *mpgu_contexts[gpu][0];
    }

    const std::array<mgpu::my_mpgu_context_t*, NUM_GPUS>& get_mgpu_contexts_for_device(uint gpu) const noexcept
    {
        return mpgu_contexts[gpu];
    }

    QDAllocator& get_device_temp_allocator(uint gpu) noexcept
    {
        return mdevice_temp_allocators[gpu];
    }

    uint get_peer_status(uint src_gpu, uint dest_gpu) const noexcept
    {
        return peer_status[src_gpu][dest_gpu];
    }

    void sync_gpu_default_stream(uint gpu) const noexcept
    {
        if (gpu == world_rank())
        {
            //(get_device_id(gpu));
            CUERR;
            cudaStreamSynchronize(get_gpu_default_stream(gpu));
            CUERR;
        }
    }

    void sync_default_streams() const noexcept
    {
        //(0);
        CUERR;
        cudaStreamSynchronize(get_gpu_default_stream(world_rank()));
        CUERR;
    }

    void sync_gpu_streams(uint gpu) const noexcept
    {
        // sync all streams associated with the corresponding GPU
        if (gpu == world_rank())
        {
            //(get_device_id(gpu));
            CUERR;
            for (uint part = 0; part < num_gpus; ++part)
            {
                if (!get_streams(gpu)[part]) {
                    printf("cuda sync %lu, %lu\n", world_rank(), get_streams(gpu).size());
                }
                cudaStreamSynchronize(get_streams(gpu)[part]);
                CUERR;
            }
        }
    }

    void sync_all_streams_mpi_safe() const noexcept
    {
        for (uint part = 0; part < num_gpus; ++part)
        {
            cudaError_t err = cudaErrorNotReady;
            int flag;
            while (err == cudaErrorNotReady) {
                err = cudaStreamQuery(get_streams(world_rank())[part]);
                MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
            }
        }
    }

    void sync_default_stream_mpi_safe() const noexcept
    {
        cudaError_t err = cudaErrorNotReady;
        int flag;
        while (err == cudaErrorNotReady) {
            err = cudaStreamQuery(get_streams(world_rank())[0]);
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        }
    }

    void sync_all_streams() const noexcept
    {
        // sync all streams of the context
        // for (uint gpu = 0; gpu < num_gpus; ++gpu)
        sync_gpu_streams(world_rank());
        CUERR;
    }

    void sync_hard() const noexcept
    {
        // sync all GPUs

        //(0);
        cudaDeviceSynchronize();
        CUERR;

        comm_world().barrier();
    }

    void print_connectivity_matrix() const
    {
        std::cout << "STATUS: connectivity matrix:" << std::endl;
        for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
        {
            for (uint dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu)
            {
                std::cout << (dst_gpu == 0 ? "STATUS: |" : "")
                    << uint(peer_status[src_gpu][dst_gpu])
                    << (dst_gpu + 1 == num_gpus ? "|\n" : " ");
            }
        }
    }
};
