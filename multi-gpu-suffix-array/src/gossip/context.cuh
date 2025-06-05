#pragma once

#include <array>
#include <iostream>
#include "../cuda_helpers.h"
#include <mpi.h>

#include <kamping/checking_casts.hpp>
#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>

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

public:
    static const uint num_gpus = NUM_GPUS;
    MultiGPUContext(const MultiGPUContext&) = delete;
    MultiGPUContext& operator=(const MultiGPUContext&) = delete;

    MultiGPUContext(const std::array<device_id_t, NUM_GPUS>* device_ids_ = nullptr)
    {
        // Copy num_gpus many device identifiers

        for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
        {
            device_ids[src_gpu] = device_ids_ ? (*device_ids_)[src_gpu] : src_gpu;
        }

        // Create num_gpus^2 streams where streams[gpu*num_gpus+part]
        // denotes the stream to be used for GPU gpu and partition part.

        for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
        {
            if (world_rank() == src_gpu)
            {
                cudaSetDevice(0); // get_device_id(src_gpu));
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

        // compute the connectivity matrix
        for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
        {
            // device_id_t src = get_device_id(src_gpu);
            // cudaSetDevice(src);
            for (uint dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu)
            {
                //     uint dst = get_device_id(dst_gpu);

                //     // Check if src can access dst.
                if (src_gpu == dst_gpu)
                {
                    peer_status[src_gpu][dst_gpu] = PEER_STATUS_DIAG;
                }
                else
                {
                    //     else {
                    //         int status;
                    //         cudaDeviceCanAccessPeer(&status, src, dst);
                    //         peer_status[src_gpu][dst_gpu] = status ? PEER_STATUS_FAST : PEER_STATUS_SLOW;
                    //     }

                    peer_status[src_gpu][dst_gpu] = PEER_STATUS_SLOW;
                }
            }
        }
        CUERR;

        // for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
        // {
        //     const device_id_t src = get_device_id(src_gpu);
        //     cudaSetDevice(src);
        //     for (uint dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu)
        //     {
        //         device_id_t dst = get_device_id(dst_gpu);

        //         if (src_gpu != dst_gpu)
        //         {
        //             if (THROW_EXCEPTIONS)
        //             {
        //                 if (src == dst)
        //                     throw std::invalid_argument("Device identifiers are not unique.");
        //             }
        //         }

        //         if (peer_status[src_gpu][dst_gpu] == PEER_STATUS_FAST)
        //         {
        //             cudaDeviceEnablePeerAccess(dst, 0);

        //             // Consume error for rendundant peer access initialization.
        //             const cudaError_t cuerr = cudaGetLastError();

        //             if (cuerr == cudaErrorPeerAccessAlreadyEnabled)
        //             {
        //                 std::cout << "STATUS: redundant enabling of peer access from GPU " << src_gpu
        //                           << " to GPU " << dst << " attempted." << std::endl;
        //             }
        //             else if (cuerr)
        //             {
        //                 std::cout << "CUDA error: "
        //                           << cudaGetErrorString(cuerr) << " : "
        //                           << __FILE__ << ", line "
        //                           << __LINE__ << std::endl;
        //             }
        //         }
        //     }
        // }
        // CUERR;
    }

    ~MultiGPUContext()
    {

        // Synchronize and destroy streams
        for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
        {
            cudaSetDevice(get_device_id(src_gpu));
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
        for (uint src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
        {
            device_id_t src = get_device_id(src_gpu);
            cudaSetDevice(src);
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
            cudaSetDevice(get_device_id(gpu));
            CUERR;
            cudaStreamSynchronize(get_gpu_default_stream(gpu));
            CUERR;
        }
    }

    void sync_default_streams() const noexcept
    {
        cudaSetDevice(0);
        CUERR;
        cudaStreamSynchronize(get_gpu_default_stream(world_rank()));
        CUERR;
        comm_world().barrier();
    }

    void sync_gpu_streams(uint gpu) const noexcept
    {
        // sync all streams associated with the corresponding GPU
        if (gpu == world_rank())
        {
            cudaSetDevice(get_device_id(gpu));
            CUERR;
            for (uint part = 0; part < num_gpus; ++part)
            {
                printf("cuda sync %lu, %d\n", world_rank(), part);
                cudaStreamSynchronize(get_streams(gpu)[part]);
                CUERR;
            }
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

        cudaSetDevice(0);
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
