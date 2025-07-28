#include <cuda_runtime.h>
#include <cstdio>
#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <ctime>
#include <span>

#include <kamping/checking_casts.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/allgather.hpp>
#include <kamping/data_buffer.hpp> 
#include <kamping/environment.hpp>
#include <kamping/measurements/printer.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/named_parameters.hpp>
#include <kamping/communicator.hpp>
#include <kamping/p2p/recv.hpp>
#include <kamping/p2p/send.hpp>
#include <kamping/request_pool.hpp>

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

__global__ void printArray(size_t* a, size_t length, size_t rank) {
    if (length >= 64) {
        for (int i = 0; i < 64; i++) {
            printf("[%lu] A[%d]:%lf\n", rank, i, a[i]);
        }
    }
    else {
        for (int i = 0; i < length; i++) {
            printf("[%lu] A[%d]:%lf\n", rank, i, a[i]);
        }
    }
}

const size_t NUM_GPUS = 4;

int main(int argc, char** argv)
{
    using namespace kamping;
    kamping::Environment e;
    Communicator comm;

    int deviceCount;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    int deviceId = world_rank() % deviceCount;
    std::cout << "Device Id: " << deviceId << std::endl;
    CUDACHECK(cudaSetDevice(deviceId));
    std::array<cudaStream_t, NUM_GPUS> streams;
    for (size_t i = 0; i < deviceCount; i++)
    {
        CUDACHECK(cudaStreamCreate(&streams[i]));
        if (world_rank() == i) {
            continue;
        }
        int canAccess;
        cudaDeviceCanAccessPeer(&canAccess, world_rank(), i);
        if (canAccess) {
            CUDACHECK(cudaDeviceEnablePeerAccess(i, 0));
            printf("[%lu] peer to [%lu] enabled\n", world_rank(), i);
        }
    }

    size_t N = size_t(1 << 30);

    // Allocate memory for A on CPU
    size_t* A = (size_t*)malloc(N * sizeof(size_t));

    // Initialize all elements of A
    for (int i = 0; i < N; i++) {
        A[i] = i + N * world_rank();
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(A, &A[N - 1], g);
    size_t* d_A;
    CUDACHECK(cudaMalloc(&d_A, N * sizeof(size_t)));
    CUDACHECK(cudaMemcpy(d_A, A, N * sizeof(size_t), cudaMemcpyHostToDevice));


    cudaIpcMemHandle_t ownHandle;
    CUDACHECK(cudaIpcGetMemHandle(&ownHandle, d_A));
    std::array<cudaIpcMemHandle_t, NUM_GPUS> handles;
    std::array<size_t*, NUM_GPUS> pointer;
    comm_world().allgather(send_buf(std::span<cudaIpcMemHandle_t>(&ownHandle, 1)), recv_buf(handles));
    printf("[%lu] received handle\n", world_rank());
    for (size_t i = 0; i < NUM_GPUS; i++)
    {
        if (world_rank() == i) {
            pointer[i] = d_A;
            continue;
        }
        void* rawothersd_A;
        cudaIpcOpenMemHandle(&rawothersd_A, handles[i], cudaIpcMemLazyEnablePeerAccess);
        size_t* other_d_A = reinterpret_cast<size_t*>(rawothersd_A);
        pointer[i] = other_d_A;
    }
    printf("[%lu] opened handles\n", world_rank());
    comm_world().barrier();

    // for (size_t i = 0; i < N; i++)
    // {
    //     //size_t dest = A[i]
    //     CUDACHECK(cudaMemcpyPeerAsync(pointer[j] + world_rank() * size_t(N / NUM_GPUS), j, pointer[world_rank()] + world_rank() * size_t(N / NUM_GPUS), world_rank(), sizeof(size_t) * size_t(N / NUM_GPUS), streams[j]));
    // }
    for (size_t j = 0; j < NUM_GPUS; j++)
    {
        CUDACHECK(cudaMemcpyPeerAsync(pointer[j] + world_rank() * size_t(N / NUM_GPUS), j, pointer[world_rank()] + world_rank() * size_t(N / NUM_GPUS), world_rank(), sizeof(size_t) * size_t(N / NUM_GPUS), streams[j]));
    }

    for (auto stream : streams)
    {
        CUDACHECK(cudaStreamSynchronize(stream));
    }
    comm_world().barrier();
    printArray << <1, 1, 0, streams[0] >> > (pointer[world_rank()], N, world_rank());
    CUDACHECK(cudaStreamSynchronize(streams[0]));


    comm_world().barrier();
    for (size_t i = 0; i < NUM_GPUS; i++)
    {
        if (world_rank() == i) {
            continue;
        }
        CUDACHECK(cudaIpcCloseMemHandle(pointer[i]));
    }
    comm_world().barrier();
    CUDACHECK(cudaFree(d_A));
    return 0;
}
