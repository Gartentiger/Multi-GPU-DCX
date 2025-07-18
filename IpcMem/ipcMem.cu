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

__global__ void printArray(double* a, size_t length, size_t rank) {
    for (int i = 0; i < length; i++) {
        printf("[%lu] A[%d]:%lf\n", rank, i, a[i]);
    }
}

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
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    if (deviceCount > 1) {
        if (world_rank() == 0) {
            int canAccess;
            cudaDeviceCanAccessPeer(&canAccess, world_rank(), 1);
            if (canAccess) {
                CUDACHECK(cudaDeviceEnablePeerAccess(1, 0));
                printf("[%lu] peer to peer enabled\n", world_rank());
            }

        }
        else if (world_rank() == 1) {
            int canAccess;
            cudaDeviceCanAccessPeer(&canAccess, world_rank(), 0);
            if (canAccess) {
                CUDACHECK(cudaDeviceEnablePeerAccess(0, 0));
                printf("[%lu] peer to peer enabled\n", world_rank());
            }
        }
    }
    long int N = 1 << 5;

    // Allocate memory for A on CPU
    double* A = (double*)malloc(N * sizeof(double));

    // Initialize all elements of A
    for (int i = 0; i < N; i++) {
        A[i] = world_rank();
    }

    double* d_A;
    CUDACHECK(cudaMalloc(&d_A, N * sizeof(double)));
    CUDACHECK(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));

    cudaIpcMemHandle_t ownHandle;
    cudaIpcMemHandle_t otherHandle;
    CUDACHECK(cudaIpcGetMemHandle(&ownHandle, d_A));
    if (world_rank() == 0) {

        comm_world().send(send_buf(std::span<cudaIpcMemHandle_t>(&ownHandle, 1)), send_count(1), destination(1));
        comm_world().recv(recv_buf(std::span<cudaIpcMemHandle_t>(&otherHandle, 1)), recv_count(1));
    }
    else {
        comm_world().recv(recv_buf(std::span<cudaIpcMemHandle_t>(&otherHandle, 1)), recv_count(1));
        comm_world().send(send_buf(std::span<cudaIpcMemHandle_t>(&ownHandle, 1)), send_count(1), destination(0));
    }
    printf("[%lu] received handle\n", world_rank());
    void* rawothersd_A;
    cudaIpcOpenMemHandle(&rawothersd_A, otherHandle, cudaIpcMemLazyEnablePeerAccess);
    double* other_d_A = reinterpret_cast<double*>(rawothersd_A);
    printf("[%lu] opened handle\n", world_rank());
    printArray << <1, 1 >> > (other_d_A, N, world_rank());
    comm_world().barrier();
    if (world_rank() == 0) {
        cudaMemcpyPeer(other_d_A, 1, d_A, 0, N * sizeof(double));
    }
    comm_world().barrier();
    if (world_rank() == 0) {
        printArray << <1, 1 >> > (other_d_A, N, world_rank());
    }
    else {
        printArray << <1, 1 >> > (d_A, N, world_rank());
    }
    cudaIpcCloseMemHandle(other_d_A);
    comm_world().barrier();
    cudaFree(d_A);
    return 0;
}
