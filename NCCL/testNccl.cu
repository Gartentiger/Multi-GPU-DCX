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
#include <nccl.h>

#include <kamping/checking_casts.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/data_buffer.hpp> 
#include <kamping/environment.hpp>
#include <kamping/measurements/printer.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/named_parameters.hpp>
#include <kamping/communicator.hpp>
#include <kamping/p2p/irecv.hpp>
#include <kamping/p2p/isend.hpp>
#include <kamping/request_pool.hpp>

#include "mpi-ext.h" /* Needed for CUDA-aware check */


static const size_t SEND_SIZE = 1024;
static const size_t SEND_TIMES = 1024;

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



    printf("Compile time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library has CUDA-aware support.\n", MPIX_CUDA_AWARE_SUPPORT);
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library does not have CUDA-aware support.\n");
#else
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */

    printf("Run time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (1 == MPIX_Query_cuda_support()) {
        printf("This MPI library has CUDA-aware support.\n");
    }
    else {
        printf("This MPI library does not have CUDA-aware support.\n");
    }
#else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */


#ifdef USE_NCCL
    ncclComm_t nccl_comm;
    ncclUniqueId Id;
    if (world_rank() == 0) {
        std::span<ncclUniqueId> unique(&Id, 1);
        ncclGetUniqueId(&Id);
        comm_world().bcast_single(send_recv_buf(Id));
    }
    else {
        Id = comm_world().bcast_single<ncclUniqueId>();
    }

    NCCLCHECK(ncclCommInitRank(&nccl_comm, world_size(), Id, world_rank()));

#endif
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



    //auto& t = kamping::measurements::timer();
    //t.synchronize_and_start("pingping");
#ifdef USE_NCCL

    // Loop From https://github.com/olcf-tutorials/MPI_ping_pong/blob/master/cuda_staged/ping_pong_cuda_staged.cu
    for (int i = 0; i <= 27; i++) {

        long int N = 1 << i;

        // Allocate memory for A on CPU
        double* A = (double*)malloc(N * sizeof(double));

        // Initialize all elements of A to random values
        for (int i = 0; i < N; i++) {
            A[i] = (double)rand() / (double)RAND_MAX;
        }

        double* d_A;
        CUDACHECK(cudaMalloc(&d_A, N * sizeof(double)));
        CUDACHECK(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));

        int tag1 = 10;
        int tag2 = 20;

        int loop_count = 50;
        // Warm-up loop
        for (int i = 1; i <= 5; i++) {
            if (world_rank() == 0) {
                CUDACHECK(cudaMemcpy(A, d_A, N * sizeof(double), cudaMemcpyDeviceToHost));
                ncclSend(A, N, ncclDouble, 1, nccl_comm, stream);
                ncclRecv(A, N, ncclDouble, 1, nccl_comm, stream);
                CUDACHECK(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));
            }
            else if (world_rank() == 1) {
                ncclRecv(A, N, ncclDouble, 0, nccl_comm, stream);
                CUDACHECK(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));
                CUDACHECK(cudaMemcpy(A, d_A, N * sizeof(double), cudaMemcpyDeviceToHost));
                ncclSend(A, N, ncclDouble, 0, nccl_comm, stream);
            }
        }

        // Time ping-pong for loop_count iterations of data transfer size 8*N bytes
        double start_time, stop_time, elapsed_time;
        start_time = MPI_Wtime();

        for (int i = 1; i <= loop_count; i++) {
            if (world_rank() == 0) {
                CUDACHECK(cudaMemcpy(A, d_A, N * sizeof(double), cudaMemcpyDeviceToHost));
                ncclSend(A, N, ncclDouble, 1, nccl_comm, stream);
                ncclRecv(A, N, ncclDouble, 1, nccl_comm, stream);
                CUDACHECK(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));
            }
            else if (world_rank() == 1) {
                ncclRecv(A, N, ncclDouble, 0, nccl_comm, stream);
                CUDACHECK(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));
                CUDACHECK(cudaMemcpy(A, d_A, N * sizeof(double), cudaMemcpyDeviceToHost));
                ncclSend(A, N, ncclDouble, 0, nccl_comm, stream);
            }
        }

        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;

        long int num_B = 8 * N;
        long int B_in_GB = 1 << 30;
        double num_GB = (double)num_B / (double)B_in_GB;
        double avg_time_per_transfer = elapsed_time / (2.0 * (double)loop_count);

        if (world_rank() == 0) printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB / avg_time_per_transfer);

        CUDACHECK(cudaFree(d_A));
        free(A);
    }

#else 
    // Loop From https://github.com/olcf-tutorials/MPI_ping_pong/blob/master/cuda_staged/ping_pong_cuda_staged.cu
    for (int i = 0; i <= 27; i++) {

        long int N = 1 << i;

        // Allocate memory for A on CPU
        double* A = (double*)malloc(N * sizeof(double));

        // Initialize all elements of A to random values
        for (int i = 0; i < N; i++) {
            A[i] = (double)rand() / (double)RAND_MAX;
        }

        double* d_A;
        CUDACHECK(cudaMalloc(&d_A, N * sizeof(double)));
        CUDACHECK(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));

        int tag1 = 10;
        int tag2 = 20;

        int loop_count = 50;
        MPI_Status stat;
        // Warm-up loop
        for (int i = 1; i <= 5; i++) {
            if (world_rank() == 0) {
                CUDACHECK(cudaMemcpy(A, d_A, N * sizeof(double), cudaMemcpyDeviceToHost));
                MPI_Send(A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
                MPI_Recv(A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
                CUDACHECK(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));
            }
            else if (world_rank() == 1) {
                MPI_Recv(A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
                CUDACHECK(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));
                CUDACHECK(cudaMemcpy(A, d_A, N * sizeof(double), cudaMemcpyDeviceToHost));
                MPI_Send(A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
            }
        }

        // Time ping-pong for loop_count iterations of data transfer size 8*N bytes
        double start_time, stop_time, elapsed_time;
        start_time = MPI_Wtime();

        for (int i = 1; i <= loop_count; i++) {
            if (world_rank() == 0) {
                CUDACHECK(cudaMemcpy(A, d_A, N * sizeof(double), cudaMemcpyDeviceToHost));
                MPI_Send(A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
                MPI_Recv(A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
                CUDACHECK(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));
            }
            else if (world_rank() == 1) {
                MPI_Recv(A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
                CUDACHECK(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));
                CUDACHECK(cudaMemcpy(A, d_A, N * sizeof(double), cudaMemcpyDeviceToHost));
                MPI_Send(A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
            }
        }

        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;

        long int num_B = 8 * N;
        long int B_in_GB = 1 << 30;
        double num_GB = (double)num_B / (double)B_in_GB;
        double avg_time_per_transfer = elapsed_time / (2.0 * (double)loop_count);

        if (world_rank() == 0) printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB / avg_time_per_transfer);

        CUDACHECK(cudaFree(d_A));
        free(A);
    }
#endif


    // std::cout << "[" << world_rank() << "]" << "Ping pong complete" << std::endl;
    // std::ofstream outFile(argv[1], std::ios::app);
    // t.aggregate_and_print(
    //     kamping::measurements::SimpleJsonPrinter{ outFile, {} });
    // std::cout << std::endl;
    // t.aggregate_and_print(kamping::measurements::FlatPrinter{});
    // std::cout << std::endl;
    return 0;
    }
