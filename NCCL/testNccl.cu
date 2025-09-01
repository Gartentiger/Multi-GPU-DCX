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
#include <kamping/p2p/recv.hpp>
#include <kamping/p2p/send.hpp>
#include <kamping/request_pool.hpp>
// #include <nvToolsExt.h>


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
    // nvtxRangePush("SuffixArray");
    // nvtxRangePop();
    int deviceCount;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    // int deviceId = world_rank() % (size_t)deviceCount;
    // std::cout << "Device Id: " << deviceId << std::endl;
    CUDACHECK(cudaSetDevice(0));


#ifdef USE_NCCL
    //ncclComm_t nccl_comm;
    //ncclUniqueId Id;
    if (world_rank() == 0) {
        //std::span<ncclUniqueId> unique(&Id, 1);
        //NCCLCHECK(ncclGetUniqueId(&Id));
        //comm_world().bcast_single(send_recv_buf(Id));
    }
    else {
        //Id = comm_world().bcast_single<ncclUniqueId>();
    }

    //NCCLCHECK(ncclCommInitRank(&nccl_comm, world_size(), Id, world_rank()));

#endif
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));
    CUDACHECK(cudaSetDevice(1));
    cudaStream_t stream1;
    CUDACHECK(cudaStreamCreate(&stream1));
    CUDACHECK(cudaSetDevice(0));
    if (deviceCount > 1) {
        // if (world_rank() == 0) {
        int canAccess;
        cudaDeviceCanAccessPeer(&canAccess, 0, 1);
        if (canAccess) {
            CUDACHECK(cudaDeviceEnablePeerAccess(1, 0));
            printf("[%lu] peer to peer enabled\n", world_rank());
        }

        // }
        // else if (world_rank() == 1) 
        {
            CUDACHECK(cudaSetDevice(1));
            int canAccess;
            cudaDeviceCanAccessPeer(&canAccess, 1, 0);
            if (canAccess) {
                CUDACHECK(cudaDeviceEnablePeerAccess(0, 0));
                printf("[%lu] peer to peer enabled\n", world_rank());
            }
        }
        // 
    }



    //auto& t = kamping::measurements::timer();
    //t.synchronize_and_start("pingping");
#ifdef USE_NCCL

    // Loop From https://github.com/olcf-tutorials/MPI_ping_pong/blob/master/cuda_staged/ping_pong_cuda_staged.cu
    for (int i = 0; i <= 27; i++) {

        long int N = 1 << i;
        CUDACHECK(cudaSetDevice(0));
        // Allocate memory for A on CPU
        double* A = (double*)malloc(N * sizeof(double));

        double* B = (double*)malloc(N * sizeof(double));
        // Initialize all elements of A to random values
        for (int i = 0; i < N; i++) {
            A[i] = (double)rand() / (double)RAND_MAX;
            B[i] = (double)rand() / (double)RAND_MAX;
        }

        double* d_A;
        CUDACHECK(cudaMalloc(&d_A, N * sizeof(double)));
        CUDACHECK(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));

        CUDACHECK(cudaSetDevice(1));
        double* d_B;
        CUDACHECK(cudaMalloc(&d_B, N * sizeof(double)));
        CUDACHECK(cudaMemcpy(d_B, B, N * sizeof(double), cudaMemcpyHostToDevice));
        int tag1 = 10;
        int tag2 = 20;

        int loop_count = 50;
        // Warm-up loop
        //ncclGroupStart();
        CUDACHECK(cudaSetDevice(0));
        for (int i = 1; i <= 5; i++) {
            CUDACHECK(cudaMemcpyPeerAsync(d_B, 1, d_A, 0, N * sizeof(double), stream));
            CUDACHECK(cudaMemcpyPeerAsync(d_A, 0, d_B, 1, N * sizeof(double), stream1));
            // if (world_rank() == 0) {

                // NCCLCHECK(ncclSend(d_A, N, ncclDouble, 1, nccl_comm, stream));
                // NCCLCHECK(ncclRecv(d_A, N, ncclDouble, 1, nccl_comm, stream));
            // }
            // else if (world_rank() == 1) {
                // NCCLCHECK(ncclRecv(d_A, N, ncclDouble, 0, nccl_comm, stream));
                // NCCLCHECK(ncclSend(d_A, N, ncclDouble, 0, nccl_comm, stream));
            // }
        }

        cudaStreamSynchronize(stream);
        CUDACHECK(cudaSetDevice(1));
        cudaStreamSynchronize(stream1);
        CUDACHECK(cudaSetDevice(0));
        //ncclGroupEnd();
        // Time ping-pong for loop_count iterations of data transfer size 8*N bytes
        double start_time, stop_time, elapsed_time;
        start_time = MPI_Wtime();
        // ncclGroupStart();
        for (int i = 1; i <= loop_count; i++) {
            // if (world_rank() == 0) {
            CUDACHECK(cudaMemcpyPeerAsync(d_B, 1, d_A, 0, N * sizeof(double), stream));
            CUDACHECK(cudaMemcpyPeerAsync(d_A, 0, d_B, 1, N * sizeof(double), stream1));
            // NCCLCHECK(ncclSend(d_A, N, ncclDouble, 1, nccl_comm, stream));
            // NCCLCHECK(ncclRecv(d_A, N, ncclDouble, 1, nccl_comm, stream));
        // }
        // else if (world_rank() == 1) {
            // NCCLCHECK(ncclRecv(d_A, N, ncclDouble, 0, nccl_comm, stream));
            // NCCLCHECK(ncclSend(d_A, N, ncclDouble, 0, nccl_comm, stream));
        // }
        }
        cudaStreamSynchronize(stream);
        CUDACHECK(cudaSetDevice(1));
        cudaStreamSynchronize(stream1);
        CUDACHECK(cudaSetDevice(0));
        // ncclGroupEnd();
        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;

        long int num_B = 8 * N;
        long int B_in_GB = 1 << 30;
        double num_GB = (double)num_B / (double)B_in_GB;
        double avg_time_per_transfer = elapsed_time / (2.0 * (double)loop_count);

        printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB / avg_time_per_transfer);

        CUDACHECK(cudaFree(d_A));
        CUDACHECK(cudaSetDevice(1));
        CUDACHECK(cudaFree(d_B));
        free(A);
        free(B);
    }

#else 
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
                comm_world().send(send_buf(std::span<double>(d_A, N)), send_count(N), destination(1), tag(tag1));
                comm_world().recv(recv_buf(std::span<double>(d_A, N)), recv_count(N), source(1), tag(tag2));
            }
            else if (world_rank() == 1) {
                comm_world().recv(recv_buf(std::span<double>(d_A, N)), recv_count(N), source(0), tag(tag1));
                comm_world().send(send_buf(std::span<double>(d_A, N)), send_count(N), destination(0), tag(tag2));
            }
        }

        // Time ping-pong for loop_count iterations of data transfer size 8*N bytes
        double start_time, stop_time, elapsed_time;
        start_time = MPI_Wtime();
        RequestPool pool;
        for (int i = 1; i <= loop_count; i++) {
            if (world_rank() == 0) {
                comm_world().isend(send_buf(std::span<double>(d_A, N)), send_count(N), destination(1), tag(tag1), request(pool.get_request()));
                comm_world().irecv(recv_buf(std::span<double>(d_A, N)), recv_count(N), source(1), tag(tag2), request(pool.get_request()));
            }
            else if (world_rank() == 1) {
                comm_world().isend(send_buf(std::span<double>(d_A, N)), send_count(N), destination(0), tag(tag2), request(pool.get_request()));
                comm_world().irecv(recv_buf(std::span<double>(d_A, N)), recv_count(N), source(0), tag(tag1), request(pool.get_request()));
            }
        }
        pool.wait_all();
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
