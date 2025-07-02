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

    cudaSetDevice(world_rank());
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

    if (world_rank() == 0) {
        int canAccess;
        cudaDeviceCanAccessPeer(&canAccess, world_rank(), 1);
        if (canAccess) {
            CUDACHECK(cudaDeviceEnablePeerAccess(1, 0));
            printf("[%lu] peer to peer enabled", world_rank());
        }

    }
    else if (world_rank() == 1) {
        int canAccess;
        cudaDeviceCanAccessPeer(&canAccess, world_rank(), 0);
        if (canAccess) {
            CUDACHECK(cudaDeviceEnablePeerAccess(0, 0));
            printf("[%lu] peer to peer enabled", world_rank());
        }
    }


    int* sendBuf, * recvBuf;
    cudaMalloc(&sendBuf, sizeof(int) * SEND_SIZE);
    cudaMalloc(&recvBuf, sizeof(int) * SEND_SIZE);

    std::vector<int> inp;
    inp.reserve(SEND_SIZE);
    std::srand(std::time(0));
    for (int i = 0; i < SEND_SIZE; i++) {
        inp[i] = std::rand();
    }

    cudaMemcpy(sendBuf, inp.data(), sizeof(int) * SEND_SIZE, cudaMemcpyHostToDevice);
    std::span<int> sb(sendBuf, SEND_SIZE);
    std::span<int> rb(recvBuf, SEND_SIZE);

    auto& t = kamping::measurements::timer();
    t.synchronize_and_start("pingping");
#ifdef USE_NCCL
    for (int i = 0; i < SEND_TIMES; i++) {
        t.synchronize_and_start("ping" + std::to_string(i));
        ncclGroupStart();
        if (world_rank() == 0) {
            ncclSend(sendBuf, SEND_SIZE, ncclInt, 1, nccl_comm, stream);
            ncclRecv(recvBuf, SEND_SIZE, ncclInt, 0, nccl_comm, stream);
        }
        else if (world_rank() == 1) {
            ncclSend(sendBuf, SEND_SIZE, ncclInt, 0, nccl_comm, stream);
            ncclRecv(recvBuf, SEND_SIZE, ncclInt, 1, nccl_comm, stream);
        }
        ncclGroupEnd();
        t.stop_and_append();
    }
#else 

    RequestPool req;
    for (int i = 0; i < SEND_TIMES; i++) {
        t.synchronize_and_start("ping" + std::to_string(i));
        if (world_rank() == 0) {
            comm_world().isend(send_buf(sb), send_count(SEND_SIZE), destination(1), request(req.get_request()));
            comm_world().irecv(recv_buf(rb), recv_count(SEND_SIZE), request(req.get_request()));
        }
        else if (world_rank() == 1) {
            comm_world().isend(send_buf(sb), send_count(SEND_SIZE), destination(0), request(req.get_request()));
            comm_world().irecv(recv_buf(rb), recv_count(SEND_SIZE), request(req.get_request()));
        }
        req.wait_all();
        t.stop_and_append();
    }
#endif
    t.stop();


    std::cout << "Ping pong complete" << std::endl;
    std::ofstream outFile(argv[1], std::ios::app);
    t.aggregate_and_print(
        kamping::measurements::SimpleJsonPrinter{ outFile, {} });
    std::cout << std::endl;
    t.aggregate_and_print(kamping::measurements::FlatPrinter{});
    std::cout << std::endl;
    return 0;
}
