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
#include <kamping/p2p/irecv.hpp>
#include <kamping/p2p/isend.hpp>
#include <kamping/request_pool.hpp>

static const size_t SEND_SIZE = 1000;
static const size_t SEND_TIMES = 1000;


int main(int argc, char** argv)
{
    using namespace kamping;
    kamping::Environment e;
    Communicator comm;

    int* sendBuf, * recvBuf;
    cudaSetDevice(0);
    cudaMalloc(&sendBuf, sizeof(int) * SEND_SIZE);
    cudaMalloc(&recvBuf, sizeof(int) * SEND_SIZE);

    std::vector<int> inp;
    inp.reserve(SEND_SIZE);
    std::srand(std::time(0));
    for (int i = 0; i < SEND_SIZE; i++) {
        inp[i] = std::rand();
    }

    cudaMemcpy(sendBuf, inp.data(), sizeof(int) * SEND_SIZE, cudaMemcpyHostToDevice);
    RequestPool req;
    std::span<int> sb(sendBuf, SEND_SIZE);
    std::span<int> rb(recvBuf, SEND_SIZE);
    auto& t = kamping::measurements::timer();
    t.synchronize_and_start("pingping");
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
    std::cout << "Ping pong complete" << std::endl;
    std::ofstream outFile(argv[1], std::ios::app);
    t.aggregate_and_print(
        kamping::measurements::SimpleJsonPrinter{ outFile, {} });
    std::cout << std::endl;
    t.aggregate_and_print(kamping::measurements::FlatPrinter{});
    std::cout << std::endl;
    return 0;
}