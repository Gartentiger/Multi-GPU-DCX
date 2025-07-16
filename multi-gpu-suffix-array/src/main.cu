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
#include <nccl.h>

int main(int argc, char** argv)
{
    using namespace kamping;
    kamping::Environment e;
    Communicator comm;
    ncclComm_t nccl_comm;
    ncclUniqueId Id;
    int device_count;
    cudaGetDeviceCount(&device_count);
    size_t dId = world_rank() % (size_t)device_count;
    printf("[%lu] dId: %lu\n", world_rank(), dId);
    cudaSetDevice(dId);
    printf("[%lu] Activating NCCL\n", world_rank());
    if (world_rank() == 0) {
        ncclGetUniqueId(&Id);
        printf("[%lu] Sending\n", world_rank());
        comm_world().bcast_single(send_recv_buf(Id));
    }
    else {
        printf("[%lu] Receiving\n", world_rank());
        Id = comm_world().bcast_single<ncclUniqueId>();
        printf("[%lu] Received\n", world_rank());
    }

    ncclCommInitRank(&nccl_comm, world_size(), Id, dId);
    printf("[%lu] Active nccl comm\n", world_rank());
    return 0;
}
