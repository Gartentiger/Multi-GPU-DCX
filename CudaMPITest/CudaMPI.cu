#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "cuda.h"
#include "cuda_runtime.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    // Get the size of the group associated with communicator MPI_COMM_WORLD
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the calling process in the communicator MPI_COMM_WORLD
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    printf("* WorldSize: %d, Rank: %d\n", world_size, world_rank);
    // std::vector<int> input2(2u * comm.size(), comm.rank_signed());
    // std::vector<int> output = comm.alltoall(send_buf(input2));
    // printf("Rank: %d, Size: %d, SRank: %d\n", comm.rank(), output[0], output[1]);
    printf("* Allocate memory [%d],GPU\n", world_rank);
    int *d_a;
    if (cudaMalloc((void **)&d_a, 100 * sizeof(int)) != cudaSuccess)
    {
        // errx(1, "cudaMalloc d_a[] failed");
        printf("Error malloc %d\n", world_rank);
        return 1;
    }
    cudaMemset(d_a, 0, 100 * sizeof(int));
    int err = 0;
    MPI_Status status;
    // From [1],GPU to [0],GPU
    if (world_rank == 1)
    {
        cudaMemset(d_a, 1, 100 * sizeof(int));
        printf("Memset %d \n", world_rank);
        err = MPI_Send(d_a, 100, MPI_INT, 0, 2, MPI_COMM_WORLD);
        printf("* Send from [%d],GPU Data %d\n", world_rank, d_a[0]);
    }
    else if (world_rank == 0)
    {
        err = MPI_Recv(d_a, 100, MPI_INT, 1, 2, MPI_COMM_WORLD, &status);
        printf("* Receive to [%d],GPU Data %d\n", world_rank, d_a[0]);
    }
    if (err != MPI_SUCCESS)
    {
        // errx(2, "MPI transport from [1],GPU to [0],GPU failed");
        printf("Error transport");
        return 1;
    }
    printf("* Free memory on [%d],GPU\n", world_rank);
    cudaFree(d_a);

    // Terminates MPI execution environment
    MPI_Finalize();
    return 0;
}