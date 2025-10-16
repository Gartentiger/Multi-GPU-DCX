#include "io.cuh"
#include "cuda_helpers.h"
#include <cstdio>
#include <kamping/checking_casts.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/allgather.hpp>
#include <kamping/named_parameters.hpp>
#include <kamping/data_buffer.hpp> 
#include <kamping/environment.hpp>
#include <kamping/communicator.hpp>
#include <kamping/data_buffer.hpp>
#include <algorithm>
#include <vector>
#include <mpi.h>
#include <span>
#include <numeric>


void write_array(const char* ofile, const sa_index_t* sa, size_t len)
{
    FILE* fp = fopen(ofile, "wb");
    if (!fp)
    {
        error("Couldn't open file for writing!");
    }

    if (fwrite(sa, sizeof(sa_index_t), len, fp) != len)
    {
        fclose(fp);
        error("Error writing file!");
    }

    fclose(fp);
}

void write_array_mpi(const char* ofile, sa_index_t* sa, size_t len)
{
    printf("[%lu] write sa\n", kamping::world_rank());
    std::vector<size_t> recv_sizes(kamping::world_size());
    kamping::comm_world().allgather(kamping::send_buf(std::span<size_t>(&len, 1)), kamping::recv_buf(recv_sizes), kamping::send_count(1));
    size_t acc = std::accumulate(recv_sizes.begin(), recv_sizes.begin() + kamping::world_rank(), 0);
    int ierr;
    MPI_File outputFile;
    ierr = MPI_File_open(MPI_COMM_WORLD, ofile,
        MPI_MODE_CREATE | MPI_MODE_WRONLY,
        MPI_INFO_NULL, &outputFile);
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr, "[%lu] Error opening file\n", kamping::world_rank());
        MPI_Abort(MPI_COMM_WORLD, ierr);
    }
    MPI_Offset offset = acc * sizeof(sa_index_t);
    ierr = MPI_File_write_at_all(outputFile, offset, sa, len, MPI_UINT32_T, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr, "[%lu] Error in MPI_File_write_at_all\n", kamping::world_rank());
        MPI_Abort(MPI_COMM_WORLD, ierr);
    }
    MPI_File_close(&outputFile);
}
size_t read_file_into_host_memory(char** contents, const char* path, size_t& real_len,
    size_t padd_to, size_t maxLength, uint num_gpus, char padd_c)
{
    FILE* file = fopen(path, "rb");
    if (!file)
    {
        error("Couldn't open file.");
    }
    fseek(file, 0, SEEK_END);

    size_t len = ftell(file);
    printf("Filesize: %lu, sizeof(size_t): %lu, max: %lu\n", len, sizeof(size_t), maxLength);
    if (len > maxLength)
        len = maxLength;

    if (len == 0)
    {
        error("File is empty!");
    }

    size_t mper_gpu = SDIV(len, num_gpus);
    mper_gpu = SDIV(mper_gpu, DCX::X) * DCX::X;
    size_t offset = mper_gpu * kamping::world_rank();

    if (kamping::world_rank() == num_gpus - 1)
        mper_gpu = len - (num_gpus - 1) * mper_gpu;

    size_t copy_len = std::min(mper_gpu + sizeof(kmer), len - offset);

    fseek(file, 0, SEEK_SET);

    size_t len_padded = SDIV(len, padd_to) * padd_to;
    cudaMallocHost(contents, len_padded);
    CUERR;

    if (fread(*contents, 1, len, file) != len)
        error("Error reading file!");

    fclose(file);

    // For logging.
    fprintf(stdout, "Read %zu bytes from %s.\n", copy_len, path);
    fprintf(stderr, "Read %zu bytes from %s.\n", copy_len, path);

    real_len = len;

    for (size_t i = len; i < len_padded; ++i)
    {
        (*contents)[i] = padd_c;
    }

    return len_padded;
}
