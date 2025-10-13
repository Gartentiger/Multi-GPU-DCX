#include "io.cuh"
#include "cuda_helpers.h"
#include <cstdio>
#include <kamping/checking_casts.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/data_buffer.hpp> 
#include <kamping/environment.hpp>
#include <kamping/communicator.hpp>

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

    fseek(file, offset, SEEK_SET);

    size_t len_padded = SDIV(copy_len, padd_to) * padd_to;
    cudaMallocHost(contents, len_padded);
    CUERR;

    if (fread(*contents, 1, copy_len, file) != copy_len)
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
