#include "io.cuh"
#include "cuda_helpers.h"
#include <cstdio>

void write_array(const char* ofile, const sa_index_t* sa, size_t len)
{
    FILE* fp = fopen(ofile, "wb");
    if (!fp)
    {
        error("Couldn't open file for writing!");
    }
    // for (size_t i = 0; i < len; i++)
    // {
    //     printf("AA: %d\n", sa[i]);
    // }
    if (fwrite(sa, sizeof(sa_index_t), len, fp) != len)
    {
        fclose(fp);
        error("Error writing file!");
    }

    fclose(fp);
}

size_t read_file_into_host_memory(char** contents, const char* path, size_t& real_len,
    size_t padd_to, size_t maxLength, char padd_c)
{
    FILE* file = fopen(path, "rb");
    if (!file)
    {
        error("Couldn't open file.");
    }
    fseek(file, 0, SEEK_END);

    size_t len = ftell(file);

    if (len > maxLength)
        len = maxLength;

    if (len == 0)
    {
        error("File is empty!");
    }

    fseek(file, 0, SEEK_SET);

    size_t len_padded = SDIV(len, padd_to) * padd_to;

    cudaMallocHost(contents, len_padded);
    CUERR

        if (fread(*contents, 1, len, file) != len)
            error("Error reading file!");

    fclose(file);

    // For logging.
    fprintf(stdout, "Read %zu bytes from %s.\n", len, path);
    fprintf(stderr, "Read %zu bytes from %s.\n", len, path);

    real_len = len;

    for (size_t i = len; i < len_padded; ++i)
    {
        (*contents)[i] = padd_c;
    }

    return len_padded;
}
