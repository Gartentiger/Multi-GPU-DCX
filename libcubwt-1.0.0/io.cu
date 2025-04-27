#include "io.cuh"
#include <cuda_runtime.h>
#include <cstdio>

int read() {
    FILE* file = fopen("", "r");
    if (file == NULL) {
        perror("Cannot open file!");
        exit(1);
    }
    return 0;
}

size_t read_file_into_host_memory(char** contents, const char* path, size_t& real_len,
    size_t padd_to, char padd_c)
{
    FILE* file = fopen(path, "rb");
    if (!file)
    {
        perror("Couldn't open file.");
    }
    fseek(file, 0, SEEK_END);

    size_t len = ftell(file);

    if (len == 0)
    {
        perror("File is empty!");
    }

    fseek(file, 0, SEEK_SET);

    size_t len_padded = ((len + padd_to - 1) / padd_to) * padd_to;

    cudaMallocHost(contents, len_padded);
    //CUERR

    if (fread(*contents, 1, len, file) != len)
        perror("Error reading file!");

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
