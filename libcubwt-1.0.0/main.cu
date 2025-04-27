#include <stdio.h>
#include "libcubwt.cuh"
#include "io.cuh"
#include <stdio.h>

int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("args");
        return -1;
    }
    char* text; //= "yabbadabbadodododfadsagldfkaölkjghksöadflhslködsfsdgadfgsahgshstfhhfjhlskghndlkfgnasökligneaölkgrnrngökren";
    size_t len;
    read_file_into_host_memory(&text, argv[1], len, sizeof(uint32_t), 0);
    //while (*(text + len++) != '\0') {}
    //printf("input_len %lu realeln %lu len %lu\n", inputLen, real_len, len);

    void* deviceStorage;
    int64_t a = libcubwt_allocate_device_storage(&deviceStorage, len);
    if (a == LIBCUBWT_NO_ERROR)
    {
        const uint8_t* bytes = (const uint8_t*)text;

        uint32_t isa[len];
        int64_t err = libcubwt_isa(deviceStorage, bytes, isa, len);
        if (err == LIBCUBWT_NO_ERROR)
        {
            //for (int i = 0; i < len; i++)
            //{
                //printf("Suffix: %c ISA: %u\n", text[i], isa[i]);
            //}
        }
        else
        {
            printf("Error:%ld\n", err);
        }
    }
    libcubwt_free_device_storage(deviceStorage);
    cudaDeviceSynchronize();
    printf("Success\n");
    return 0;
}