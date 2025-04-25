#include <stdio.h>
#include "libcubwt.cuh"

int main()
{
    void* deviceStorage;
    const char* text = "yabbadabbadodododfadsagldfkaölkjghksöadflhslködsfsdgadfgsahgshstfhhfjhlskghndlkfgnasökligneaölkgrnrngökren";
    size_t len = 0;
    while (*(text + len++) != '\0') {}
    int64_t a = libcubwt_allocate_device_storage(&deviceStorage, len);
    if (a == LIBCUBWT_NO_ERROR)
    {
        const uint8_t* bytes = (const uint8_t*)text;

        uint32_t isa[len];
        int64_t err = libcubwt_isa(deviceStorage, bytes, isa, len);
        if (err == LIBCUBWT_NO_ERROR)
        {
            for (int i = 0; i < len; i++)
            {
                printf("Suffix: %c ISA: %u\n", text[i], isa[i]);
            }
        }
        else
        {
            printf("Error:%ld\n", err);
        }
    }
    libcubwt_free_device_storage(deviceStorage);
    cudaDeviceSynchronize();

    return 0;
}