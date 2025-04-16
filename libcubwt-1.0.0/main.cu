#include <stdio.h>
#include "libcubwt.cuh"

// CUDA Kernel-Funktion
__global__ void helloFromGPU()
{
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main()
{
    printf("Hello World from CPU!\n");
    void *deviceStorage;
    int64_t a = libcubwt_allocate_device_storage(&deviceStorage, 20);
    if (a == LIBCUBWT_NO_ERROR)
    {
        const char *text = "yabbadabbadodododo";

        const uint8_t *bytes = (const uint8_t *)text;

        uint32_t isa[18];
        int64_t err = libcubwt_isa(deviceStorage, bytes, isa, 18);
        if (err == LIBCUBWT_NO_ERROR)
        {
            for (int i = 0; i < 18; i++)
            {
                printf("Suffix: %c ISA: %u\n", text[i], isa[i]);
            }
        }
        else
        {
            printf("Error:%ld\n", err);
        }
    }
    // Starte den Kernel mit 1 Block und 5 Threads
    helloFromGPU<<<1, 5>>>();

    // Warten, bis alle GPU-Aufgaben abgeschlossen sind
    cudaDeviceSynchronize();

    return 0;
}