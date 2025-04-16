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
        const char *text = "yabbadabbado";

        const uint8_t *bytes = (const uint8_t *)text;
        uint32_t isa[12];
        int64_t err = libcubwt_isa(deviceStorage, bytes, isa, 12);
        if (err == LIBCUBWT_NO_ERROR)
        {
            printf("ISA: %u\n", isa[0]);
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