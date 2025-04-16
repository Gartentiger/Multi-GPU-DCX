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
        const uint8_t *s = "yabbadabbado";
        uint32_t *isa;
        libcubwt_isa(deviceStorage, s, isa, 12);
        for (int i = 0; i < 12; i++)
        {
            printf("ISA: %u", *isa++);
        }
    }
    // Starte den Kernel mit 1 Block und 5 Threads
    helloFromGPU<<<1, 5>>>();

    // Warten, bis alle GPU-Aufgaben abgeschlossen sind
    cudaDeviceSynchronize();

    return 0;
}