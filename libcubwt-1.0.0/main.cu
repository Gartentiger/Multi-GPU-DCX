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
        char text[] = "yabbadabbado";
        uint64_t len = strlen(text);

        uint8_t bytes[12]; // oder malloc, wenn dynamisch

        // Kopieren
        for (uint64_t i = 0; i < len; i++)
        {
            bytes[i] = (uint8_t)text[i];
        }
        uint32_t isa;
        int64_t err = libcubwt_isa(deviceStorage, bytes, &isa, len);
        if (err == LIBCUBWT_NO_ERROR)
        {

            printf("ISA: %u\n", isa);
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