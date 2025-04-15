#include <iostream>
#include <cuda_runtime.h>

__global__ void hello_from_gpu(int gpu_id)
{
    printf("Hello from GPU %d, thread %d\n", gpu_id, threadIdx.x);
}

int main()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0)
    {
        std::cerr << "Keine CUDA-fähigen Geräte gefunden." << std::endl;
        return 1;
    }

    std::cout << "Anzahl GPUs gefunden: " << device_count << std::endl;

    for (int i = 0; i < device_count; ++i)
    {
        cudaSetDevice(i);
        for (int j = 0; j < device_count; j++)
        {
            if (i != j)
            {
                int canAccessPeer = 0;
                cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
                if (canAccessPeer)
                {
                    cudaDeviceEnablePeerAccess(j, 0);
                    std::cout << "GPU " << i << " kann auf GPU " << j << " zugreifen (Peer Access aktiviert)." << std::endl;
                }
                else
                {
                    std::cout << "GPU " << i << " kann NICHT auf GPU " << j << " zugreifen." << std::endl;
                }
            }
        }
        std::cout << "Starte Kernel auf GPU " << i << std::endl;

        hello_from_gpu<<<1, 4>>>(dev); // 4 Threads zur Demonstration
        cudaDeviceSynchronize();
    }

    return 0;
}