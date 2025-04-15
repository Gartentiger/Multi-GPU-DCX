#include <iostream>
#include <cuda_runtime.h>

__global__ void hello_from_gpu(int gpu_id) {
    printf("Hello from GPU %d, thread %d\n", gpu_id, threadIdx.x);
}

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "Keine CUDA-fähigen Geräte gefunden." << std::endl;
        return 1;
    }

    std::cout << "Anzahl GPUs gefunden: " << device_count << std::endl;

    for (int dev = 0; dev < device_count; ++dev) {
        cudaSetDevice(dev);

        std::cout << "Starte Kernel auf GPU " << dev << std::endl;

        hello_from_gpu<<<1, 4>>>(dev);  // 4 Threads zur Demonstration
        cudaDeviceSynchronize();
    }

    return 0;
}