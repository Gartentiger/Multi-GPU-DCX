#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int device_count;
    cudaGetDeviceCount(&device_count);

    if (device_count < 2)
    {
        std::cerr << "Mindestens zwei GPUs sind erforderlich!" << std::endl;
        return 1;
    }

    // Prüfen, ob Peer Access zwischen GPU 0 und GPU 1 möglich ist
    int canAccessPeer01 = 0, canAccessPeer10 = 0;
    cudaDeviceCanAccessPeer(&canAccessPeer01, 0, 1);
    cudaDeviceCanAccessPeer(&canAccessPeer10, 1, 0);

    if (!canAccessPeer01 || !canAccessPeer10)
    {
        std::cerr << "Peer-to-peer Zugriff zwischen GPU 0 und 1 nicht möglich." << std::endl;
        return 1;
    }

    // Peer Access aktivieren
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);

    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);

    std::cout << "Peer Access zwischen GPU 0 und GPU 1 aktiviert.\n";

    // Speicher auf GPU 0 und GPU 1 allokieren
    int *d_gpu0 = nullptr, *d_gpu1 = nullptr;
    cudaSetDevice(0);
    cudaMalloc(&d_gpu0, sizeof(int));
    int h_value = 42;
    cudaMemcpy(d_gpu0, &h_value, sizeof(int), cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    cudaMalloc(&d_gpu1, sizeof(int));

    // Peer-to-peer Copy von GPU 0 nach GPU 1
    cudaMemcpyPeer(d_gpu1, 1, d_gpu0, 0, sizeof(int));

    // Wert zurück zur CPU kopieren zum Überprüfen
    int result = 0;
    cudaMemcpy(&result, d_gpu1, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Wert von GPU 0 → GPU 1 übertragen: " << result << std::endl;

    // Aufräumen
    cudaSetDevice(0);
    cudaFree(d_gpu0);

    cudaSetDevice(1);
    cudaFree(d_gpu1);

    return 0;
}
