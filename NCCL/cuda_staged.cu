#include <stdio.h>
#include <stdlib.h>
#include <chrono>
// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
	cudaError_t cuErr = call;                                                             \
	if(cudaSuccess != cuErr){                                                             \
		printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
		exit(0);                                                                            \
	}                                                                                     \
}while(0)

class Timer {
private:
	std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;
	bool running = false;

public:
	void start() {
		start_time = std::chrono::high_resolution_clock::now();
		running = true;
	}

	void stop() {
		end_time = std::chrono::high_resolution_clock::now();
		running = false;
	}

	double elapsedMilliseconds() const {
		std::chrono::time_point<std::chrono::high_resolution_clock> end;
		if (running) {
			end = std::chrono::high_resolution_clock::now();
		}
		else {
			end = end_time;
		}
		return std::chrono::duration<double, std::milli>(end - start_time).count();
	}

	double elapsedSeconds() const {
		return elapsedMilliseconds() / 1000.0;
	}
};


int main(int argc, char* argv[])
{
	cudaErrorCheck(cudaSetDevice(0));
	cudaStream_t stream0;
	cudaErrorCheck(cudaStreamCreate(&stream0));

	cudaErrorCheck(cudaSetDevice(1));
	cudaStream_t stream1;
	cudaErrorCheck(cudaStreamCreate(&stream1));


	cudaErrorCheck(cudaSetDevice(0));
	int canAccess;
	cudaErrorCheck(cudaDeviceCanAccessPeer(&canAccess, 0, 1));
	if (canAccess) {
		cudaErrorCheck(cudaDeviceEnablePeerAccess(1, 0));
		printf("[0] peer to peer enabled\n");
	}

	cudaErrorCheck(cudaSetDevice(1));
	cudaErrorCheck(cudaDeviceCanAccessPeer(&canAccess, 1, 0));
	if (canAccess) {
		cudaErrorCheck(cudaDeviceEnablePeerAccess(0, 0));
		printf("[1] peer to peer enabled\n");
	}



	/* -------------------------------------------------------------------------------------------
		Loop from 8 B to 1 GB
	--------------------------------------------------------------------------------------------*/

	for (int i = 0; i <= 27; i++) {
		cudaErrorCheck(cudaSetDevice(1));
		cudaErrorCheck(cudaStreamSynchronize(stream1));
		cudaErrorCheck(cudaSetDevice(0));
		cudaErrorCheck(cudaStreamSynchronize(stream0));

		long int N = 1 << i;

		// Allocate memory for A on CPU
		double* A;
		cudaErrorCheck(cudaMallocHost(&A, N * sizeof(double)));

		// Allocate memory for A on CPU
		cudaErrorCheck(cudaSetDevice(1));
		double* B;
		cudaErrorCheck(cudaMallocHost(&B, N * sizeof(double)));

		// Initialize all elements of A to random values
		for (int i = 0; i < N; i++) {
			A[i] = (double)rand() / (double)RAND_MAX;
			B[i] = (double)rand() / (double)RAND_MAX;
		}

		double* d_A;
		double* d_B;
		cudaErrorCheck(cudaSetDevice(0));
		cudaErrorCheck(cudaMallocAsync(&d_A, N * sizeof(double), stream0));
		cudaErrorCheck(cudaMemcpyAsync(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice, stream0));


		cudaErrorCheck(cudaSetDevice(1));
		cudaErrorCheck(cudaMallocAsync(&d_B, N * sizeof(double), stream1));
		cudaErrorCheck(cudaMemcpyAsync(d_B, B, N * sizeof(double), cudaMemcpyHostToDevice, stream1));

		cudaErrorCheck(cudaStreamSynchronize(stream1));
		cudaErrorCheck(cudaSetDevice(0));
		cudaErrorCheck(cudaStreamSynchronize(stream0));
		int tag1 = 10;
		int tag2 = 20;

		int loop_count = 50;
		// Warm-up loop
		for (int i = 1; i <= 5; i++) {
			cudaErrorCheck(cudaSetDevice(0));
			cudaErrorCheck(cudaMemcpyPeer(d_B, 1, d_A, 0, N));
			cudaErrorCheck(cudaSetDevice(1));
			cudaErrorCheck(cudaMemcpyPeer(d_A, 0, d_B, 1, N));
		}

		// Time ping-pong for loop_count iterations of data transfer size 8*N bytes
		double start_time, stop_time, elapsed_time;
		Timer t;
		t.start();
		for (int i = 1; i <= loop_count; i++) {
			cudaErrorCheck(cudaSetDevice(0));
			cudaErrorCheck(cudaMemcpyPeer(d_B, 1, d_A, 0, N));
			cudaErrorCheck(cudaSetDevice(1));
			cudaErrorCheck(cudaMemcpyPeer(d_A, 0, d_B, 1, N));
		}
		t.stop();
		elapsed_time = t.elapsedMilliseconds();

		long int num_B = 8 * N;
		long int B_in_GB = 1 << 30;
		double num_GB = (double)num_B / (double)B_in_GB;
		double avg_time_per_transfer = elapsed_time / (2.0 * (double)loop_count);

		printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB / avg_time_per_transfer);

		cudaErrorCheck(cudaSetDevice(0));
		cudaErrorCheck(cudaFreeAsync(d_A, stream0));
		cudaErrorCheck(cudaFreeHost(A));
		cudaErrorCheck(cudaSetDevice(1));
		cudaErrorCheck(cudaFreeAsync(d_B, stream1));
		cudaErrorCheck(cudaFreeHost(B));
	}

	return 0;
}