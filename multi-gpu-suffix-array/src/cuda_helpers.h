#ifndef CUDA_HELPERS_H_
#define CUDA_HELPERS_H_

#include "util.h"
#include <mpi.h>
#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>

// safe division
#define SDIV(x,y)(((x)+(y)-1)/(y))

// error macro
#define CUERR {                                                              \
    cudaError_t err;                                                         \
    if ((err = cudaGetLastError()) != cudaSuccess) {                         \
         error("CUDA error %lu (%s, %d): %s",world_rank() , __FILE__, __LINE__, cudaGetErrorString(err));    \
         exit(1);                                                            \
    }                                                                        \
}

#define CUERR_CHECK(err) { \
    if ((err) != cudaSuccess) {                                              \
         error("CUDA error (%s, %d): %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
         exit(1);                                                            \
    }                                                                        \
}

// convenient timers
#define TIMERSTART(label)                                                    \
        cudaSetDevice(0);                                                    \
        cudaEvent_t start##label, stop##label;                               \
        float time##label;                                                   \
        cudaEventCreate(&start##label);                                      \
        cudaEventCreate(&stop##label);                                       \
        cudaEventRecord(start##label, 0);

#define TIMERSTOP(label)                                                     \
        cudaSetDevice(0);                                                    \
        cudaEventRecord(stop##label, 0);                                     \
        cudaEventSynchronize(stop##label);                                   \
        cudaEventElapsedTime(&time##label, start##label, stop##label);       \
        std::cout << "#" << time##label                                      \
                  << " ms (" << #label << ")" << std::endl;

#endif /* CUDA_HELPERS_H_ */
