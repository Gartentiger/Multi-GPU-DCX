#ifndef UTIL_H
#define UTIL_H

void error(const char *format, ...);

void _handle_assert(const char *expression, const char *_file_, int _line_, const char* function);
void _handle_assert(const char *expression, const char *_file_, int _line_, const char* function, const char* msg);

#ifdef __clang_analyzer__
// make Clang-analyzer understand our ASSERT
#include <cassert>
#define ASSERT(e) assert(e)
#define ASSERT_MSG(e,msg) assert(e)
#else
#define ASSERT(e) ((e) ? (void)0 : _handle_assert(#e, __FILE__, __LINE__, __PRETTY_FUNCTION__))
#define ASSERT_MSG(e,msg) ((e) ? (void)0 : _handle_assert(#e, __FILE__, __LINE__, __PRETTY_FUNCTION__, msg))
#endif

#if defined(__CUDACC__)
#define HOST_DEVICE __forceinline__ __device__ __host__
#define _DEVICE_ __device__ __forceinline__
#else
// Better syntax highlighting for CUDA-unaware editors.
#define HOST_DEVICE
#define _DEVICE_
#define __device__
#define __host__
#define __global__
#define __shared__
#endif

#endif // UTIL_H
