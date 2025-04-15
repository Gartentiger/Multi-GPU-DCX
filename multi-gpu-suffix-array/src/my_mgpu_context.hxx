#ifndef MY_MGPU_CONTEXT_HXX
#define MY_MGPU_CONTEXT_HXX

#include "moderngpu/context.hxx"
#include <cstdio>
#include "cuda_helpers.h"
#include "qdallocator.hpp"

namespace mgpu {

class my_mpgu_context_t : public mgpu::standard_context_t {
    public:
        my_mpgu_context_t(cudaStream_t stream_, QDAllocator& qd_allocator)
            : standard_context_t(false, stream_), mqd_allocator(qd_allocator)
        {}

        void reset_temp_memory() {
            mqd_allocator.reset();
        }

        void set_device_temp_mem(void* Device_temp_mem, size_t size_device_temp_mem) {
            mqd_allocator.init(Device_temp_mem, size_device_temp_mem);
        }

        // Alloc GPU memory.
        virtual void* alloc(size_t size, memory_space_t space) {
            if (memory_space_device != space){
                throw(std::runtime_error("mgpu: Cannot alloc host memory!"));
            }
            return mqd_allocator.get_raw<16>(size);
        }

        virtual void free(void* p, memory_space_t space) {
            (void) p; (void) space;
        }

    private:
        QDAllocator& mqd_allocator;
};

class my_mpgu_context_allocator_t : public my_mpgu_context_t {
    public:
        my_mpgu_context_allocator_t(cudaStream_t stream)
            : my_mpgu_context_t(stream, mqd_allocator)  {}
    private:
        QDAllocator mqd_allocator;
};

}
#endif // MY_MGPU_CONTEXT_HXX
