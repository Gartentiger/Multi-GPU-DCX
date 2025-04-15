#ifndef QDALLOCATOR_HPP
#define QDALLOCATOR_HPP

#include "cuda_helpers.h"

class QDAllocator {
    public:
        QDAllocator(void* base_, size_t size_)
            : base(base_), size(size_), handed_out(0)
        {}

        QDAllocator()
            : base(nullptr), size(0), handed_out(0)
        {}

        template<typename T, size_t ALIGN=16>
        T* get(size_t num_elements) {
            size_t requested = num_elements * sizeof(T);
            handed_out = align<ALIGN>(handed_out);
            size_t offset = handed_out;
            handed_out += requested;
            ASSERT(handed_out <= size);
            return reinterpret_cast<T*>(reinterpret_cast<char*>(base) + offset);
        }

        template<size_t ALIGN=16>
        void* get_raw(size_t requested_size) {
            handed_out = align<ALIGN>(handed_out);
            size_t offset = handed_out;
            handed_out += requested_size;
            ASSERT(handed_out <= size);
            return reinterpret_cast<void*>(reinterpret_cast<char*>(base) + offset);
        }

        void reset() {
            handed_out = 0;
        }

        void init(void* base_, size_t size_) {
            base = base_;
            size = size_;
            handed_out = 0;
        }


    private:
        template <size_t ALIGN>
        static inline size_t align(size_t t) {
            return (t % ALIGN == 0) ? t : t + ALIGN - t % ALIGN;
        }

        void* base;
        size_t size, handed_out;
};

#endif // QDALLOCATOR_HPP
