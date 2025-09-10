#ifndef CONFIG_H
#define CONFIG_H
#include <array>
#include <cstdint>
#include <cuda/std/tuple>

using uint = unsigned int;
using sa_index_t = uint32_t;
const sa_index_t SA_INDEX_T_MAX = UINT32_MAX;

struct MergeStageSuffixS12 {
    sa_index_t index, own_rank, rank_p1p2;
    unsigned char chars[2], _padding[2];
};

struct MergeStageSuffixS0 {
    sa_index_t index, rank_p1, rank_p2;
    unsigned char chars[2], _padding[2];
};

// Because of little endian, this will be sorted first according to chars[0], then rank_p1 when sorting
// the 40 least significant bits.
struct MergeStageSuffixS0HalfKey {
    sa_index_t rank_p1;
    unsigned char chars[2], _padding[2];
};

struct MergeStageSuffixS0HalfValue {
    sa_index_t index, rank_p2;
};

struct MergeStageSuffixS12HalfKey {
    sa_index_t own_rank, index;
};

struct MergeStageSuffixS12HalfValue {
    sa_index_t rank_p1p2;
    unsigned char chars[2], _padding[2];
};


struct Sk7 {
    unsigned char prefix[7], padd[5];
    sa_index_t ranks[7];
    size_t index;
};

struct DC7L {
    static constexpr uint32_t lookupL[7][7] = {
        {1, 1, 2, 1, 4, 4, 2},

        {1, 0, 0, 1, 0, 3, 3},

        {2, 0, 0, 6, 0, 6, 2},

        {1, 1, 6, 1, 5, 6, 5},

        {4, 0, 0, 5, 0, 4, 5},

        {4, 3, 6, 6, 4, 3, 3},

        {2, 3, 2, 5, 5, 3, 2} };
};

struct decomposer_t
{
    __host__ __device__ cuda::std::tuple<unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&> operator()(Sk7& key) const
    {
        return { key.prefix[0], key.prefix[1], key.prefix[2], key.prefix[3], key.prefix[4], key.prefix[5], key.prefix[6] };
    }
};
using MergeStageSuffix = MergeStageSuffixS0;

#endif // CONFIG_H
