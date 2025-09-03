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


struct Sk5 {
    size_t index;
    sa_index_t ranks0;
    sa_index_t ranks1;
    sa_index_t ranks2;
    sa_index_t ranks3;
    sa_index_t ranks4;

    unsigned char xPrefix0;
    unsigned char xPrefix1;
    unsigned char xPrefix2;
    unsigned char xPrefix3;
    unsigned char xPrefix4;
};

struct decomposer_t
{
    __host__ __device__ cuda::std::tuple<unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, sa_index_t&, sa_index_t&, sa_index_t&, sa_index_t&, sa_index_t&, size_t&> operator()(Sk5& key) const
    {
        return { key.xPrefix0, key.xPrefix1, key.xPrefix2, key.xPrefix3, key.xPrefix4, key.ranks0, key.ranks1,key.ranks2,key.ranks3,key.ranks4, key.index };
    }
};
using MergeStageSuffix = MergeStageSuffixS0;

#endif // CONFIG_H
