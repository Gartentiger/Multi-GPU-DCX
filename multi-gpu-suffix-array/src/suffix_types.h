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

// device dcx struct
template<uint X, uint C>
struct _D_DCX {
    uint samplePosition[C];
    uint nextNonSample[X - C];
    uint nextSample[X][X];
};


struct DC3 {
    static constexpr uint32_t X = 3;
    static constexpr uint32_t C = 2;
    static constexpr uint32_t samplePosition[C] = { 1, 2 };
    static constexpr uint32_t nextNonSample[X - C] = { 0 };
    static constexpr uint32_t nextSample[X][X] = {
        {1,1,2},
        {1,0,0},
        {2,0,0} };
};

struct DC7 {
    static constexpr uint32_t X = 7;
    static constexpr uint32_t C = 3;
    static constexpr uint32_t samplePosition[C] = { 1, 2, 4 };
    static constexpr uint32_t nextNonSample[X - C] = { 0, 3, 5, 6 };
    static constexpr uint32_t nextSample[X][X] = {
        {1, 1, 2, 1, 4, 4, 2},

        {1, 0, 0, 1, 0, 3, 3},

        {2, 0, 0, 6, 0, 6, 2},

        {1, 1, 6, 1, 5, 6, 5},

        {4, 0, 0, 5, 0, 4, 5},

        {4, 3, 6, 6, 4, 3, 3},

        {2, 3, 2, 5, 5, 3, 2} };

};

using MergeStageSuffix = MergeStageSuffixS0;
using DCX = DC7;
using D_DCX = _D_DCX<DCX::X, DCX::C>;

struct Sk {
    unsigned char prefix[DCX::X];
    sa_index_t ranks[DCX::X];
    size_t index;
};

struct decomposer_t
{
    __host__ __device__ cuda::std::tuple<unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&> operator()(Sk& key) const
    {
        return { key.prefix[0], key.prefix[1], key.prefix[2], key.prefix[3], key.prefix[4], key.prefix[5], key.prefix[6] };
    }
};

#endif // CONFIG_H
