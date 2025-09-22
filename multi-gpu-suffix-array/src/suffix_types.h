#ifndef CONFIG_H
#define CONFIG_H

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
    uint inverseSamplePosition[X];
    uint nextSample[X][X][2];
};


struct DC3 {
    static constexpr uint32_t X = 3;
    static constexpr uint32_t C = 2;
    static constexpr uint32_t nonSampleCount = X - C;
    static constexpr uint32_t samplePosition[C] = { 1, 2 };
    static constexpr uint32_t inverseSamplePosition[X] = { 0, 0, 1 };
    static constexpr uint32_t nextNonSample[X - C] = { 0 };
    static constexpr uint32_t nextSample[X][X][2] = {
        {{1,0},{1,0},{2,1}},
        {{1,1},{0,0},{0,0}},
        {{2,1},{0,0},{0,0}} };
};

struct DC7 {
    static constexpr uint32_t X = 7;
    static constexpr uint32_t C = 3;
    static constexpr uint32_t nonSampleCount = X - C;
    static constexpr uint32_t samplePosition[C] = { 1, 2, 4 };
    static constexpr uint32_t inverseSamplePosition[X] = { 0, 0, 1, 0, 2, 0, 0 };
    static constexpr uint32_t nextNonSample[X - C] = { 0, 3, 5, 6 };

    static constexpr uint32_t nextSample[X][X][2] = {
        {{1,0}, {1,0}, {2,1}, {1,0}, {4,2}, {4,2}, {2,1}},

        {{1,1}, {0,0}, {0,0}, {1,1}, {0,0}, {3,2}, {3,2}},

        {{2,1}, {0,0}, {0,0}, {6,2}, {0,0}, {6,2}, {2,1}},

        {{1,0}, {1,0}, {6,2}, {1,0}, {5,1}, {6,2}, {5,1}},

        {{4,1}, {0,0}, {0,0}, {5,2}, {0,0}, {4,1}, {5,2}},

        {{4,1}, {3,0}, {6,2}, {6,2}, {4,1}, {3,0}, {3,0}},

        {{2,0}, {3,1}, {2,0}, {5,2}, {5,2}, {3,1}, {2,0}} };

};

using MergeStageSuffix = MergeStageSuffixS0;
using DCX = DC7;
using D_DCX = _D_DCX<DCX::X, DCX::C>;
struct MergeSuffixes {
    sa_index_t index;
    sa_index_t ranks[DCX::C];
    unsigned char prefix[DCX::X];
};


__constant__ uint32_t lookupNext[DCX::X][DCX::X][2];

struct NonSampleKey {
    sa_index_t rankL;
    unsigned char prefix[DCX::X];
};


struct NonSampleValue {
    sa_index_t index;
    sa_index_t ranks[DCX::X];
};

struct non_sample_prefix_decomp
{
    __host__ __device__ cuda::std::tuple<unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, sa_index_t&> operator()(MergeSuffixes& key) const
    {
        return { key.prefix[0], key.prefix[1], key.prefix[2], key.prefix[3], key.prefix[4], key.prefix[5], key.prefix[6], key.ranks[0] };
    }
};

struct rank_decomposer
{
    __host__ __device__ cuda::std::tuple<sa_index_t&> operator()(MergeSuffixes& key) const
    {

        return { key.ranks[0] };
    }
};

struct DC7Comparator
{
    __host__ __device__ __forceinline__ bool operator()(const MergeSuffixes& a, const MergeSuffixes& b)
    {
        for (size_t i = 0; i < lookupNext[a.index % DCX::X][b.index % DCX::X][0]; i++)
        {
            if (a.prefix[i] < b.prefix[i]) {
                return true;
            }
            else if (a.prefix[i] > b.prefix[i]) {
                return false;
            }
        }

        return a.ranks[lookupNext[a.index % DCX::X][b.index % DCX::X][1]] < b.ranks[lookupNext[b.index % DCX::X][a.index % DCX::X][1]];
    }
};

#endif // CONFIG_H
