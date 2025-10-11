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
    uint inverseSamplePosition[X];
    uint nextSample[X][X][2];
};


struct DC3 {
    static constexpr uint32_t X = 3;
    static constexpr uint32_t C = 2;
    static constexpr uint32_t nonSampleCount = X - C;
    static constexpr uint32_t samplePosition[C] = { 1, 2 };
    static constexpr uint32_t inverseSamplePosition[X - C] = { 0 };
    static constexpr uint32_t nextNonSample[X - C] = { 0 };
    static constexpr uint32_t nextSample[X][X][3] = { {{1, 0, 0}, {1, 0, 1}, {2, 1, 1}},

                                                            {{1, 1, 0}, {0, 0, 0}, {0, 0, 0}},

                                                            {{2, 1, 1}, {0, 0, 0}, {0, 0, 0}} };
};

struct DC7 {
    static constexpr uint32_t X = 7;
    static constexpr uint32_t C = 3;
    static constexpr uint32_t nonSampleCount = X - C;
    static constexpr uint32_t samplePosition[C] = { 1, 2, 4 };
    static constexpr uint32_t inverseSamplePosition[X - C] = { 0, 2, 3, 3 };
    static constexpr uint32_t nextNonSample[X - C] = { 0, 3, 5, 6 };

    static constexpr uint32_t nextSample[X][X][3] = {
        {{1, 0, 0}, {1, 0, 1}, {2, 1, 1}, {1, 0, 0}, {4, 2, 1}, {4, 2, 1}, {2, 1, 0}},

        {{1, 1, 0}, {0, 0, 0}, {0, 0, 0}, {1, 1, 0}, {0, 0, 0}, {3, 2, 0}, {3, 2, 1}},

        {{2, 1, 1}, {0, 0, 0}, {0, 0, 0}, {6, 2, 2}, {0, 0, 0}, {6, 2, 2}, {2, 1, 0}},

        {{1, 0, 0}, {1, 0, 1}, {6, 2, 2}, {1, 0, 0}, {5, 1, 2}, {6, 2, 2}, {5, 1, 2}},

        {{4, 1, 2}, {0, 0, 0}, {0, 0, 0}, {5, 2, 1}, {0, 0, 0}, {4, 1, 1}, {5, 2, 2}},

        {{4, 1, 2}, {3, 0, 2}, {6, 2, 2}, {6, 2, 2}, {4, 1, 1}, {3, 0, 0}, {3, 0, 1}},

        {{2, 0, 1}, {3, 1, 2}, {2, 0, 1}, {5, 2, 1}, {5, 2, 2}, {3, 1, 0}, {2, 0, 0}} };

};

struct DC13 {
    static constexpr uint32_t X = 13;
    static constexpr uint32_t C = 4;
    static constexpr uint32_t nonSampleCount = X - C;
    static constexpr uint32_t samplePosition[C] = { 1, 2, 4, 10 };
    static constexpr uint32_t inverseSamplePosition[X - C] = { 0, 2, 3, 3, 3, 3, 3, 4, 4 };
    static constexpr uint32_t nextNonSample[X - C] = { 0, 3, 5, 6, 7, 8, 9, 11, 12 };

    static constexpr uint32_t nextSample[X][X][3] = { {{1, 0, 0},
                                                             {1, 0, 1},
                                                             {2, 1, 1},
                                                             {1, 0, 0},
                                                             {10, 3, 2},
                                                             {10, 3, 2},
                                                             {4, 2, 0},
                                                             {10, 3, 3},
                                                             {2, 1, 0},
                                                             {1, 0, 0},
                                                             {4, 2, 1},
                                                             {4, 2, 1},
                                                             {2, 1, 0}},

                                                            {{1, 1, 0},
                                                             {0, 0, 0},
                                                             {0, 0, 0},
                                                             {1, 1, 0},
                                                             {0, 0, 0},
                                                             {9, 3, 1},
                                                             {9, 3, 2},
                                                             {3, 2, 0},
                                                             {9, 3, 3},
                                                             {1, 1, 0},
                                                             {0, 0, 0},
                                                             {3, 2, 0},
                                                             {3, 2, 1}},

                                                            {{2, 1, 1},
                                                             {0, 0, 0},
                                                             {0, 0, 0},
                                                             {12, 3, 3},
                                                             {0, 0, 0},
                                                             {12, 3, 3},
                                                             {8, 2, 1},
                                                             {8, 2, 2},
                                                             {2, 1, 0},
                                                             {8, 2, 3},
                                                             {0, 0, 0},
                                                             {12, 3, 3},
                                                             {2, 1, 0}},

                                                            {{1, 0, 0},
                                                             {1, 0, 1},
                                                             {12, 3, 3},
                                                             {1, 0, 0},
                                                             {11, 2, 3},
                                                             {12, 3, 3},
                                                             {11, 2, 3},
                                                             {7, 1, 1},
                                                             {7, 1, 2},
                                                             {1, 0, 0},
                                                             {7, 1, 3},
                                                             {12, 3, 3},
                                                             {11, 2, 3}},

                                                            {{10, 2, 3},
                                                             {0, 0, 0},
                                                             {0, 0, 0},
                                                             {11, 3, 2},
                                                             {0, 0, 0},
                                                             {10, 2, 2},
                                                             {11, 3, 3},
                                                             {10, 2, 3},
                                                             {6, 1, 1},
                                                             {6, 1, 2},
                                                             {0, 0, 0},
                                                             {6, 1, 2},
                                                             {11, 3, 3}},

                                                            {{10, 2, 3},
                                                             {9, 1, 3},
                                                             {12, 3, 3},
                                                             {12, 3, 3},
                                                             {10, 2, 2},
                                                             {5, 0, 0},
                                                             {9, 1, 2},
                                                             {10, 2, 3},
                                                             {9, 1, 3},
                                                             {5, 0, 1},
                                                             {5, 0, 2},
                                                             {12, 3, 3},
                                                             {5, 0, 2}},

                                                            {{4, 0, 2},
                                                             {9, 2, 3},
                                                             {8, 1, 2},
                                                             {11, 3, 2},
                                                             {11, 3, 3},
                                                             {9, 2, 1},
                                                             {4, 0, 0},
                                                             {8, 1, 2},
                                                             {9, 2, 3},
                                                             {8, 1, 3},
                                                             {4, 0, 1},
                                                             {4, 0, 1},
                                                             {11, 3, 3}},

                                                            {{10, 3, 3},
                                                             {3, 0, 2},
                                                             {8, 2, 2},
                                                             {7, 1, 1},
                                                             {10, 3, 2},
                                                             {10, 3, 2},
                                                             {8, 2, 1},
                                                             {3, 0, 0},
                                                             {7, 1, 2},
                                                             {8, 2, 3},
                                                             {7, 1, 3},
                                                             {3, 0, 0},
                                                             {3, 0, 1}},

                                                            {{2, 0, 1},
                                                             {9, 3, 3},
                                                             {2, 0, 1},
                                                             {7, 2, 1},
                                                             {6, 1, 1},
                                                             {9, 3, 1},
                                                             {9, 3, 2},
                                                             {7, 2, 1},
                                                             {2, 0, 0},
                                                             {6, 1, 2},
                                                             {7, 2, 3},
                                                             {6, 1, 2},
                                                             {2, 0, 0}},

                                                            {{1, 0, 0},
                                                             {1, 0, 1},
                                                             {8, 3, 2},
                                                             {1, 0, 0},
                                                             {6, 2, 1},
                                                             {5, 1, 0},
                                                             {8, 3, 1},
                                                             {8, 3, 2},
                                                             {6, 2, 1},
                                                             {1, 0, 0},
                                                             {5, 1, 2},
                                                             {6, 2, 2},
                                                             {5, 1, 2}},

                                                            {{4, 1, 2},
                                                             {0, 0, 0},
                                                             {0, 0, 0},
                                                             {7, 3, 1},
                                                             {0, 0, 0},
                                                             {5, 2, 0},
                                                             {4, 1, 0},
                                                             {7, 3, 1},
                                                             {7, 3, 2},
                                                             {5, 2, 1},
                                                             {0, 0, 0},
                                                             {4, 1, 1},
                                                             {5, 2, 2}},

                                                            {{4, 1, 2},
                                                             {3, 0, 2},
                                                             {12, 3, 3},
                                                             {12, 3, 3},
                                                             {6, 2, 1},
                                                             {12, 3, 3},
                                                             {4, 1, 0},
                                                             {3, 0, 0},
                                                             {6, 2, 1},
                                                             {6, 2, 2},
                                                             {4, 1, 1},
                                                             {3, 0, 0},
                                                             {3, 0, 1}},

                                                            {{2, 0, 1},
                                                             {3, 1, 2},
                                                             {2, 0, 1},
                                                             {11, 3, 2},
                                                             {11, 3, 3},
                                                             {5, 2, 0},
                                                             {11, 3, 3},
                                                             {3, 1, 0},
                                                             {2, 0, 0},
                                                             {5, 2, 1},
                                                             {5, 2, 2},
                                                             {3, 1, 0},
                                                             {2, 0, 0}} };
};




using MergeStageSuffix = MergeStageSuffixS0;
using DCX = DC13;

struct kmerDCX {
    unsigned char kmer[DCX::X];
};
struct dc3_kmer_decomposer
{
    __host__ __device__ cuda::std::tuple<unsigned char&, unsigned char&, unsigned char&> operator()(kmerDCX& key) const
    {
        return { key.kmer[0],key.kmer[1],key.kmer[2] };
    }
};
struct dc7_kmer_decomposer
{
    __host__ __device__ cuda::std::tuple<unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&> operator()(kmerDCX& key) const
    {
        return { key.kmer[0],key.kmer[1],key.kmer[2],key.kmer[3],key.kmer[4],key.kmer[5],key.kmer[6] };
    }
};
struct dc13_kmer_decomposer
{
    __host__ __device__ cuda::std::tuple<unsigned char&, unsigned char&, unsigned  char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&> operator()(kmerDCX& key) const
    {
        return { key.kmer[0],key.kmer[1],key.kmer[2],key.kmer[3],key.kmer[4],key.kmer[5],key.kmer[6],key.kmer[7],key.kmer[8],key.kmer[9],key.kmer[10],key.kmer[11],key.kmer[12] };
    }
};
using kmer = kmerDCX; // for dc3 is uint64_t better but also needs some readjustment in code



using DCXKmerDecomposer = dc13_kmer_decomposer;
using D_DCX = _D_DCX<DCX::X, DCX::C>;

struct MergeSuffixes {
    sa_index_t index;
    std::array<sa_index_t, DCX::C> ranks;
    std::array<unsigned char, DCX::X> prefix;
};

__host__ __forceinline__ bool operator<(const MergeSuffixes& a, const MergeSuffixes& b)
{
    uint32_t l = DCX::nextSample[a.index % DCX::X][b.index % DCX::X][0];
    uint32_t r1 = DCX::nextSample[a.index % DCX::X][b.index % DCX::X][1];
    uint32_t r2 = DCX::nextSample[a.index % DCX::X][b.index % DCX::X][2];
    for (size_t i = 0; i < l; i++)
    {
        if (a.prefix[i] < b.prefix[i]) {
            return true;
        }
        if (a.prefix[i] > b.prefix[i]) {
            return false;
        }
    }
    return a.ranks[r1] < b.ranks[r2];

}
__constant__ uint32_t lookupNext[DCX::X][DCX::X][3];

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

__host__ __device__ __forceinline__ bool operator==(const kmerDCX& a, const kmerDCX& b)
{
    for (size_t i = 0; i < DCX::X; i++)
    {
        if (a.kmer[i] < b.kmer[i]) {
            return false;
        }
        if (a.kmer[i] > b.kmer[i]) {
            return false;
        }
    }
    return true;
}


struct KmerComparator
{
    __device__ __forceinline__ bool operator()(const kmerDCX& a, const kmerDCX& b)
    {

        for (size_t i = 0; i < DCX::X; i++)
        {
            if (a.kmer[i] < b.kmer[i]) {
                return true;
            }
            if (a.kmer[i] > b.kmer[i]) {
                return false;
            }
        }
        return false;
    }
};

struct DC7Comparator
{
    __device__ __forceinline__ bool operator()(const MergeSuffixes& a, const MergeSuffixes& b)
    {
        uint32_t l = lookupNext[a.index % DCX::X][b.index % DCX::X][0];
        uint32_t r1 = lookupNext[a.index % DCX::X][b.index % DCX::X][1];
        uint32_t r2 = lookupNext[a.index % DCX::X][b.index % DCX::X][2];
        for (size_t i = 0; i < l; i++)
        {
            if (a.prefix[i] < b.prefix[i]) {
                return true;
            }
            if (a.prefix[i] > b.prefix[i]) {
                return false;
            }
        }

        return a.ranks[r1] < b.ranks[r2];
    }
};
struct DC7ComparatorHost
{
    __host__ __forceinline__ bool operator()(const MergeSuffixes& a, const MergeSuffixes& b)
    {
        uint32_t l = DCX::nextSample[a.index % DCX::X][b.index % DCX::X][0];
        uint32_t r1 = DCX::nextSample[a.index % DCX::X][b.index % DCX::X][1];
        uint32_t r2 = DCX::nextSample[a.index % DCX::X][b.index % DCX::X][2];
        for (size_t i = 0; i < l; i++)
        {
            if (a.prefix[i] < b.prefix[i]) {
                return true;
            }
            if (a.prefix[i] > b.prefix[i]) {
                return false;
            }
        }

        return a.ranks[r1] < b.ranks[r2];
    }
};
#endif // CONFIG_H
