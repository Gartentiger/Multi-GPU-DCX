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

struct DC21 {
    static constexpr uint32_t X = 21;
    static constexpr uint32_t C = 5;
    static constexpr uint32_t nonSampleCount = X - C;
    static constexpr uint32_t samplePosition[C] = { 1, 2, 7, 9, 19 };
    static constexpr uint32_t nextNonSample[X - C] = { 0, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20 };

    static constexpr uint32_t inverseSamplePosition[X - C] = { 0, 2, 2, 2, 2, 3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5 };
    static constexpr uint32_t nextSample[X][X][3] = {
        {{1, 0, 0}, {1, 0, 1}, {7, 2, 2},  {19, 4, 3}, {19, 4, 4}, {2, 1, 0}, {1, 0, 0},
         {2, 1, 1}, {1, 0, 0}, {19, 4, 4}, {9, 3, 0},  {19, 4, 4}, {7, 2, 0}, {9, 3, 1},
         {9, 3, 2}, {7, 2, 1}, {7, 2, 2},  {2, 1, 0},  {1, 0, 0},  {9, 3, 3}, {2, 1, 0}},

        {{1, 1, 0}, {0, 0, 0}, {0, 0, 0}, {6, 2, 1},  {18, 4, 3}, {18, 4, 4}, {1, 1, 0},
         {0, 0, 0}, {1, 1, 0}, {0, 0, 0}, {18, 4, 3}, {8, 3, 0},  {18, 4, 4}, {6, 2, 0},
         {8, 3, 1}, {8, 3, 2}, {6, 2, 1}, {6, 2, 2},  {1, 1, 0},  {0, 0, 0},  {8, 3, 2}},

        {{7, 2, 2}, {0, 0, 0},  {0, 0, 0}, {20, 4, 4}, {5, 1, 1},  {17, 3, 3}, {17, 3, 4},
         {0, 0, 0}, {20, 4, 4}, {0, 0, 0}, {20, 4, 4}, {17, 3, 3}, {7, 2, 0},  {17, 3, 4},
         {5, 1, 0}, {7, 2, 1},  {7, 2, 2}, {5, 1, 1},  {5, 1, 2},  {0, 0, 0},  {20, 4, 4}},

        {{19, 3, 4}, {6, 1, 2},  {20, 4, 4}, {4, 0, 0},  {19, 3, 4}, {4, 0, 1},  {16, 2, 3},
         {16, 2, 4}, {20, 4, 4}, {19, 3, 4}, {20, 4, 4}, {19, 3, 4}, {16, 2, 3}, {6, 1, 0},
         {16, 2, 4}, {4, 0, 0},  {6, 1, 1},  {6, 1, 2},  {4, 0, 1},  {4, 0, 2},  {20, 4, 4}},

        {{19, 4, 4}, {18, 3, 4}, {5, 1, 1},  {19, 4, 3}, {3, 0, 0},  {18, 3, 4}, {3, 0, 1},
         {15, 2, 3}, {15, 2, 3}, {19, 4, 4}, {18, 3, 3}, {19, 4, 4}, {18, 3, 4}, {15, 2, 3},
         {5, 1, 0},  {15, 2, 4}, {3, 0, 0},  {5, 1, 1},  {5, 1, 2},  {3, 0, 1},  {3, 0, 1}},

        {{2, 0, 1},  {18, 4, 4}, {17, 3, 3}, {4, 1, 0},  {18, 4, 3}, {2, 0, 0},  {17, 3, 4},
         {2, 0, 1},  {14, 2, 2}, {14, 2, 3}, {18, 4, 3}, {17, 3, 3}, {18, 4, 4}, {17, 3, 4},
         {14, 2, 3}, {4, 1, 0},  {14, 2, 4}, {2, 0, 0},  {4, 1, 1},  {4, 1, 2},  {2, 0, 0}},

        {{1, 0, 0},  {1, 0, 1},  {17, 4, 3}, {16, 3, 2}, {3, 1, 0},  {17, 4, 3}, {1, 0, 0},
         {16, 3, 4}, {1, 0, 0},  {13, 2, 2}, {13, 2, 2}, {17, 4, 3}, {16, 3, 3}, {17, 4, 4},
         {16, 3, 4}, {13, 2, 3}, {3, 1, 0},  {13, 2, 4}, {1, 0, 0},  {3, 1, 1},  {3, 1, 1}},

        {{2, 1, 1},  {0, 0, 0},  {0, 0, 0},  {16, 4, 2}, {15, 3, 2}, {2, 1, 0},  {16, 4, 3},
         {0, 0, 0},  {15, 3, 3}, {0, 0, 0},  {12, 2, 1}, {12, 2, 2}, {16, 4, 3}, {15, 3, 3},
         {16, 4, 4}, {15, 3, 4}, {12, 2, 3}, {2, 1, 0},  {12, 2, 4}, {0, 0, 0},  {2, 1, 0}},

        {{1, 0, 0},  {1, 0, 1},  {20, 4, 4}, {20, 4, 4}, {15, 3, 2}, {14, 2, 2}, {1, 0, 0},
         {15, 3, 3}, {1, 0, 0},  {14, 2, 3}, {20, 4, 4}, {11, 1, 1}, {11, 1, 2}, {15, 3, 3},
         {14, 2, 3}, {15, 3, 4}, {14, 2, 4}, {11, 1, 3}, {1, 0, 0},  {11, 1, 4}, {20, 4, 4}},

        {{19, 4, 4}, {0, 0, 0},  {0, 0, 0},  {19, 4, 3}, {19, 4, 4}, {14, 3, 2}, {13, 2, 2},
         {0, 0, 0},  {14, 3, 2}, {0, 0, 0},  {13, 2, 2}, {19, 4, 4}, {10, 1, 1}, {10, 1, 2},
         {14, 3, 3}, {13, 2, 3}, {14, 3, 4}, {13, 2, 4}, {10, 1, 3}, {0, 0, 0},  {10, 1, 3}},

        {{9, 0, 3},  {18, 3, 4}, {20, 4, 4}, {20, 4, 4}, {18, 3, 3}, {18, 3, 4}, {13, 2, 2},
         {12, 1, 2}, {20, 4, 4}, {13, 2, 2}, {9, 0, 0},  {12, 1, 2}, {18, 3, 4}, {9, 0, 1},
         {9, 0, 2},  {13, 2, 3}, {12, 1, 3}, {13, 2, 4}, {12, 1, 4}, {9, 0, 3},  {20, 4, 4}},

        {{19, 4, 4}, {8, 0, 3},  {17, 3, 3}, {19, 4, 3}, {19, 4, 4}, {17, 3, 3}, {17, 3, 4},
         {12, 2, 2}, {11, 1, 1}, {19, 4, 4}, {12, 2, 1}, {8, 0, 0},  {11, 1, 2}, {17, 3, 4},
         {8, 0, 1},  {8, 0, 2},  {12, 2, 3}, {11, 1, 3}, {12, 2, 4}, {11, 1, 4}, {8, 0, 2}},

        {{7, 0, 2},  {18, 4, 4}, {7, 0, 2},  {16, 3, 2}, {18, 4, 3}, {18, 4, 4}, {16, 3, 3},
         {16, 3, 4}, {11, 2, 1}, {10, 1, 1}, {18, 4, 3}, {11, 2, 1}, {7, 0, 0},  {10, 1, 2},
         {16, 3, 4}, {7, 0, 1},  {7, 0, 2},  {11, 2, 3}, {10, 1, 3}, {11, 2, 4}, {10, 1, 3}},

        {{9, 1, 3},  {6, 0, 2},  {17, 4, 3}, {6, 0, 1}, {15, 3, 2}, {17, 4, 3}, {17, 4, 4},
         {15, 3, 3}, {15, 3, 3}, {10, 2, 1}, {9, 1, 0}, {17, 4, 3}, {10, 2, 1}, {6, 0, 0},
         {9, 1, 2},  {15, 3, 4}, {6, 0, 1},  {6, 0, 2}, {10, 2, 3}, {9, 1, 3},  {10, 2, 3}},

        {{9, 2, 3},  {8, 1, 3},  {5, 0, 1},  {16, 4, 2}, {5, 0, 1}, {14, 3, 2}, {16, 4, 3},
         {16, 4, 4}, {14, 3, 2}, {14, 3, 3}, {9, 2, 0},  {8, 1, 0}, {16, 4, 3}, {9, 2, 1},
         {5, 0, 0},  {8, 1, 2},  {14, 3, 4}, {5, 0, 1},  {5, 0, 2}, {9, 2, 3},  {8, 1, 2}},

        {{7, 1, 2},  {8, 2, 3},  {7, 1, 2},  {4, 0, 0},  {15, 4, 2}, {4, 0, 1}, {13, 3, 2},
         {15, 4, 3}, {15, 4, 3}, {13, 3, 2}, {13, 3, 2}, {8, 2, 0},  {7, 1, 0}, {15, 4, 3},
         {8, 2, 1},  {4, 0, 0},  {7, 1, 2},  {13, 3, 4}, {4, 0, 1},  {4, 0, 2}, {8, 2, 2}},

        {{7, 2, 2},  {6, 1, 2},  {7, 2, 2},  {6, 1, 1},  {3, 0, 0},  {14, 4, 2}, {3, 0, 1},
         {12, 3, 2}, {14, 4, 2}, {14, 4, 3}, {12, 3, 1}, {12, 3, 2}, {7, 2, 0},  {6, 1, 0},
         {14, 4, 3}, {7, 2, 1},  {3, 0, 0},  {6, 1, 2},  {12, 3, 4}, {3, 0, 1},  {3, 0, 1}},

        {{2, 0, 1}, {6, 2, 2},  {5, 1, 1},  {6, 2, 1},  {5, 1, 1},  {2, 0, 0},  {13, 4, 2},
         {2, 0, 1}, {11, 3, 1}, {13, 4, 2}, {13, 4, 2}, {11, 3, 1}, {11, 3, 2}, {6, 2, 0},
         {5, 1, 0}, {13, 4, 3}, {6, 2, 1},  {2, 0, 0},  {5, 1, 2},  {11, 3, 4}, {2, 0, 0}},

        {{1, 0, 0},  {1, 0, 1}, {5, 2, 1},  {4, 1, 0},  {5, 2, 1},  {4, 1, 1},  {1, 0, 0},
         {12, 4, 2}, {1, 0, 0}, {10, 3, 1}, {12, 4, 1}, {12, 4, 2}, {10, 3, 1}, {10, 3, 2},
         {5, 2, 0},  {4, 1, 0}, {12, 4, 3}, {5, 2, 1},  {1, 0, 0},  {4, 1, 2},  {10, 3, 3}},

        {{9, 3, 3}, {0, 0, 0},  {0, 0, 0}, {4, 2, 0},  {3, 1, 0},  {4, 2, 1},  {3, 1, 1},
         {0, 0, 0}, {11, 4, 1}, {0, 0, 0}, {9, 3, 0},  {11, 4, 1}, {11, 4, 2}, {9, 3, 1},
         {9, 3, 2}, {4, 2, 0},  {3, 1, 0}, {11, 4, 3}, {4, 2, 1},  {0, 0, 0},  {3, 1, 1}},

        {{2, 0, 1}, {8, 2, 3},  {20, 4, 4}, {20, 4, 4}, {3, 1, 0},  {2, 0, 0},  {3, 1, 1},
         {2, 0, 1}, {20, 4, 4}, {10, 3, 1}, {20, 4, 4}, {8, 2, 0},  {10, 3, 1}, {10, 3, 2},
         {8, 2, 1}, {8, 2, 2},  {3, 1, 0},  {2, 0, 0},  {10, 3, 3}, {3, 1, 1},  {2, 0, 0}} };
};

using MergeStageSuffix = MergeStageSuffixS0;
//Change for different DC----------------------------------------------------------------------------------------------------------------------------
using DCX = DC7;
//------------------------------------------------------------------------------------------------------------------------------------

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
    __host__ __device__ cuda::std::tuple<unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&> operator()(kmerDCX& key) const
    {
        return { key.kmer[0],key.kmer[1],key.kmer[2],key.kmer[3],key.kmer[4],key.kmer[5],key.kmer[6],key.kmer[7],key.kmer[8],key.kmer[9],key.kmer[10],key.kmer[11],key.kmer[12] };
    }
};
struct dc21_kmer_decomposer
{
    __host__ __device__ cuda::std::tuple<unsigned char&, unsigned char&, unsigned  char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&> operator()(kmerDCX& key) const
    {
        return { key.kmer[0],key.kmer[1],key.kmer[2],key.kmer[3],key.kmer[4],key.kmer[5],key.kmer[6],key.kmer[7],key.kmer[8],key.kmer[9],key.kmer[10],key.kmer[11],key.kmer[12],key.kmer[13],key.kmer[14],key.kmer[15],key.kmer[16],key.kmer[17],key.kmer[18],key.kmer[19],key.kmer[20] };
    }
};

using kmer = kmerDCX; // for dc3 is uint64_t better but also needs some readjustment in code
//Change for different DC----------------------------------------------------------------------------------------------------------------------------
using DCXKmerDecomposer = dc7_kmer_decomposer;
//------------------------------------------------------------------------------------------------------------------------------------

using D_DCX = _D_DCX<DCX::X, DCX::C>;
struct MergeSuffixes {
    sa_index_t index;
    std::array<sa_index_t, DCX::C> ranks;
    std::array<unsigned char, DCX::X> prefix;
};

__constant__ uint32_t lookupNext[DCX::X][DCX::X][3];

struct decomposer_3_prefix
{
    __host__ __device__ cuda::std::tuple<unsigned char&, unsigned char&, unsigned char&> operator()(MergeSuffixes& key) const
    {
        return { key.prefix[0],key.prefix[1],key.prefix[2] };
    }
};
struct decomposer_7_prefix
{
    __host__ __device__ cuda::std::tuple<unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&> operator()(MergeSuffixes& key) const
    {
        return { key.prefix[0],key.prefix[1],key.prefix[2],key.prefix[3],key.prefix[4],key.prefix[5],key.prefix[6] };
    }
};
struct decomposer_13_prefix
{
    __host__ __device__ cuda::std::tuple<unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&> operator()(MergeSuffixes& key) const
    {
        return { key.prefix[0],key.prefix[1],key.prefix[2],key.prefix[3],key.prefix[4],key.prefix[5],key.prefix[6],key.prefix[7],key.prefix[8],key.prefix[9],key.prefix[10],key.prefix[11],key.prefix[12] };
    }
};
struct decomposer_21_prefix
{
    __host__ __device__ cuda::std::tuple<unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&, unsigned char&> operator()(MergeSuffixes& key) const
    {
        return { key.prefix[0],key.prefix[1],key.prefix[2],key.prefix[3],key.prefix[4],key.prefix[5],key.prefix[6],key.prefix[7],key.prefix[8],key.prefix[9],key.prefix[10],key.prefix[11],key.prefix[12],key.prefix[13],key.prefix[14],key.prefix[15],key.prefix[16],key.prefix[17],key.prefix[18],key.prefix[19],key.prefix[20] };
    }
};

//Change for different DC----------------------------------------------------------------------------------------------------------------------------
using decomposer_x_prefix = decomposer_7_prefix;
//------------------------------------------------------------------------------------------------------------------------------------


struct Compare_Prefix_Opt {
    __device__ __forceinline__ static int cmp8(const unsigned char* pa, const unsigned char* pb)
    {
        uint64_t va = *reinterpret_cast<const uint64_t*>(pa);
        uint64_t vb = *reinterpret_cast<const uint64_t*>(pb);
        if (va == vb) return 0;

        uint64_t diff = va ^ vb;               // non-zero
        int bit = __ffsll(diff);               // 1-based index of least-significant set bit
        int byte = (bit - 1) >> 3;             // byte index inside this 8-byte word (0..7)
        uint8_t ca = static_cast<uint8_t>((va >> (byte * 8)) & 0xFF);
        uint8_t cb = static_cast<uint8_t>((vb >> (byte * 8)) & 0xFF);
        return (ca < cb) ? -1 : 1;
    }

    __device__ __forceinline__ static int prefix_cmp(const unsigned char* pa, const unsigned char* pb)
    {
        // Compare 8 bytes at a time
        size_t offset = 0;
        const size_t N = DCX::X;
        for (; offset + 8 <= N; offset += 8)
        {
            int c = cmp8(pa + offset, pb + offset);
            if (c < 0) return -1;
            if (c > 0) return 1;
        }

        // tail: remaining bytes (<8)
        for (; offset < N; ++offset)
        {
            if (pa[offset] < pb[offset]) return -1;
            if (pa[offset] > pb[offset]) return 1;
        }

        return 0;
    }
};

struct DCXComparatorDeviceOpt
{
    __device__ __forceinline__ bool operator()(const MergeSuffixes& a, const MergeSuffixes& b) const
    {
        const unsigned char* pa = reinterpret_cast<const unsigned char*>(a.prefix.data());
        const unsigned char* pb = reinterpret_cast<const unsigned char*>(b.prefix.data());
        int c = Compare_Prefix_Opt::prefix_cmp(pa, pb);
        if (c < 0) return true;
        if (c > 0) return false;

        // prefixes equal -> compare ranks (same as original)
        uint32_t ia = a.index % DCX::X;
        uint32_t ib = b.index % DCX::X;
        uint32_t r1 = lookupNext[ia][ib][1];
        uint32_t r2 = lookupNext[ia][ib][2];
        return a.ranks[r1] < b.ranks[r2];
    }
};

// this struct is only used to make sure the right == operator is used during the DeviceRunLengthEncode
struct MergeSuffixesPrefixCompare {
    sa_index_t index;
    std::array<sa_index_t, DCX::C> ranks;
    std::array<unsigned char, DCX::X> prefix;
};
__device__ __forceinline__ bool operator==(const MergeSuffixesPrefixCompare& a, const MergeSuffixesPrefixCompare& b)
{
    const unsigned char* pa = reinterpret_cast<const unsigned char*>(a.prefix.data());
    const unsigned char* pb = reinterpret_cast<const unsigned char*>(b.prefix.data());
    if (Compare_Prefix_Opt::prefix_cmp(pa, pb) != 0) {
        return false;
    }
    return true;
}

__host__ __device__ __forceinline__ bool operator==(const kmerDCX& a, const kmerDCX& b)
{
    for (size_t i = 0; i < DCX::X; i++)
    {
        if (a.kmer[i] == b.kmer[i]) {
            continue;
        }
        return false;
    }
    return true;
}

struct KmerComparator
{
    __host__ __device__ __forceinline__ bool operator()(const kmerDCX& a, const kmerDCX& b)
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

struct DCXCompareRanks
{
    __device__ __forceinline__ bool operator()(const MergeSuffixes& a, const MergeSuffixes& b)
    {
        uint32_t r1 = lookupNext[a.index % DCX::X][b.index % DCX::X][1];
        uint32_t r2 = lookupNext[a.index % DCX::X][b.index % DCX::X][2];
        return a.ranks[r1] < b.ranks[r2];
    }
};

__host__ __device__ __forceinline__ bool operator==(const MergeSuffixes& a, const MergeSuffixes& b)
{
    for (size_t i = 0; i < DCX::X; i++)
    {
        if (a.prefix[i] != b.prefix[i]) {
            return false;
        }
    }
    for (size_t i = 0; i < DCX::C; i++)
    {
        if (a.ranks[i] != b.ranks[i]) {
            return false;
        }
    }
    return a.index == b.index;
}


struct DCXComparatorDeviceUnOpt
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
        if (a.ranks[r1] < b.ranks[r2]) {
            return true;
        }
        if (a.ranks[r1] > b.ranks[r2]) {
            return false;
        }
        return a.index < b.index;
    }
};

struct DCXComparatorHost
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
        if (a.ranks[r1] < b.ranks[r2]) {
            return true;
        }
        if (a.ranks[r1] > b.ranks[r2]) {
            return false;
        }
        return a.index < b.index;
    }
};

__host__ __forceinline__ bool operator<(const MergeSuffixes& a, const MergeSuffixes& b)
{
    sizeof(MergeSuffixes);
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
    if (a.ranks[r1] < b.ranks[r2]) {
        return true;
    }
    if (a.ranks[r1] > b.ranks[r2]) {
        return false;
    }
    return a.index < b.index;

}

using DCXComparatorDevice = DCXComparatorDeviceOpt;
#endif // CONFIG_H
