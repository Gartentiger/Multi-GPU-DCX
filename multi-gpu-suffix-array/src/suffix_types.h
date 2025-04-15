#ifndef CONFIG_H
#define CONFIG_H

#include <cstdint>

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

using MergeStageSuffix = MergeStageSuffixS0;

#endif // CONFIG_H
