#ifndef DISTRIB_MERGE_ARRAY_HPP
#define DISTRIB_MERGE_ARRAY_HPP

#include <array>

namespace distrib_merge {

using uint = unsigned int;

template<typename key_t, typename value_t, typename index_t>
struct DistributedArrayNode {
    uint node_index;
    index_t count;
    key_t* keys;
    value_t* values;

    key_t* keys_buffer;
    value_t* values_buffer;
};

template <typename key_t, typename value_t, typename index_t, size_t NUM_NODES>
using DistributedArray = std::array<DistributedArrayNode<key_t, value_t, index_t>, NUM_NODES>;

}


#endif // DISTRIB_MERGE_ARRAY_HPP
