#pragma once

#include <vector>
#include <array>

using uint = unsigned int;

template<typename index_t>
struct InterNodeCopy {
    uint src_node, dest_node;
    index_t src_index, dest_index;
    size_t len;
    int extra;
};

template <typename key_t, typename value_t, typename index_t>
struct MultiSplitNodeInfoT {
    key_t* src_keys;
    value_t* src_values;
    index_t src_len;

    key_t* dest_keys;
    value_t* dest_values;
    index_t dest_len;
};

template <typename key_t, typename value_t, typename index_t>
struct All2AllNodeInfoT {
        key_t* src_keys;
        value_t* src_values;
        index_t src_len;

        key_t* dest_keys;
        value_t* dest_values;
        index_t dest_len;

        key_t* temp_keys;
        value_t* temp_values;
        index_t temp_len;
};


template<typename table_t, size_t NUM_GPUS>
using split_table_tt = std::array<std::array<table_t, NUM_GPUS>, NUM_GPUS>;

