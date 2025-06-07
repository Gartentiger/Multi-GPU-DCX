#ifndef MERGENODEUTILS_H
#define MERGENODEUTILS_H

#include "merge_types.hpp"
#include <vector>

namespace crossGPUReMerge {

    class MergeNodeUtils
    {
    public:
        MergeNodeUtils() {
        }

        void init(std::vector<size_t>&& num_elements) {
            mnode_elements = std::move(num_elements);
        }

        sa_index_t get_node_num_elements(uint node) const {
            return mnode_elements[node];
        }

        // first call range.start == 0, range.end == NUM_GPUS-1 
        template <typename func_t>
        void convert_to_per_node_ranges(const MergeRange& range, const func_t& f) const {
            ASSERT(range.start.node <= range.end.node);
            if (range.start.node == range.end.node) {
                f(range.start.node, { range.start.index, range.end.index });
            }
            else {
                f(range.start.node, { range.start.index, get_node_num_elements(range.start.node) });
                for (uint node = range.start.node + 1; node < range.end.node; ++node) {
                    f(node, { 0, get_node_num_elements(node) });
                }
                f(range.end.node, { 0, range.end.index });
            }
        }

        MergePosition safe_increment(MergePosition p, const MergeRange& range) const {
            ASSERT(p.node >= range.start.node && p.node <= range.end.node);
            ++p.index;
            // We allow pointing past the very last element without switching to the next node.
            if (p.index == get_node_num_elements(p.node) && p.node != range.end.node) {
                p.index = 0;
                ++p.node;
                ASSERT(p.node <= range.end.node);
            }
            ASSERT(p.node == range.end.node ? p.index <= range.end.index :
                p.index <= get_node_num_elements(p.node));
            return p;
        }

        static MergeRange combine_ranges(const MergeRange& r1, const MergeRange& r2) {
            return { { r1.start.node, r1.start.index }, { r2.end.node, r2.end.index } };
        }


    private:
        std::vector<size_t> mnode_elements;
    };

}
#endif // MERGENODEUTILS_H
