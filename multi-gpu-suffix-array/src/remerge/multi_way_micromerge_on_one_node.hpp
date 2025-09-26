#ifndef MULTI_WAY_MICROMERGE_ON_ONE_NODE_HPP
#define MULTI_WAY_MICROMERGE_ON_ONE_NODE_HPP

#include "merge_types.hpp"
#include "gossip/context.cuh"
#include "moderngpu/kernel_merge.hxx"

namespace crossGPUReMerge {

    template <size_t NUM_GPUS, class mtypes, class comp_fun_t>
    class NodeMultiMerger {
        using Context = MultiGPUContext<NUM_GPUS>;
        using MergeNode = MergeNode<mtypes>;
    public:
        NodeMultiMerger(Context& context, const MergeNode& node, const MultiWayMergePartition& part,
            comp_fun_t comp, bool do_values)
            : mcontext(context), mnode(node), mpart(part), mcomp(comp),
            mranges(part.dest_micro_ranges), mdo_values(do_values)
        {
            mnranges.reserve(mranges.size());
            msrc_ranges = &mranges; // Careful: these can cause mayhem when stored in a growing vector.
            mdest_ranges = &mnranges; //        without calling reserve beforehand.
            msrc_key_buff = node.info.key_buffer;
            mdest_key_buff = node.info.keys;
            msrc_value_buff = node.info.value_buffer;
            mdest_value_buff = node.info.values;
            mstreams_used.reserve(NUM_GPUS);
        }

        bool do_merge_step() {
            //(mcontext.get_device_id(mnode.info.index));
            mstreams_used.clear();
            if (msrc_ranges->size() > 1) {
                uint stream_index = 0;
                mdest_ranges->clear();

                for (uint i = 0; i < msrc_ranges->size() - 1; i += 2) {
                    // Yes, we try to do multiple merges at once, but it does not really work, since modernGPU
                    // tries to use too much shared memory when increasing the number of values processed by
                    // one thread. A real multi-way-merge would be needed.
                    mgpu::my_mpgu_context_t& mgpu_context = *mcontext.get_mgpu_contexts_for_device(mnode.info.index)[stream_index];
                    const sa_index_range_t& r1 = (*msrc_ranges)[i];
                    const sa_index_range_t& r2 = (*msrc_ranges)[i + 1];
                    //                    printf("\nMerging %d - %d, %d - %d to %d on node %u, stream %u\n", r1.start, r1.end, r2.start, r2.end,
                    //                           r1.start, mnode.info.index, stream_index);
                    if (mdo_values) {
                        mgpu::merge(msrc_key_buff + r1.start, msrc_value_buff + r1.start, r1.end - r1.start,
                            msrc_key_buff + r2.start, msrc_value_buff + r2.start, r2.end - r2.start,
                            mdest_key_buff + r1.start, mdest_value_buff + r1.start, mcomp, mgpu_context);CUERR;
                    }
                    else {
                        mgpu::merge(msrc_key_buff + r1.start, r1.end - r1.start,
                            msrc_key_buff + r2.start, r2.end - r2.start,
                            mdest_key_buff + r1.start, mcomp, mgpu_context);CUERR;
                        mcontext.sync_all_streams();
                        printf("[%lu] merge\n", world_rank());
                    }

                    mdest_ranges->push_back({ r1.start, r2.end });

                    if (mstreams_used.size() < NUM_GPUS)
                        mstreams_used.push_back(stream_index);
                    stream_index = (stream_index + 1) % NUM_GPUS;
                }
                if (msrc_ranges->size() > 1 && msrc_ranges->size() % 2 > 0) {
                    const sa_index_range_t& odd_range = msrc_ranges->back();
                    mdest_ranges->push_back(msrc_ranges->back());

                    cudaMemcpyAsync(mdest_key_buff + odd_range.start, msrc_key_buff + odd_range.start,
                        sizeof(typename mtypes::key_t) * (odd_range.end - odd_range.start), cudaMemcpyDeviceToDevice,
                        mcontext.get_streams(mnode.info.index)[stream_index]); CUERR
                        mcontext.sync_all_streams();
                    printf("[%lu] cudamemcpy\n", world_rank());
                    if (mstreams_used.size() < NUM_GPUS)
                        mstreams_used.push_back(stream_index);
                    if (mdo_values) {
                        stream_index = (stream_index + 1) % NUM_GPUS;

                        cudaMemcpyAsync(mdest_value_buff + odd_range.start, msrc_value_buff + odd_range.start,
                            sizeof(typename mtypes::value_t) * (odd_range.end - odd_range.start), cudaMemcpyDeviceToDevice,
                            mcontext.get_streams(mnode.info.index)[stream_index]); CUERR

                            if (mstreams_used.size() < NUM_GPUS)
                                mstreams_used.push_back(stream_index);
                    }
                }
                std::swap(msrc_ranges, mdest_ranges);
                std::swap(msrc_key_buff, mdest_key_buff);
                std::swap(msrc_value_buff, mdest_value_buff);
                return false;
            }
            else {
                if (msrc_key_buff == mnode.info.key_buffer) {
                    //                    printf("\nDoing extra copy...\n");
                    cudaMemcpyAsync(mnode.info.keys + mpart.dest_range.start, mnode.info.key_buffer + mpart.dest_range.start,
                        (mpart.dest_range.end - mpart.dest_range.start) * sizeof(typename mtypes::key_t),
                        cudaMemcpyDeviceToDevice,
                        mcontext.get_gpu_default_stream(mnode.info.index));CUERR;
                    mcontext.sync_all_streams();
                    printf("[%lu] cudamemcpy2\n", world_rank());

                    //                            mcontext.sync_gpu_default_stream(node.info.index);
                    mstreams_used.push_back(0);
                    if (mdo_values) {
                        cudaMemcpyAsync(mnode.info.values + mpart.dest_range.start, mnode.info.value_buffer + mpart.dest_range.start,
                            (mpart.dest_range.end - mpart.dest_range.start) * sizeof(typename mtypes::value_t),
                            cudaMemcpyDeviceToDevice,
                            mcontext.get_streams(mnode.info.index)[1]);CUERR;
                        mstreams_used.push_back(1);
                    }
                }
                return true;
            }
        }

        void sync_used_streams() {
            for (uint stream_index : mstreams_used) {
                cudaStreamSynchronize(mcontext.get_streams(mnode.info.index)[stream_index]);CUERR;
            }
            mstreams_used.clear();
        }

    private:
        Context& mcontext;
        const MergeNode& mnode;
        const MultiWayMergePartition& mpart;
        comp_fun_t mcomp;
        std::vector<sa_index_range_t> mranges;
        std::vector<sa_index_range_t> mnranges;
        std::vector<uint> mstreams_used;
        std::vector <sa_index_range_t>* msrc_ranges;
        std::vector <sa_index_range_t>* mdest_ranges;
        typename mtypes::key_t* msrc_key_buff;
        typename mtypes::key_t* mdest_key_buff;
        typename mtypes::value_t* msrc_value_buff;
        typename mtypes::value_t* mdest_value_buff;
        bool mdo_values;
    };

    // Here we plan for one node sequentially, using only one stream. This is not much slower, unfortunately.
    template <size_t NUM_GPUS, class mtypes>
    void do_multi_merges(MultiGPUContext<NUM_GPUS>& context, const MergeNode<mtypes>& node,
        const MultiWayMergePartition& p, bool do_values) {
        std::vector<sa_index_range_t> ranges;
        std::vector<sa_index_range_t> nranges;

        ranges = p.dest_micro_ranges;
        nranges.reserve(ranges.size());

        mgpu::my_mpgu_context_t& mgpu_context = context.get_mgpu_context_for_device(node.info.index);

        std::vector <sa_index_range_t>* src_ranges = &ranges;
        std::vector <sa_index_range_t>* dest_ranges = &nranges;

        sa_index_t* src_key_buff = node.info.key_buffer;
        sa_index_t* dest_key_buff = node.info.keys;

        sa_index_t* src_value_buff = node.info.value_buffer;
        sa_index_t* dest_value_buff = node.info.values;

        while (src_ranges->size() > 1) {
            dest_ranges->clear();
            for (uint i = 0; i < src_ranges->size() - 1; i += 2) {
                const sa_index_range_t& r1 = (*src_ranges)[i];
                const sa_index_range_t& r2 = (*src_ranges)[i + 1];
                printf("\nMerging %d - %d,  %d - %d to %d on node %u", r1.start, r1.end, r2.start, r2.end,
                    r1.start, node.info.index);

                mgpu_context.reset_temp_memory();

                if (do_values) {
                    mgpu::merge(src_key_buff + r1.start, src_value_buff + r1.start, r1.end - r1.start,
                        src_key_buff + r2.start, src_value_buff + r2.start, r2.end - r2.start,
                        dest_key_buff + r1.start, dest_value_buff + r1.start,
                        mgpu::less_t<sa_index_t>(), mgpu_context);CUERR;
                }
                else {
                    mgpu::merge(src_key_buff + r1.start, r1.end - r1.start,
                        src_key_buff + r2.start, r2.end - r2.start,
                        dest_key_buff + r1.start, mgpu::less_t<sa_index_t>(), mgpu_context);CUERR;
                }
                dest_ranges->push_back({ r1.start, r2.end });
            }
            if (src_ranges->size() > 1 && src_ranges->size() % 2 > 0)
                dest_ranges->push_back(src_ranges->back());
            std::swap(src_ranges, dest_ranges);
            std::swap(src_key_buff, dest_key_buff);
            std::swap(src_value_buff, dest_value_buff);
        }

        // Just got swapped...
        if (src_key_buff == node.info.key_buffer) {
            //                printf("\nDoing extra copy...\n");
            cudaMemcpyAsync(node.info.keys + p.dest_range.start, node.info.key_buffer + p.dest_range.start,
                (p.dest_range.end - p.dest_range.start) * sizeof(typename mtypes::key_t),
                cudaMemcpyDeviceToDevice,
                context.get_gpu_default_stream(node.info.index));CUERR;
            cudaMemcpyAsync(node.info.values + p.dest_range.start, node.info.value_buffer + p.dest_range.start,
                (p.dest_range.end - p.dest_range.start) * sizeof(typename mtypes::value_t),
                cudaMemcpyDeviceToDevice,
                context.get_streams(node.info.index)[1]);CUERR;

            //                mcontext.sync_gpu_default_stream(node.info.index);
        }
    }

}
#endif // MULTI_WAY_MICROMERGE_ON_ONE_NODE_HPP
