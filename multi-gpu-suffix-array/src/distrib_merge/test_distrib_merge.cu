#include "distrib_merge.hpp"
#include "../gossip/context.cuh"
#include <cuda_runtime.h>
#include <algorithm>

// This test is currently limited to NUM_GPUS*SIZE_PER_GPU < 4 G, as InterNodeCopy is fixed to using 32 bit
// indices somewhere.

static const size_t NUM_GPUS = 8;
static const size_t SIZE_PER_GPU = 1024 * 1024 * 400;

#ifdef DGX1_TOPOLOGY
static_assert(NUM_GPUS == 8, "DGX-1 topology can only be used with 8 GPUs");
template<typename key_t, typename value_t, typename index_t, size_t NUM_GPUS>
using DistribMergeTopology = distrib_merge::DGX1TopologyHelper<key_t, value_t, index_t, NUM_GPUS>;
#else
template<typename key_t, typename value_t, typename index_t, size_t NUM_GPUS>
using DistribMergeTopology = distrib_merge::DistribMergeAllConnectedTopologyHelper<key_t, value_t, index_t, NUM_GPUS>;
#endif



template<typename type_t>
struct LessThanFunctor {
    HOST_DEVICE bool operator()(const type_t& a, const type_t& b) const {
        return a < b;
    }
};

template <typename key_t, typename value_t, typename index_t, size_t NUM_GPUS_,
    template <typename, typename, typename, size_t> class TopologyHelper,
    size_t SIZE_PER_GPU>
class DistribMergeTest {
public:
    using Context = MultiGPUContext<NUM_GPUS_>;
    static const size_t NUM_GPUS = NUM_GPUS_;
private:
    using DistributedArray = distrib_merge::DistributedArray<key_t, value_t, uint, NUM_GPUS>;

    Context& mcontext;

    key_t* h_inout, * h_temp, * h_result;

    static const size_t HOST_TEMP_SIZE = 4096;
    static const size_t DEVICE_TEMP_SIZE = 4 * 1024 * 1024;

    // We load input to d_inout, then copy to buffer (within merge) and merge back to d_inout.
    std::array<key_t*, NUM_GPUS> d_inout;
    std::array<key_t*, NUM_GPUS> d_buffer;
    std::array<key_t*, NUM_GPUS> d_temp;

public:

    DistribMergeTest(Context& context)
        : mcontext(context) {
    }

    key_t* inp_a() {
        return h_inout;
    }

    key_t* inp_b() {
        return h_inout + SIZE_PER_GPU * NUM_GPUS;
    }

    key_t* out() {
        return h_inout;
    }

    void upload_data() {
        for (uint i = 0; i < NUM_GPUS; ++i) {
            //(mcontext.get_device_id(i));
            cudaMemcpyAsync(d_inout[i], inp_a() + i * SIZE_PER_GPU,
                SIZE_PER_GPU * sizeof(key_t),
                cudaMemcpyHostToDevice, mcontext.get_gpu_default_stream(i));
            CUERR;
            cudaMemcpyAsync(d_inout[i] + SIZE_PER_GPU, inp_b() + i * SIZE_PER_GPU,
                SIZE_PER_GPU * sizeof(key_t),
                cudaMemcpyHostToDevice, mcontext.get_gpu_default_stream(i));
            CUERR;
        }
        mcontext.sync_default_streams();
    }

    void download_data() {

        //            memset(h_inout, 0, sizeof(key_t)*SIZE_PER_GPU * NUM_GPUS*2);

        for (uint i = 0; i < NUM_GPUS; ++i) {
            //(mcontext.get_device_id(i));
            cudaMemcpyAsync(h_inout + i * 2 * SIZE_PER_GPU, d_inout[i], 2 * SIZE_PER_GPU * sizeof(key_t),
                cudaMemcpyDeviceToHost,
                mcontext.get_gpu_default_stream(i));
            CUERR;
        }
        mcontext.sync_default_streams();
    }

    void alloc() {
        cudaMallocHost(&h_inout, SIZE_PER_GPU * NUM_GPUS * 2 * sizeof(key_t));
        cudaMallocHost(&h_temp, HOST_TEMP_SIZE);
        h_result = new key_t[SIZE_PER_GPU * NUM_GPUS * 2];
        CUERR;
        for (uint i = 0; i < NUM_GPUS; ++i) {
            //(mcontext.get_device_id(i));
            cudaMalloc(&d_inout[i], 2 * SIZE_PER_GPU * sizeof(key_t));
            cudaMalloc(&d_buffer[i], 2 * SIZE_PER_GPU * sizeof(key_t));
            cudaMalloc(&d_temp[i], DEVICE_TEMP_SIZE);
        } CUERR;
    }

    void free() {
        cudaFreeHost(h_inout);
        cudaFreeHost(h_temp);
        for (uint i = 0; i < NUM_GPUS; ++i) {
            //(mcontext.get_device_id(i));
            cudaFree(d_inout[i]);
            cudaFree(d_buffer[i]);
            cudaFree(d_temp[i]);
        }
        delete[] h_result;
    }

    void merge_host() {
        std::merge(h_inout, h_inout + NUM_GPUS * SIZE_PER_GPU,
            h_inout + NUM_GPUS * SIZE_PER_GPU, h_inout + 2 * NUM_GPUS * SIZE_PER_GPU,
            h_result);
    }

    bool check() const {
        return std::equal(h_result, h_result + 2 * NUM_GPUS * SIZE_PER_GPU,
            h_inout);
    }

    void merge_gpu() {
        distrib_merge::DistributedArray<key_t, value_t, index_t, NUM_GPUS> inp_a, inp_b, out;
        for (uint i = 0; i < NUM_GPUS; ++i) {
            inp_a[i] = { i, SIZE_PER_GPU, d_inout[i], nullptr, nullptr, nullptr };
            inp_b[i] = { i, SIZE_PER_GPU, d_inout[i] + SIZE_PER_GPU, nullptr, nullptr, nullptr };
            out[i] = { i, 2 * SIZE_PER_GPU, d_inout[i], nullptr, d_buffer[i], nullptr };
            mcontext.get_device_temp_allocator(i).init(d_temp[i], DEVICE_TEMP_SIZE);
        }
        QDAllocator qd_alloc(h_temp, HOST_TEMP_SIZE);
        distrib_merge::DistributedMerge<key_t, value_t, index_t, NUM_GPUS,
            TopologyHelper>::
            merge_async(inp_a, inp_b, out, LessThanFunctor<key_t>(), false, mcontext, qd_alloc);
        mcontext.sync_default_streams();
    }

};

template <typename int_t>
void prepare_input_data(int_t* inp_a, size_t a_size,
    int_t* inp_b, size_t b_size) {
    for (int_t i = 0; i < a_size; ++i) {
        inp_a[i] = 2 * i;
    }
    for (int_t i = 0; i < b_size; ++i) {
        inp_b[i] = 2 * i + 1;
    }
}

template <typename key_t>
void dump_array(const key_t* array, size_t size) {
    std::copy(array, array + size, std::ostream_iterator<int>(std::cout, " ,"));
    std::cout << "\n";
}

int main() {
    MultiGPUContext<NUM_GPUS> context;

    DistribMergeTest<uint64_t, uint64_t, uint, NUM_GPUS, DistribMergeTopology, SIZE_PER_GPU> merge_test(context);

    TIMERSTART(alloc);
    merge_test.alloc();
    TIMERSTOP(alloc);

    TIMERSTART(create_input);
    prepare_input_data(merge_test.inp_a(), SIZE_PER_GPU * NUM_GPUS,
        merge_test.inp_b(), SIZE_PER_GPU * NUM_GPUS);
    TIMERSTOP(create_input);

    //    std::cout << "Input data: \nA:\n";
    //    dump_array(merge_test.inp_a(), SIZE_PER_GPU*NUM_GPUS);
    //    std::cout << "B:\n";
    //    dump_array(merge_test.inp_b(), SIZE_PER_GPU*NUM_GPUS);

    TIMERSTART(upload);
    merge_test.upload_data();
    TIMERSTOP(upload);

    TIMERSTART(merge_gpu);
    merge_test.merge_gpu();
    TIMERSTOP(merge_gpu);

    context.sync_default_streams();

    TIMERSTART(merge_cpu);
    merge_test.merge_host();
    TIMERSTOP(merge_cpu);

    TIMERSTART(download);
    merge_test.download_data();
    TIMERSTOP(download);


    TIMERSTART(check);
    bool correct = merge_test.check();
    TIMERSTOP(check);

    if (correct)
        std::cout << "Everything ok.\n";
    else
        std::cout << "WRONG.\n";

    //    std::cout << "Output data: \n";
    //    dump_array(merge_test.out(), 2*SIZE_PER_GPU*NUM_GPUS);

    merge_test.free();

}
