#include "suffix_array_kernels.cuh"
#include <iostream>
#include <cassert>

namespace kernels {

    // TODO: const correctness, vectorizing, shared memory

    // Write thread index if input[i-1] != input[i], else write default value
    __global__ void write_ranks_diff(const sa_index_t* Input_data, sa_index_t* Output_rank, sa_index_t base, sa_index_t def_value, size_t N) {
        for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
            tidx < N; tidx += blockDim.x * gridDim.x) {
            sa_index_t out;
            if (tidx > 0 && Input_data[tidx - 1] != Input_data[tidx]) {
                out = tidx + base;
            }
            else {
                out = (tidx == 0) ? base : def_value;
            }
            Output_rank[tidx] = out;
        }
    }

    __global__ void write_if_eq(const sa_index_t* Input, sa_index_t* Output, sa_index_t value_eq,
        sa_index_t value_to_write, sa_index_t N) {
        for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
            tidx < N; tidx += blockDim.x * gridDim.x) {
            if (Input[tidx] == value_eq) {
                Output[tidx] = value_to_write;
            }
        }

    }

    __global__ void write_compact_flags_multi(const sa_index_t* Ranks, const sa_index_t* Last_rank_prev,
        const sa_index_t* First_rank_next, sa_index_t* flags,
        size_t N) {
        for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
            tidx < N; tidx += blockDim.x * gridDim.x) {
            sa_index_t v, b, n;
            v = Ranks[tidx];
            if (tidx > 0) {
                b = Ranks[tidx - 1];
            }
            else {
                if (Last_rank_prev) {
                    b = *Last_rank_prev;
                }
                else {
                    b = v - 1; // take something that will be different
                }
            }
            if (tidx < N - 1) {
                n = Ranks[tidx + 1];
            }
            else {
                if (First_rank_next) {
                    n = *First_rank_next;
                }
                else {
                    n = v + 1; // take something that will be different
                }
            }
            flags[tidx] = !(v != b && v != n);
        }
    }


    __device__ __forceinline__ sa_index_t cap_index(sa_index_t index, sa_index_t max_index, sa_index_t h) {
        uint64_t i = (uint64_t)index + (uint64_t)h;
        if (i > (uint64_t)max_index)
            return 0;
        return (sa_index_t)i;
    }

    __global__ void write_sa_index_adding_h(const sa_index_t* Index, sa_index_t h, sa_index_t max_index,
        sa_index_t* Index_out, sa_index_t* Org_index_out, size_t N)
    {
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i = tidx; i < N / 2; i += blockDim.x * gridDim.x) {
            uint2 index, org_index;
            index = reinterpret_cast<const uint2*>(Index)[i];
            index.x = cap_index(index.x, max_index, h);
            index.y = cap_index(index.y, max_index, h);
            org_index.x = i * 2;
            org_index.y = i * 2 + 1;
            reinterpret_cast<uint2*>(Index_out)[i] = index;
            reinterpret_cast<uint2*>(Org_index_out)[i] = org_index;
        }

        if (tidx == blockDim.x * gridDim.x - 1 && N % 2 == 1) {
            uint index = Index[N - 1];
            index = cap_index(index, max_index, h);
            Index_out[N - 1] = index;
            Org_index_out[N - 1] = N - 1;
        }
    }

    __global__ void sub_value(const sa_index_t* values, sa_index_t* Out,
        sa_index_t sub, sa_index_t N)
    {
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i = tidx; i < N; i += blockDim.x * gridDim.x) {
            sa_index_t v = values[i];
            v -= sub;
            Out[i] = v;
        }
    }


    __global__ void fetch_isa_multi(const sa_index_t* Sa_idx, const sa_index_t* Isa, sa_index_t base_offset,
        sa_index_t* Rank_out, size_t N, size_t ISA_LEN) {
        for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
            tidx < N; tidx += blockDim.x * gridDim.x) {
            sa_index_t sa = Sa_idx[tidx];
            sa_index_t r = 0;
            if (sa > 0) {
                sa -= base_offset;
                assert(sa < ISA_LEN);
                r = Isa[sa];
            }
            Rank_out[tidx] = r;
        }
    }

    __global__ void write_to_rank(const sa_index_t* Input_ranks, const sa_index_t* Target_index,
        sa_index_t* Output_ranks, size_t N)
    {
        for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
            tidx < N; tidx += blockDim.x * gridDim.x) {
            sa_index_t index = Target_index[tidx];
            sa_index_t rank = Input_ranks[tidx];
            Output_ranks[index] = rank;
        }
    }

    __global__ void write_to_isa_2(const sa_index_t* Input_ranks, const sa_index_t* Target_index,
        sa_index_t* Isa, sa_index_t N, size_t ISA_N)
    {
        // Non-coalesced write, maybe we can do better?
        for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
            tidx < N; tidx += blockDim.x * gridDim.x) {
            sa_index_t index = Target_index[tidx];
            sa_index_t rank = Input_ranks[tidx];
            assert(index < ISA_N);
            Isa[index] = rank;
        }
    }

    __global__ void write_to_isa_sub_offset(const sa_index_t* Input_ranks, const sa_index_t* Target_index,
        sa_index_t* Isa, sa_index_t Isa_base_offset, sa_index_t N, size_t ISA_N)
    {
        // Non-coalesced write, maybe we can do better?
        for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
            tidx < N; tidx += blockDim.x * gridDim.x) {
            sa_index_t index = Target_index[tidx] - Isa_base_offset;
            sa_index_t rank = Input_ranks[tidx];
            assert(index < ISA_N);
            Isa[index] = rank;
        }
    }

    __global__ void prepare_isa_transform(const sa_index_t* Isa, sa_index_t base_offset,
        sa_index_t* Ranks_out, sa_index_t* Index_out, size_t N) {
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i = tidx; i < N / 2; i += blockDim.x * gridDim.x) {
            uint2 rank;
            uint2 index_out;
            rank = reinterpret_cast<const uint2*>(Isa)[i];
            assert(rank.x > 0 && rank.y > 0);
            --rank.x;
            --rank.y;
            index_out.x = i * 2 + base_offset;
            index_out.y = i * 2 + 1 + base_offset;
            reinterpret_cast<uint2*>(Index_out)[i] = index_out;
            reinterpret_cast<uint2*>(Ranks_out)[i] = rank;
        }

        if (tidx == blockDim.x * gridDim.x - 1 && N % 2 == 1) {
            const uint rank = Isa[N - 1];
            assert(rank > 0);
            Index_out[N - 1] = N - 1 + base_offset;
            Ranks_out[N - 1] = rank - 1;
        }
    }

#define make_uchar4_be(x,y,z,w) make_uchar4(w,z,y,x)

    __device__ __forceinline__ uint get_quadruplet(uint4 values, uint indexmod) {
        uchar4 v0 = *reinterpret_cast<uchar4*>(&values.x);
        uchar4 v4 = *reinterpret_cast<uchar4*>(&values.y);
        uchar4 v8 = *reinterpret_cast<uchar4*>(&values.z);
        uchar4 v12 = *reinterpret_cast<uchar4*>(&values.w);
        uchar4 out;
        uint o;
        switch (indexmod) {
        case 1:
            out = make_uchar4_be(v0.y, v0.z, v0.w, v4.x);
            break;
        case 2:
            out = make_uchar4_be(v0.z, v0.w, v4.x, v4.y);
            break;
        case 4:
            out = make_uchar4_be(v4.x, v4.y, v4.z, v4.w);
            break;
        case 5:
            out = make_uchar4_be(v4.y, v4.z, v4.w, v8.x);
            break;
        case 7:
            out = make_uchar4_be(v4.w, v8.x, v8.y, v8.z);
            break;
        case 8:
            out = make_uchar4_be(v8.x, v8.y, v8.z, v8.w);
            break;
        case 10:
            out = make_uchar4_be(v8.z, v8.w, v12.x, v12.y);
            break;
        case 11:
            out = make_uchar4_be(v8.w, v12.x, v12.y, v12.z);
            break;
        }
        o = *reinterpret_cast<uint*>(&out);
        o |= 3;
        return o;
    }




    __global__ void produce_index_kmer_tuples_12(const char* Input, sa_index_t start_index, sa_index_t* Output_index,
        sa_index_t* Output_kmers, size_t N) {
        assert(N % 12 == 0); // No one wants to deal with the "tails" here, we just write some more and don't care.
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i = tidx; i < N / 12; i += blockDim.x * gridDim.x) {
            uint3 a = *(reinterpret_cast<const uint3*>(Input) + i);
            uint b = *(reinterpret_cast<const uint*>(Input) + (i + 1) * 3);
            uint4 inp = make_uint4(a.x, a.y, a.z, b);
            // Now generate
            uint4 res1;
            uint4 res2;
            uint base_idx = start_index + 8 * i;
            uint4 index1 = make_uint4(base_idx + 0, base_idx + 1, base_idx + 2, base_idx + 3);
            uint4 index2 = make_uint4(base_idx + 4, base_idx + 5, base_idx + 6, base_idx + 7);
            res1.x = get_quadruplet(inp, 1);
            res1.y = get_quadruplet(inp, 2);
            res1.z = get_quadruplet(inp, 4);
            res1.w = get_quadruplet(inp, 5);

            res2.x = get_quadruplet(inp, 7);
            res2.y = get_quadruplet(inp, 8);
            res2.z = get_quadruplet(inp, 10);
            res2.w = get_quadruplet(inp, 11);

            *(reinterpret_cast<uint4*>(Output_kmers) + i * 2) = res1;
            *(reinterpret_cast<uint4*>(Output_kmers) + i * 2 + 1) = res2;
            *(reinterpret_cast<uint4*>(Output_index) + i * 2) = index1;
            *(reinterpret_cast<uint4*>(Output_index) + i * 2 + 1) = index2;
        }
        // Maybe we could have nicer coalescing load/safe pattern with shared memory?
    }


    typedef unsigned char uchar;
    __device__ __forceinline__ ulong1 make_uchar8_be(uchar a, uchar b, uchar c, uchar d,
        uchar e, uchar f, uchar g, uchar h) {
        uint2 out;
        uchar4 x, y;
        x = make_uchar4(h, g, f, e);
        y = make_uchar4(d, c, b, a);
        out.x = *reinterpret_cast<uint*>(&x);
        out.y = *reinterpret_cast<uint*>(&y);
        return *reinterpret_cast<ulong1*>(&out);
    }

    __device__ __forceinline__ ulong1 get_octet(uint4 values, uint rem, uint indexmod) {
        uchar4 v0 = *reinterpret_cast<uchar4*>(&values.x);
        uchar4 v4 = *reinterpret_cast<uchar4*>(&values.y);
        uchar4 v8 = *reinterpret_cast<uchar4*>(&values.z);
        uchar4 v12 = *reinterpret_cast<uchar4*>(&values.w);
        uchar4 v16 = *reinterpret_cast<uchar4*>(&rem);
        ulong1 out;
        switch (indexmod) {
        case 1:
            out = make_uchar8_be(v0.y, v0.z, v0.w, v4.x, v4.y, v4.z, v4.w, v8.x);
            break;
        case 2:
            out = make_uchar8_be(v0.z, v0.w, v4.x, v4.y, v4.z, v4.w, v8.x, v8.y);
            break;
        case 4:
            out = make_uchar8_be(v4.x, v4.y, v4.z, v4.w, v8.x, v8.y, v8.z, v8.w);
            break;
        case 5:
            out = make_uchar8_be(v4.y, v4.z, v4.w, v8.x, v8.y, v8.z, v8.w, v12.x);
            break;
        case 7:
            out = make_uchar8_be(v4.w, v8.x, v8.y, v8.z, v8.w, v12.x, v12.y, v12.z);
            break;
        case 8:
            out = make_uchar8_be(v8.x, v8.y, v8.z, v8.w, v12.x, v12.y, v12.z, v12.w);
            break;
        case 10:
            out = make_uchar8_be(v8.z, v8.w, v12.x, v12.y, v12.z, v12.w, v16.x, v16.y);
            break;
        case 11:
            out = make_uchar8_be(v8.w, v12.x, v12.y, v12.z, v12.w, v16.x, v16.y, v16.z);
            break;
        }
        out.x &= ~((1ull << 13) - 1ull);
        out.x |= 7ull << 13;
        return out;
    }
    __device__ __forceinline__ ulong1 get_octet2(uint4 values, uint rem, uint indexmod) {
        uchar4 v0 = *reinterpret_cast<uchar4*>(&values.x);
        uchar4 v4 = *reinterpret_cast<uchar4*>(&values.y);
        uchar4 v8 = *reinterpret_cast<uchar4*>(&values.z);
        uchar4 v12 = *reinterpret_cast<uchar4*>(&values.w);
        uchar4 v16 = *reinterpret_cast<uchar4*>(&rem);
        ulong1 out;
        switch (indexmod) {
        case 1:
            out = make_uchar8_be(v0.y, v0.z, v0.w, v4.x, v4.y, v4.z, v4.w, v8.x);
            break;
        case 2:
            out = make_uchar8_be(v0.z, v0.w, v4.x, v4.y, v4.z, v4.w, v8.x, v8.y);
            break;
        case 4:
            out = make_uchar8_be(v4.x, v4.y, v4.z, v4.w, v8.x, v8.y, v8.z, v8.w);
            break;
        case 5:
            out = make_uchar8_be(v4.y, v4.z, v4.w, v8.x, v8.y, v8.z, v8.w, v12.x);
            break;
        case 6:
            out = make_uchar8_be(v4.z, v4.w, v8.x, v8.y, v8.z, v8.w, v12.x, v12.y);
            break;
        case 7:
            out = make_uchar8_be(v4.w, v8.x, v8.y, v8.z, v8.w, v12.x, v12.y, v12.z);
            break;
        case 8:
            out = make_uchar8_be(v8.x, v8.y, v8.z, v8.w, v12.x, v12.y, v12.z, v12.w);
            break;
        case 9:
            out = make_uchar8_be(v8.y, v8.z, v8.w, v12.x, v12.y, v12.z, v12.w, v16.x);
            break;
        case 10:
            out = make_uchar8_be(v8.z, v8.w, v12.x, v12.y, v12.z, v12.w, v16.x, v16.y);
            break;
        case 11:
            out = make_uchar8_be(v8.w, v12.x, v12.y, v12.z, v12.w, v16.x, v16.y, v16.z);
            break;
        }
        out.x &= ~((1ull << 13) - 1ull);
        out.x |= 7ull << 13;
        return out;
    }

    __global__ void produce_index_kmer_tuples_12_64_dcx(const char* Input, sa_index_t start_index, sa_index_t* Output_index,
        ulong1* Output_kmers, size_t N, size_t pd_elements) {
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;

        for (uint i = tidx; i < N / 7; i += blockDim.x * gridDim.x) {

        }

    }

    __global__ void produce_index_kmer_tuples_12_64_dc7(const char* Input, sa_index_t start_index, sa_index_t* Output_index,
        ulong1* Output_kmers, size_t N) {
        assert(N % 14 == 0); // No one wants to deal with the "tails" here, we just write some more and don't care.
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i = tidx; i < N / 14; i += blockDim.x * gridDim.x) {
            // printf("[%u] 1\n", tidx);
            // uint3 a = *(reinterpret_cast<const uint3*>(Input + 10 * i));
            const char* baseInput = Input + 14 * i;
            uchar axx = *(baseInput);     uchar ayx = *(baseInput + 4); uchar azx = *(baseInput + 8);
            uchar axy = *(baseInput + 1); uchar ayy = *(baseInput + 5); uchar azy = *(baseInput + 9);
            uchar axz = *(baseInput + 2); uchar ayz = *(baseInput + 6); uchar azz = *(baseInput + 10);
            uchar axw = *(baseInput + 3); uchar ayw = *(baseInput + 7); uchar azw = *(baseInput + 11);
            uchar4 ax = make_uchar4(axx, axy, axz, axw);
            uchar4 ay = make_uchar4(ayx, ayy, ayz, ayw);
            uchar4 az = make_uchar4(azx, azy, azz, azw);
            uchar4 b = make_uchar4(*(baseInput + 12), *(baseInput + 13), *(baseInput + 14), *(baseInput + 15));
            uchar4 c = make_uchar4(*(baseInput + 16), *(baseInput + 17), *(baseInput + 18), *(baseInput + 19));
            // uint b = *(reinterpret_cast<const uint*>(Input + 10 * i) + (3 * i) + 3);
            // uint c = *(reinterpret_cast<const uint*>(Input + 10 * i) + (3 * i) + 4);
            uint4 inp = make_uint4(*reinterpret_cast<uint*>(&ax), *reinterpret_cast<uint*>(&ay), *reinterpret_cast<uint*>(&az), *reinterpret_cast<uint*>(&b));
            // Now generate
            // printf("[%u]\n", tidx);
            ulong3 res1;
            ulong3 res2;
            uint base_idx = start_index + 6 * i;
            uint3 index1 = make_uint3(base_idx + 0, base_idx + 1, base_idx + 2);
            uint3 index2 = make_uint3(base_idx + 3, base_idx + 4, base_idx + 5);
            res1.x = get_octet2(inp, *reinterpret_cast<uint*>(&c), 1).x;
            res1.y = get_octet2(inp, *reinterpret_cast<uint*>(&c), 2).x;
            res1.z = get_octet2(inp, *reinterpret_cast<uint*>(&c), 4).x;
            //res1.w = get_octet(inp, c, 5).x;

            res2.x = get_octet2(inp, *reinterpret_cast<uint*>(&c), 8).x;
            res2.y = get_octet2(inp, *reinterpret_cast<uint*>(&c), 9).x;
            res2.z = get_octet2(inp, *reinterpret_cast<uint*>(&c), 11).x;
            //res2.w = get_octet(inp, c, 11).x;

            *(reinterpret_cast<ulong3*>(Output_kmers) + i * 2) = res1;
            *(reinterpret_cast<ulong3*>(Output_kmers) + i * 2 + 1) = res2;
            *(reinterpret_cast<uint3*>(Output_index) + i * 2) = index1;
            *(reinterpret_cast<uint3*>(Output_index) + i * 2 + 1) = index2;
        }
        // Maybe we could have a nicer coalescing load/safe pattern with shared memory?
    }

    __global__ void produce_index_kmer_tuples_12_64(const char* Input, sa_index_t start_index, sa_index_t* Output_index,
        ulong1* Output_kmers, size_t N) {
        assert(N % 12 == 0); // No one wants to deal with the "tails" here, we just write some more and don't care.
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i = tidx; i < N / 12; i += blockDim.x * gridDim.x) {
            uint3 a = *(reinterpret_cast<const uint3*>(Input) + i);
            uint b = *(reinterpret_cast<const uint*>(Input) + (3 * i) + 3);
            uint c = *(reinterpret_cast<const uint*>(Input) + (3 * i) + 4);
            uint4 inp = make_uint4(a.x, a.y, a.z, b);
            // Now generate
            ulong4 res1;
            ulong4 res2;
            uint base_idx = start_index + 8 * i;
            uint4 index1 = make_uint4(base_idx + 0, base_idx + 1, base_idx + 2, base_idx + 3);
            uint4 index2 = make_uint4(base_idx + 4, base_idx + 5, base_idx + 6, base_idx + 7);
            res1.x = get_octet(inp, c, 1).x;
            res1.y = get_octet(inp, c, 2).x;
            res1.z = get_octet(inp, c, 4).x;
            res1.w = get_octet(inp, c, 5).x;

            res2.x = get_octet(inp, c, 7).x;
            res2.y = get_octet(inp, c, 8).x;
            res2.z = get_octet(inp, c, 10).x;
            res2.w = get_octet(inp, c, 11).x;

            *(reinterpret_cast<ulong4*>(Output_kmers) + i * 2) = res1;
            *(reinterpret_cast<ulong4*>(Output_kmers) + i * 2 + 1) = res2;
            *(reinterpret_cast<uint4*>(Output_index) + i * 2) = index1;
            *(reinterpret_cast<uint4*>(Output_index) + i * 2 + 1) = index2;
        }
        // Maybe we could have a nicer coalescing load/safe pattern with shared memory?
    }

    __global__ void fixup_last_two_12_kmers(sa_index_t* address) {
        uint last, before_last;
        before_last = *address;
        last = *(address + 1);

        before_last &= ~1;
        before_last |= 2;
        last &= ~1;
        last &= ~2;

        *address = before_last;
        *(address + 1) = last;
    }


    __global__ void fixup_last_four_12_kmers_64(ulong1* address) {

        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;

        if (tidx < 4) {
            ulong1 my_value;
            my_value = address[tidx];
            my_value.x &= ~(7ull << 13);

            my_value.x |= ((3ull - tidx)) << 13;

            address[tidx] = my_value;
        }
    }

    __global__ void fixup_last_six_12_kmers_64(ulong1* address) {

        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;

        if (tidx < 6) {
            ulong1 my_value;
            my_value = address[tidx];
            my_value.x &= ~(7ull << 13);

            my_value.x |= ((5ull - tidx)) << 12;

            address[tidx] = my_value;
        }
    }

    // Needs at least two suffixes per node.
    __global__ void prepare_S12(const sa_index_t* Isa, const unsigned char* Input,
        const sa_index_t* next_Isa, const unsigned char* next_input,
        sa_index_t node_pd_offset,
        MergeStageSuffixS12* out, size_t N) {
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i = tidx; i < N - 1; i += blockDim.x * gridDim.x) {
            uint index = node_pd_offset + i;
            unsigned char x = Input[i];
            unsigned char x_p1 = Input[i + 1];
            index = (index / 2) * 3 + index % 2 + 1; // Transform from PD 0/1 to full index.
            uint rank = Isa[i] - 1;
            uint rank_p;
            if (index % 3 == 1)
                rank_p = Isa[i + 1];
            else
                rank_p = i < N - 2 ? Isa[i + 2] : ((next_Isa) ? *next_Isa : 0);

            MergeStageSuffixS12 suffix_info;
            suffix_info.index = index;
            suffix_info.chars[0] = x;
            suffix_info.chars[1] = x_p1;
            suffix_info.own_rank = rank;
            suffix_info.rank_p1p2 = rank_p;
            suffix_info._padding[0] = suffix_info._padding[1] = 0;
            out[i] = suffix_info;
        }
        if (tidx == blockDim.x * gridDim.x - 1) {
            uint i = N - 1;
            uint index = node_pd_offset + i;
            uint rank = Isa[i] - 1;
            uint rank_p;
            unsigned char x = Input[i];
            unsigned char x_p1 = (next_input) ? *next_input : 0;
            index = (index / 2) * 3 + index % 2 + 1; // Transform from PD 0/1 to full index.

            if (index % 3 == 1)
                rank_p = next_Isa ? next_Isa[0] : 0;
            else
                rank_p = next_Isa ? next_Isa[1] : 0;
            MergeStageSuffixS12 suffix_info;
            suffix_info.index = index;
            suffix_info.own_rank = rank;
            suffix_info.rank_p1p2 = rank_p;
            suffix_info.chars[0] = x;
            suffix_info.chars[1] = x_p1;
            suffix_info._padding[0] = suffix_info._padding[1] = 0;
            out[i] = suffix_info;
        }
    }

    // Needs at least two suffixes per node.
    __global__ void prepare_S12_ind(const sa_index_t* indices, const sa_index_t* Isa, const unsigned char* Input,
        const sa_index_t* next_Isa, const unsigned char* next_Input,
        sa_index_t offset, size_t num_chars,
        MergeStageSuffixS12* out, size_t N) {
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i_ = tidx; i_ < N; i_ += blockDim.x * gridDim.x) {
            uint i = indices[i_];
            uint index = (i / 2) * 3 + i % 2 + 1; // Transform from PD 0/1 to full index, assume offset % 3 == 0.
            unsigned char x = Input[index];
            unsigned char x_p1 = index < num_chars ? Input[index + 1] : (next_Input ? *next_Input : 0);
            uint rank = Isa[i] - 1;
            uint rank_p;

            if (i < N - 1)
                rank_p = Isa[i + 1];
            else
                rank_p = next_Isa ? next_Isa[0] : 0;

            MergeStageSuffixS12 suffix_info;
            suffix_info.index = offset + index;
            suffix_info.own_rank = rank;
            suffix_info.rank_p1p2 = rank_p;
            suffix_info.chars[0] = x;
            suffix_info.chars[1] = x_p1;
            suffix_info._padding[0] = suffix_info._padding[1] = 0;
            out[i_] = suffix_info;
        }
    }

    __global__ void prepare_SK_ind_kv(const sa_index_t* indices, const sa_index_t* Isa, const unsigned char* Input,
        const sa_index_t* next_Isa, const unsigned char* next_Input,
        sa_index_t offset, size_t num_chars,
        MergeSuffixes* out_keys, size_t N, D_DCX* dcx)
    {
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i_ = tidx; i_ < N; i_ += blockDim.x * gridDim.x) {
            uint i = i_;
            uint index = (i / DCX::C) * DCX::X + dcx->samplePosition[i % DCX::C];
            MergeSuffixes sk;
            sk.index = index + offset;
            uint nexInputIndex = 0;
            for (uint x = 0; x < DCX::X; x++) {
                if (index + x < num_chars) {
                    sk.prefix[x] = Input[index + x];
                }
                else {
                    sk.prefix[x] = (next_Input ? next_Input[nexInputIndex] : 0);
                    nexInputIndex++;
                }
            }
            uint nexIsaIndex = 0;
            for (uint c = 0; c < DCX::C; c++) {
                if ((i + c) < N) {

                    sk.ranks[c] = Isa[i + c] + 1;
                }
                else {
                    if (next_Isa) {
                        sk.ranks[c] = next_Isa[nexIsaIndex] + 1;
                    }
                    else {
                        sk.ranks[c] = index < num_chars - 1 ? 1 : 0;
                    }
                    nexIsaIndex++;
                }
            }

            out_keys[i_] = sk;
        }
    }

    __global__ void prepare_S12_ind_kv(const sa_index_t* indices, const sa_index_t* Isa, const unsigned char* Input,
        const sa_index_t* next_Isa, const unsigned char* next_Input,
        sa_index_t offset, size_t num_chars, size_t pd_per_gpu,
        MergeStageSuffixS12HalfKey* out_keys, MergeStageSuffixS12HalfValue* out_values, size_t N)
    {
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i_ = tidx; i_ < N; i_ += blockDim.x * gridDim.x) {
            uint i = indices[i_];
            uint index = (i / 2) * 3 + i % 3 + 1; // Transform from PD 0/1 to full index, assume offset % 3 == 0.
            unsigned char x = Input[index];
            unsigned char x_p1 = index < num_chars ? Input[index + 1] : (next_Input ? *next_Input : 0);
            uint rank = Isa[i] - 1;
            uint rank_p;

            if (i < N - 1)
                rank_p = Isa[i + 1] + 1;
            else {
                if (next_Isa)
                    rank_p = next_Isa[0] + 1;
                else
                    rank_p = index < num_chars - 1 ? 1 : 0;
            }

            MergeStageSuffixS12HalfKey suffix_key;
            MergeStageSuffixS12HalfValue suffix_value;
            suffix_key.own_rank = rank % pd_per_gpu;
            suffix_key.index = offset + index;
            suffix_value.rank_p1p2 = rank_p;
            suffix_value.chars[0] = x;
            suffix_value.chars[1] = x_p1;
            suffix_value._padding[0] = suffix_value._padding[1] = 0;
            out_keys[i_] = suffix_key;
            out_values[i_] = suffix_value;
        }
    }


    __global__ void write_indices(sa_index_t* Out, size_t N) {
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i = tidx; i < N; i += blockDim.x * gridDim.x) {
            Out[i] = i;
        }
    }

    __global__ void write_S12_back(const MergeStageSuffixS12* inp, MergeStageSuffix* outp, size_t base_offset, size_t N)
    {
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i = tidx; i < N; i += blockDim.x * gridDim.x) {
            MergeStageSuffixS12 suff = inp[i];
            sa_index_t target_index = suff.own_rank - base_offset;
            MergeStageSuffix out;
            out.index = suff.index;
            out.rank_p1 = suff.rank_p1p2;
            out.rank_p2 = suff.rank_p1p2;
            out.chars[0] = suff.chars[0];
            out.chars[1] = suff.chars[1];
            out._padding[0] = suff._padding[0];
            out._padding[1] = suff._padding[1];
            outp[target_index] = out;
        }
    }

    __global__ void prepare_non_sample(const sa_index_t* Isa, const unsigned char* Input,
        const sa_index_t* next_Isa, const unsigned char* next_Input,
        sa_index_t offset, size_t num_chars, size_t isa_size,
        MergeSuffixes* out_keys, size_t N, sa_index_t non_sample_pos, sa_index_t f)
    {
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i = tidx; i < N; i += blockDim.x * gridDim.x) {
            uint index = non_sample_pos + i * DCX::X;
            // uint index = (i / (DCX::X - DCX::C)) * DCX::X + dcx->nextNonSample[i % (DCX::X - DCX::C)];
            MergeSuffixes sv;
            sv.index = index + offset;
            uint nexInputIndex = 0;

            for (uint x = 0; x < DCX::X; x++) {
                if (index + x < num_chars) {
                    sv.prefix[x] = Input[index + x];
                }
                else {
                    sv.prefix[x] = (next_Input ? next_Input[nexInputIndex] : 0);
                    nexInputIndex++;
                }
            }
            uint nexIsaIndex = 0;
            uint starting_isa_index = i * DCX::C + f;
            for (uint c = 0; c < DCX::C; c++) {
                if (starting_isa_index < isa_size) {
                    sv.ranks[c] = Isa[starting_isa_index++] + 1;
                }
                else {
                    if (next_Isa) {
                        sv.ranks[c] = next_Isa[nexIsaIndex] + 1;
                    }
                    else {
                        sv.ranks[c] = index < num_chars - 1 ? 1 : 0;
                    }
                    nexIsaIndex++;
                }
            }
            out_keys[i] = sv;
        }
    }

    __global__ void prepare_S0(const sa_index_t* Isa, const unsigned char* Input,
        sa_index_t node_offset,
        size_t no_chars,
        size_t Isa_size,
        bool last,
        MergeStageSuffixS0HalfKey* Out_key,
        MergeStageSuffixS0HalfValue* Out_value,
        size_t N) {
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i = tidx; i < N; i += blockDim.x * gridDim.x) {
            sa_index_t index = i * 3;
            unsigned char x = Input[index];
            unsigned char x_p1 = index < no_chars ? Input[index + 1] : 0;
            sa_index_t rank_p1 = 0, rank_p2 = 0;

            if (i * 2 < Isa_size)
                rank_p1 = Isa[2 * i] + 1;
            else if (last) {
                rank_p1 = (2 * i < Isa_size - 1) ? 1 : 0;
            }

            if (i * 2 + 1 < Isa_size) {
                rank_p2 = Isa[2 * i + 1] + 1;
            }
            else if (last) {
                rank_p2 = (2 * i + 1 < Isa_size - 1) ? 1 : 0;
            }

            MergeStageSuffixS0HalfValue value;
            value.index = node_offset + index;
            value.rank_p2 = rank_p2;

            MergeStageSuffixS0HalfKey key;
            key.chars[0] = x;
            key.chars[1] = x_p1;
            key.rank_p1 = rank_p1;
            key._padding[0] = key._padding[1] = 0;

            Out_key[i] = key;
            Out_value[i] = value;
        }
    }

    __global__ void combine_S0_kv(const MergeStageSuffixS0HalfKey* Keys,
        const MergeStageSuffixS0HalfValue* Values,
        MergeStageSuffix* Out, size_t N) {
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i = tidx; i < N; i += blockDim.x * gridDim.x) {
            MergeStageSuffixS0HalfKey k = Keys[i];
            MergeStageSuffixS0HalfValue v = Values[i];
            MergeStageSuffix s;
            s.chars[0] = k.chars[0];
            s.rank_p1 = k.rank_p1;
            s.chars[1] = k.chars[1];
            s._padding[0] = k._padding[0];
            s._padding[1] = k._padding[1];
            s.rank_p2 = v.rank_p2;
            s.index = v.index;
            Out[i] = s;
        }
    }

    __global__ void combine_S12_kv(const MergeStageSuffixS12HalfKey* Keys,
        const MergeStageSuffixS12HalfValue* Values,
        MergeStageSuffix* Out, size_t N)
    {
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i = tidx; i < N; i += blockDim.x * gridDim.x) {
            MergeStageSuffixS12HalfKey k = Keys[i];
            MergeStageSuffixS12HalfValue v = Values[i];
            MergeStageSuffix s;
            s.chars[0] = v.chars[0];
            s.rank_p1 = v.rank_p1p2;
            s.chars[1] = v.chars[1];
            s._padding[0] = v._padding[0];
            s._padding[1] = v._padding[1];
            s.rank_p2 = v.rank_p1p2;
            s.index = k.index;
            Out[i] = s;
        }
    }

    // testing
    __global__ void combine_S12_kv_non_coalesced(const MergeStageSuffixS12HalfKey* Keys,
        const MergeStageSuffixS12HalfValue* Values,
        MergeStageSuffix* outp, size_t N)
    {
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i = tidx; i < N; i += blockDim.x * gridDim.x) {
            MergeStageSuffixS12HalfKey k = Keys[i];
            MergeStageSuffixS12HalfValue v = Values[i];
            sa_index_t target_index = k.own_rank;
            MergeStageSuffix out;
            out.index = k.index;
            out.rank_p1 = v.rank_p1p2;
            out.rank_p2 = v.rank_p1p2;
            out.chars[0] = v.chars[0];
            out.chars[1] = v.chars[1];
            out._padding[0] = v._padding[0];
            out._padding[1] = v._padding[1];
            outp[target_index] = out;
        }
    }

    __global__ void from_merge_suffix_to_index(const MergeStageSuffix* Merge_suffixes, sa_index_t* Out, size_t N)
    {
        uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint i = tidx; i < N; i += blockDim.x * gridDim.x) {
            MergeStageSuffix suff = Merge_suffixes[i];
            Out[i] = suff.index;
        }
    }





    __global__ void produce_sk_tuples(const unsigned char* Input, sa_index_t* ranks, MergeSuffixes* output) {
        //     // corresponds with the index
        //     uint tidx = blockIdx.x * blockDim.x + threadIdx.x;

        //     output[tidx].index = tidx;

        //     output[tidx].ranks[0] = ranks[tidx];
        //     output[tidx].ranks[1] = ranks[tidx + 1];
        //     output[tidx].ranks[2] = ranks[tidx + 2];
        //     output[tidx].ranks[3] = ranks[tidx + 3];
        //     output[tidx].ranks[4] = ranks[tidx + 4];
        //     output[tidx].ranks[5] = ranks[tidx + 5];
        //     output[tidx].ranks[6] = ranks[tidx + 6];
        //     output[tidx].prefix[0] = Input[tidx];
        //     output[tidx].prefix[1] = Input[tidx + 1];
        //     output[tidx].prefix[2] = Input[tidx + 2];
        //     output[tidx].prefix[3] = Input[tidx + 3];
        //     output[tidx].prefix[4] = Input[tidx + 4];
        //     output[tidx].prefix[5] = Input[tidx + 5];
        //     output[tidx].prefix[6] = Input[tidx + 6];
        //     // printf("[%u] %u, ranks: %u, xPrefix: %c\n", tidx, i, output[tidx].ranks[i], output[tidx].xPrefix[i]);


    }

    template<uint MAX_GPUS>
    __global__ void sampleSort(MergeSuffixes* sk, MergeSuffixes* output, MergeSuffixes splitter, size_t offset, size_t splitter_size, DC7Comparator comp) {
        const uint thidx = blockDim.x * blockIdx.x + threadIdx.x;

        __shared__ uint splitter_index[MAX_GPUS];
        if (thidx == 0) {
            for (size_t i = 0; i < splitter_size; i++)
            {
                splitter_index[i] = 0;
            }
        }
        __syncthreads();

        uint k = 0;

        for (size_t i = 0; i < splitter_size; i++)
        {
            if (comp(sk[thidx], splitter)) {
                break;
            }
            else {
                k++;
            }
        }
        *(output + splitter_index[k] + k * offset) = sk[thidx];

    }

    __global__ void writeSamples(size_t* sample_pos, MergeSuffixes* data, MergeSuffixes* out) {
        const uint thidx = blockDim.x * blockIdx.x + threadIdx.x;
        out[thidx] = data[sample_pos[thidx]];
    }

    __global__ void selectSplitter(MergeSuffixes* samples, size_t sample_count) {
        const uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
        samples[tidx] = samples[sample_count * (tidx + 1)];
    }
}
