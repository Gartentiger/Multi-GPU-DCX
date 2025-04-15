// Assembled from https://github.com/owensgroup/GpuMultisplit/blob/master/src/main/main_wms.cu

// Original license header below.
/*
GpuMultisplit is the proprietary property of The Regents of the University of
California ("The Regents") and is copyright © 2016 The Regents of the University
of California, Davis campus. All Rights Reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted by nonprofit educational or research institutions for
noncommercial use only, provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.
* The name or other trademarks of The Regents may not be used to endorse or
promote products derived from this software without specific prior written
permission.

The end-user understands that the program was developed for research purposes
and is advised not to rely exclusively on the program for any reason.

THE SOFTWARE PROVIDED IS ON AN "AS IS" BASIS, AND THE REGENTS HAVE NO OBLIGATION
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS. THE
REGENTS SPECIFICALLY DISCLAIM ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING BUT
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE REGENTS BE LIABLE
TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, EXEMPLARY OR
CONSEQUENTIAL DAMAGES, INCLUDING BUT NOT LIMITED TO  PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES, LOSS OF USE, DATA OR PROFITS, OR BUSINESS INTERRUPTION,
HOWEVER CAUSED AND UNDER ANY THEORY OF LIABILITY WHETHER IN CONTRACT, STRICT
LIABILITY OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

If you do not agree to these terms, do not download or use the software.  This
license may be modified only in a writing signed by authorized signatory of both
parties.

For license information please contact copyright@ucdavis.edu re T11-005.
*/
#ifndef _DISPATCH_MULTISPLIT_CUH_
#define _DISPATCH_MULTISPLIT_CUH_

#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cub/cub.cuh>
#include <functional>

#include "api/wms_api.h"
#include "config/config_wms.h"

#include "kernels/wms/wms_postscan.cuh"
#include "kernels/wms/wms_postscan_pairs.cuh"
#include "kernels/wms/wms_prescan.cuh"

#include <climits>
#include <type_traits>

constexpr uint32_t next_power_of_two32 (uint32_t x) { return 1u<<(sizeof(uint32_t) * 8 - __builtin_clz(x)); }

__global__ void export_offsets(const uint32_t* d_histogram, uint32_t* out, uint32_t offset_offset, uint32_t N) {
    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx > 0 && tidx < N) {
        out[tidx] = d_histogram[offset_offset * tidx];
    }
    else if (tidx == 0) {
        out[0] = 0;
    }
}

template <uint NUM_BUCKETS>
class DispatchMultisplit {
    public:

        template<class BucketIdentifier>
        static void multisplit(const uint32_t* d_key_in,  uint32_t* d_key_out,
                               uint32_t* d_offsets_out, BucketIdentifier bucket_identifier,
                               size_t num_elements, QDAllocator& temp_allocator, cudaStream_t stream) {
            DispatchMultisplit ms(temp_allocator, num_elements, false);
            ms.dispatch_k_only(d_key_in, d_key_out, d_offsets_out, bucket_identifier, stream);
        }

        template<class BucketIdentifier>
        static void multisplit_kv(const uint32_t* d_key_in, uint32_t* d_key_out,
                                  const uint32_t* d_value_in, uint32_t* d_value_out,
                                  uint32_t* d_offsets_out, BucketIdentifier bucket_identifier,
                                  size_t num_elements, QDAllocator& temp_allocator, cudaStream_t stream) {
            DispatchMultisplit ms(temp_allocator, num_elements, true);
            ms.dispatch_kv(d_key_in, d_key_out, d_value_in, d_value_out, d_offsets_out,
                           bucket_identifier, stream);
        }

        static constexpr uint32_t effective_num_buckets() {
            return num_buckets;
        }

    private:
        uint32_t num_elements;

        static const uint32_t num_buckets = next_power_of_two32(NUM_BUCKETS);
        uint32_t size_sub_prob;
        uint32_t size_block;
        uint32_t num_sub_prob_per_block, num_sub_prob;
        uint32_t num_blocks;

        uint32_t* d_histogram;
        void* d_temp_storage;
        size_t temp_storage_bytes;

        DispatchMultisplit(QDAllocator& temp_allocator, size_t num_elements_,
                           bool include_values)
            : size_sub_prob(1),
              size_block(1),
              num_elements(num_elements_)
        {
            if (include_values)
                size_sub_prob = subproblem_size_wms_key_value(num_buckets, size_block);
            else
                size_sub_prob = subproblem_size_wms_key_only(num_buckets, size_block);

            assert((size_sub_prob != 1) && (size_block != 1));

            num_sub_prob_per_block = size_block / size_sub_prob;
            num_sub_prob = (num_elements + size_sub_prob - 1) / (size_sub_prob);
            num_sub_prob = (num_sub_prob + num_sub_prob_per_block - 1) / num_sub_prob_per_block *
                num_sub_prob_per_block;  // making sure block is full of subproblems,
                                         // even if some will be invalidated afterwards.
            num_blocks = (num_elements + size_block - 1) / size_block;
//            printf("n = %d, num_blocks = %d, size_sub_prob = %d, num_sub_prob = %d\n",
//                   num_elements, num_blocks, size_sub_prob, num_sub_prob);

            d_histogram = temp_allocator.get<uint32_t>(num_buckets * num_sub_prob);

            d_temp_storage = nullptr;
            temp_storage_bytes = 0;

            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                          d_histogram, d_histogram,
                                          num_buckets * num_sub_prob);
            d_temp_storage = temp_allocator.get_raw(temp_storage_bytes);
        }

        template<typename BucketIdentifier>
        void dispatch_k_only(const uint32_t* d_key_in,  uint32_t* d_key_out, uint32_t* d_offsets_out,
                             BucketIdentifier bucket_identifier,
                             cudaStream_t stream) {
            switch (num_buckets) {
              case 2:
                  multisplit2_WMS_prescan_protected<NUM_TILES_K_1, NUM_ROLLS_K_1, num_buckets, 1>
                      <<<num_blocks, 32 * NUM_WARPS_K_1, 0, stream>>>
                          (d_key_in, num_elements, d_histogram, bucket_identifier);
                break;
              case 4:
                  multisplit2_WMS_prescan_protected<NUM_TILES_K_2, NUM_ROLLS_K_2, num_buckets, 2>
                      <<<num_blocks, 32 * NUM_WARPS_K_2, 0, stream>>>
                          (d_key_in, num_elements, d_histogram, bucket_identifier);
                break;
              case 8:
                  multisplit2_WMS_prescan_protected<NUM_TILES_K_3, NUM_ROLLS_K_3, num_buckets, 3>
                      <<<num_blocks, 32 * NUM_WARPS_K_3, 0, stream>>>
                          (d_key_in, num_elements, d_histogram, bucket_identifier);
                break;
              case 16:
                  multisplit2_WMS_prescan_protected<NUM_TILES_K_4, NUM_ROLLS_K_4, num_buckets, 4>
                      <<<num_blocks, 32 * NUM_WARPS_K_4, 0, stream>>>
                          (d_key_in, num_elements, d_histogram, bucket_identifier);
                break;
              case 32:
                  multisplit2_WMS_prescan_protected<NUM_TILES_K_5, NUM_ROLLS_K_5, num_buckets, 5>
                      <<<num_blocks, 32 * NUM_WARPS_K_5, 0, stream>>>
                          (d_key_in, num_elements, d_histogram, bucket_identifier);
                break;
            }

            // printf("Histogram process finished in %.3f ms (%.3f Gkey/s)\n",
            // pre_scan_time, float(n_elements)/pre_scan_time/1000.0f);

            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                          d_histogram, d_histogram,
                                          num_buckets * num_sub_prob, stream);

            if (d_offsets_out){
                export_offsets<<<1, 32, 0, stream>>>(d_histogram, d_offsets_out, num_sub_prob, num_buckets);
            }


            switch (num_buckets) {
              case 2:
                  multisplit2_WMS_postscan_4rolls_protected<NUM_WARPS_K_1, NUM_TILES_K_1,
                          NUM_ROLLS_K_1, num_buckets, 1>
                      <<<num_blocks, 32 * NUM_WARPS_K_1, 0, stream>>>
                        (d_key_in, d_key_out, num_elements, d_histogram, bucket_identifier);
                break;
              case 4:
                  multisplit2_WMS_postscan_4rolls_protected<NUM_WARPS_K_2, NUM_TILES_K_2,
                          NUM_ROLLS_K_2, num_buckets, 2>
                      <<<num_blocks, 32 * NUM_WARPS_K_2, 0, stream>>>
                        (d_key_in, d_key_out, num_elements, d_histogram, bucket_identifier);
                break;
              case 8:
                  multisplit2_WMS_postscan_4rolls_protected<NUM_WARPS_K_3, NUM_TILES_K_3,
                          NUM_ROLLS_K_3, num_buckets, 3>
                      <<<num_blocks, 32 * NUM_WARPS_K_3, 0, stream>>>
                        (d_key_in, d_key_out, num_elements, d_histogram, bucket_identifier);

                break;
              case 16:
                  multisplit2_WMS_postscan_4rolls_protected<NUM_WARPS_K_4, NUM_TILES_K_4,
                          NUM_ROLLS_K_4, num_buckets, 4>
                      <<<num_blocks, 32 * NUM_WARPS_K_4, 0, stream>>>
                        (d_key_in, d_key_out, num_elements, d_histogram, bucket_identifier);

                break;
              case 32:
                  multisplit2_WMS_postscan_4rolls_protected<NUM_WARPS_K_5, NUM_TILES_K_5,
                          NUM_ROLLS_K_5, num_buckets, 5>
                      <<<num_blocks, 32 * NUM_WARPS_K_5, 0, stream>>>
                        (d_key_in, d_key_out, num_elements, d_histogram, bucket_identifier);
                break;
            }
        }

        template<typename BucketIdentifier>
        void dispatch_kv(const uint32_t* d_key_in, uint32_t* d_key_out, const uint32_t* d_value_in,
                       uint32_t* d_value_out, uint32_t* d_offsets_out,
                       BucketIdentifier bucket_identifier,
                       cudaStream_t stream) {
            switch (num_buckets) {
              case 2:
                  multisplit2_WMS_prescan_protected<NUM_TILES_KV_1, NUM_ROLLS_KV_1, num_buckets, 1>
                      <<<num_blocks, 32 * NUM_WARPS_KV_1, 0, stream>>>
                          (d_key_in, num_elements, d_histogram, bucket_identifier);
                break;
              case 4:
                  multisplit2_WMS_prescan_protected<NUM_TILES_KV_2, NUM_ROLLS_KV_2, num_buckets, 2>
                      <<<num_blocks, 32 * NUM_WARPS_KV_2, 0, stream>>>
                          (d_key_in, num_elements, d_histogram, bucket_identifier);
                break;
              case 8:
                  multisplit2_WMS_prescan_protected<NUM_TILES_KV_3, NUM_ROLLS_KV_3, num_buckets, 3>
                      <<<num_blocks, 32 * NUM_WARPS_KV_3, 0, stream>>>
                          (d_key_in, num_elements, d_histogram, bucket_identifier);
                break;
              case 16:
                  multisplit2_WMS_prescan_protected<NUM_TILES_KV_4, NUM_ROLLS_KV_4, num_buckets, 4>
                      <<<num_blocks, 32 * NUM_WARPS_KV_4, 0, stream>>>
                          (d_key_in, num_elements, d_histogram, bucket_identifier);
                break;
              case 32:
                  multisplit2_WMS_prescan_protected<NUM_TILES_KV_5, NUM_ROLLS_KV_5, num_buckets, 5>
                      <<<num_blocks, 32 * NUM_WARPS_KV_5, 0, stream>>>
                          (d_key_in, num_elements, d_histogram, bucket_identifier);
                break;
            }


            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                          d_histogram, d_histogram,
                                          num_buckets * num_sub_prob, stream);

            if (d_offsets_out){
                export_offsets<<<1, 32, 0, stream>>>(d_histogram, d_offsets_out, num_sub_prob, num_buckets);
            }

            // post scan stage:
            switch (num_buckets) {
              case 2:
                  multisplit2_WMS_postscan_4rolls_pairs_protected<
                      NUM_WARPS_KV_1, NUM_TILES_KV_1, NUM_ROLLS_KV_1, num_buckets, 1>
                      <<<num_blocks, 32 * NUM_WARPS_KV_1, 0, stream>>>
                          (d_key_in, d_value_in, d_key_out, d_value_out, num_elements,
                           d_histogram, bucket_identifier);
                break;
              case 4:
                  multisplit2_WMS_postscan_4rolls_pairs_protected<
                      NUM_WARPS_KV_2, NUM_TILES_KV_2, NUM_ROLLS_KV_2, num_buckets, 2>
                      <<<num_blocks, 32 * NUM_WARPS_KV_2, 0, stream>>>
                          (d_key_in, d_value_in, d_key_out, d_value_out, num_elements,
                           d_histogram, bucket_identifier);
                break;
              case 8:
                  multisplit2_WMS_postscan_4rolls_pairs_protected<
                      NUM_WARPS_KV_3, NUM_TILES_KV_3, NUM_ROLLS_KV_3, num_buckets, 3>
                      <<<num_blocks, 32 * NUM_WARPS_KV_3, 0, stream>>>
                          (d_key_in, d_value_in, d_key_out, d_value_out, num_elements,
                           d_histogram, bucket_identifier);
                break;
              case 16:
                  multisplit2_WMS_postscan_4rolls_pairs_protected<
                      NUM_WARPS_KV_4, NUM_TILES_KV_4, NUM_ROLLS_KV_4, num_buckets, 4>
                      <<<num_blocks, 32 * NUM_WARPS_KV_4, 0, stream>>>
                          (d_key_in, d_value_in, d_key_out, d_value_out, num_elements,
                           d_histogram, bucket_identifier);
                break;
              case 32:
                  multisplit2_WMS_postscan_4rolls_pairs_protected<
                      NUM_WARPS_KV_5, NUM_TILES_KV_5, NUM_ROLLS_KV_5, num_buckets, 5>
                      <<<num_blocks, 32 * NUM_WARPS_KV_5, 0, stream>>>
                          (d_key_in, d_value_in, d_key_out, d_value_out, num_elements,
                           d_histogram, bucket_identifier);
                break;
            }
        }


};

#endif // _DISPATCH_MULTISPLIT_CUH_
