Multi-GPU Suffix Array Construction
=====

For implementation details and benchmarks see our paper [Suffix Array Construction on Multi-GPU Systems](https://doi.org/10.1145/3307681.3325961).


Installation
-----
To compile, you need CMake 3.8 or greater and CUDA on the path.

 1. Create a build folder, preferably outside the source tree.
 2. cmake ../suffix-array/
 3. Optional: ccmake ../suffix-array/ (toggle settings, hit 'c', and 'g')
 4. make -j8

The compile option DGX1\_TOPOLOGY should be toggled reflecting the setting of NUM\_GPUS in suffix\_array.cu.

With CUDA 10, there are many deprecation warnings caused by the warp intrinsics the multisplit by Ashkiani et al. uses (without _sync), 
however, there have been no errors so far. Another multisplit implementation could be used instead.

Notes
-----
Inputs up to a size of 2^32-2 characters can theoretically be sorted; on the DGX-1 (16GB), about 3550 MB should work, 3520~MB have been successfully processed.

There is a bug with the merge-copy-detour-heuristics for worst-case inputs; these can be turned off by uncommenting the lines 305-334 of merge\_copy\_detour\_guide.hpp which may cost performance.

Some refactoring would be needed. This was written on limited time.

[distrib-merge](src/distrib_merge) means merging 2 distributed arrays that each are globally sorted.

[remerge](src/remerge) means merging multiple ranges of one distributed array, each consisting of several per-GPU locally sorted ranges, in parallel.

[gossip](src/gossip) contains an old version of our [multi-GPU communication library gossip](https://github.com/funatiq/gossip/) and has since been vastly enhanced.

[multisplit](src/multiplit) contains the [multisplit by Ashkiani et al.](https://github.com/owensgroup/GpuMultisplit) with a wrapper class (dispatch_multisplit.cuh). These files are governed by a separate license.

[deps](deps) contains the dependencies [ModernGPU](https://github.com/moderngpu/moderngpu/) and [CUB](https://nvlabs.github.io/cub/). These files are governed by separate licenses.

For the files not governed by a separate license, the license contained in [LICENSE.txt](LICENSE.txt) applies.
