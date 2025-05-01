#!/bin/bash
#SBATCH -J Test
#SBATCH --partition=feigenbaum
#SBATCH --get-user-env
#SBATCH --time=20

spack env activate nvidia

cd libcubwt-1.0.0
mkdir build
cd build
cmake ..
make -j8
./BenchmarkLibcub input output