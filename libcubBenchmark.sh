#!/bin/bash
#SBATCH -J BenchmarkLibcub
#SBATCH --partition=feigenbaum
#SBATCH --get-user-env
#SBATCH --time=20

spack env activate nvidia

cd libcubwt-1.0.0
mkdir build
cd build 
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
./BenchmarkLibcub ../../../../../share/instances/text/trec-text.terms output