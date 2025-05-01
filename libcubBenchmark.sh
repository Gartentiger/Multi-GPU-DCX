#!/bin/bash
#SBATCH -J Test
#SBATCH --partition=feigenbaum
#SBATCH --get-user-env
#SBATCH --time=5

spack env activate nvidia

cd libcubwt-1.0.0

nvcc main.cu libcubwt.cu -o Test
./Test input output