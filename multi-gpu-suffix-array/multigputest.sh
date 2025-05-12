#!/bin/bash
#SBATCH -J TestMultiGPU
#SBATCH --partition=dev_accelerated
#SBATCH --get-user-env
#SBATCH --time=10
#SBATCH --nodes="1"
#SBATCH --mem="10240"

cd build 
./suffix_array output.json dblp.xml dna english pitches proteins sources