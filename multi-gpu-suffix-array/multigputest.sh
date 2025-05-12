#!/bin/bash
#SBATCH -J TestMultiGPU
#SBATCH --partition=dev_accelerated
#SBATCH --get-user-env
#SBATCH --time=10
#SBATCH --nodes="1"
#SBATCH --mem="4096"
#SBATCH --gres=gpu:4

module load compiler/gnu/12

cd build 
./suffix_array output.json dblp.xml dna pitches sources
python ../../TestFramework/plot.py output.json