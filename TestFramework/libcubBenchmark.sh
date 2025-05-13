#!/bin/bash
#SBATCH -J BenchmarkLibcub
#SBATCH --partition=feigenbaum
#SBATCH --get-user-env
#SBATCH --time=20

spack env activate nvidia

./BenchmarkLibcub output.json ../../../../../share/instances/text/trec-text.terms