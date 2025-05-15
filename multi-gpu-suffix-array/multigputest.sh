#!/bin/bash
#SBATCH -J TestMultiGPU
#SBATCH --partition=dev_accelerated
#SBATCH --get-user-env
#SBATCH --time=20
#SBATCH --nodes="1"
#SBATCH --mem="118000"
#SBATCH --gres=gpu:4

module load compiler/gnu/12

cd build
#./suffix_array output.json ../../TestData/dblp.xml ../../TestData/dna ../../TestData/pitches ../../TestData/sources
#./suffix_array output.json ../../TestData/manzini/chr22.dna ../../TestData/manzini/etext99 ../../TestData/manzini/gcc-$#./suffix_array outputGene.json ../../TestData/Genes/GCA_000003625.1_OryCun2.0_genomic.fna
#./suffix_array outputGene.json outputDataGorgor ../../TestData/Genes/GCA_000151905.3_gorGor4_genomic.fna
#./suffix_array outputGene.json outputDataMmul ../../TestData/Genes/GCA_000772875.3_Mmul_8.0.1_genomic.fna
#./suffix_array outputGene.json outputDataPhaCin ../../TestData/Genes/GCA_002099425.1_phaCin_unsw_v4.1_genomic.fna
#./suffix_array outputGene.json outputDataEquCab ../../TestData/Genes/GCA_002863925.1_EquCab3.0_genomic.fna
./suffix_array outputGene.json outputDataGrCh38 ../../TestData/Genes/GCF_000001405.40_GRCh38.p14_genomic.fna