#!/bin/bash

cd libcubwt-1.0.0
spack env activate nvidia
nvcc main.cu libcubwt.cu -o Test && ./Test input.txt output