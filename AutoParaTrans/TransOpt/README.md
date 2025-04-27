# CUDA-Parallelized TransOpt

## Overview

This package is a CUDA-parallelized version of the TransOpt code for calculating electrical transport properties. TransOpt allows VASP users to calculate Seebeck coefficients, electrical conductivities, and electronic thermal conductivities using either the momentum matrix method (Method 1) or the derivative method (Method 2, similar to BoltzTrap).

Our implementation maintains all the functionality of the original TransOpt while significantly improving performance through CUDA GPU acceleration.

## Different Versions

Multiple optimization approaches are available, each addressing different aspects of GPU parallelization:

| File | Description |
|------|-------------|
| `TransOpt_cuda_based.cuf` | Base CUDA implementation |
| `TransOpt_cuda_sharedMemory.cuf` | Implementation using GPU shared memory |
| `SharedMemory_v2.cuf` | Improved shared memory implementation with additional optimizations |
| `TransOpt_nv1.cuf` | Initial CUDA implementation |
| `TransOpt_nv2.cuf` | Version with loop unrolling optimizations |
| `TransOpt_nv3.cuf` | Multi-GPU implementation (builds on previous optimizations) |
| `TransOpt_nv4.cuf` | Multi-GPU implementation without shared memory |
| `TransOpt_nv5.cuf` | Single GPU with asynchronous data transfer |
| `TransOpt_nv6.cuf` | Implementation using pinned memory + asynchronous data transfer |
| `TransOpt_nv7.cuf` | Implementation using pinned memory without asynchronous transfer |

## Key Optimizations

1. **CUDA Parallelization**: Core computations are offloaded to GPU for massive parallelism
2. **Shared Memory**: Utilizes GPU shared memory to reduce global memory access latency
3. **Loop Unrolling**: Improves instruction-level parallelism
4. **Multi-GPU Support**: Distributes computation across multiple GPUs for further speedup
5. **Asynchronous Data Transfer**: Overlaps computation and data transfer
6. **Pinned Memory**: Improves host-device transfer speeds

## Compilation Instructions

To compile the CUDA-parallelized versions of TransOpt, you'll need:
- NVIDIA CUDA Toolkit (10.0 or later recommended)
- NVIDIA HPC SDK (formerly PGI Compiler) or similar CUDA Fortran compiler
- Compatible NVIDIA GPU(s)

### Basic Compilation

```bash
# For basic compilation using NVIDIA HPC SDK:
nvfortran -Mcuda -O3 TransOpt_nv2.cuf -o TransOpt_nv2

# For specific optimizations:
nvfortran -Mcuda=pinned -O3 TransOpt_nv6.cuf -o TransOpt_nv6

# For multi-GPU version:
nvfortran -Mcuda=mpgi -O3 TransOpt_nv3.cuf -o TransOpt_nv3
```

### Creating a Compilation Script

You can create a compilation script (e.g., `compile.sh`):

```bash
#!/bin/bash

# Compile all versions
nvfortran -Mcuda -O3 TransOpt_cuda_based.cuf -o TransOpt_cuda
nvfortran -Mcuda -O3 TransOpt_cuda_sharedMemory.cuf -o TransOpt_sharedMem
nvfortran -Mcuda -O3 SharedMemory_v2.cuf -o TransOpt_sharedMem_v2
nvfortran -Mcuda -O3 TransOpt_nv1.cuf -o TransOpt_nv1
nvfortran -Mcuda -O3 TransOpt_nv2.cuf -o TransOpt_nv2
nvfortran -Mcuda=mpgi -O3 TransOpt_nv3.cuf -o TransOpt_nv3
nvfortran -Mcuda=mpgi -O3 TransOpt_nv4.cuf -o TransOpt_nv4
nvfortran -Mcuda -O3 TransOpt_nv5.cuf -o TransOpt_nv5
nvfortran -Mcuda=pinned -O3 TransOpt_nv6.cuf -o TransOpt_nv6
nvfortran -Mcuda=pinned -O3 TransOpt_nv7.cuf -o TransOpt_nv7

echo "Compilation complete"
```

Make it executable and run:
```bash
chmod +x compile.sh
./compile.sh
```

## Usage

The CUDA-parallelized versions use the same input files and produce the same output files as the original TransOpt. The only difference is the significantly improved performance.

```bash
# Run the desired version:
./TransOpt_nv2

# For multi-GPU versions, you can specify GPU devices:
CUDA_VISIBLE_DEVICES=0,1 ./TransOpt_nv3

# For performance comparison, you can time the execution:
time ./TransOpt_nv2
```

### Required Input Files

Same as original TransOpt:
1. POSCAR
2. EIGENVAL
3. SYMMETRY (from modified VASP, vasp.*.symm version)
4. Group velocity information:
   - GROUPVEC for Method 1 (from modified VASP, vasp.*.vk version)
   - Or GVEC for Method 2 (from derivation code)
5. TransOpt.input (For TransOpt v2.0 and later)

### Output Files

Same as original TransOpt:
1. CRTA-trace-e.txt (constant relaxation time approximation)
2. RTA-trace-e.txt (if the relaxation time is calculated)
3. CRTA-tensor-e.txt (constant relaxation time approximation with full tensor)
4. RTA-tensor-e.txt (full tensor with calculated relaxation time)

## Performance Notes

- For most systems, `TransOpt_nv2.cuf` (with loop unrolling) offers a good balance of performance and compatibility
- For very large systems, consider the multi-GPU versions (`TransOpt_nv3.cuf` or `TransOpt_nv4.cuf`)
- If your system has fast NVMe storage, the asynchronous versions (`TransOpt_nv5.cuf` and `TransOpt_nv6.cuf`) may provide additional benefits
- The pinned memory versions (`TransOpt_nv6.cuf` and `TransOpt_nv7.cuf`) typically offer the best host-to-device transfer performance

## References

If you use this CUDA-parallelized version of TransOpt, please cite the original TransOpt references as listed in the original documentation, plus any additional publication about this CUDA implementation.

Original TransOpt references:
1. Journal of Electronic Materials 38, 1397 (2009)
2. Journal of American Chemical Society 140, 10785-10793 (2018)
   - Computational Materials Science 186, 110074 (2021) (The code TransOpt itself, also CEPCA)
3. Nanoscale 11, 10828 (2019) (TransOpt interfaced with Quantum Espresso)
4. Journal of Materiomics DOI: 10.1016/J.JMAT.2022.05.003 (The introduction of ionized impurity scattering)