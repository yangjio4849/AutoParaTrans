# CUDA-Parallelized WP3_phonopy

## Overview

This package is a CUDA-parallelized version of the WP3_phonopy program that calculates weighted scattering phase space and related properties using second-order force constants. The original WP3_phonopy program processes three-phonon scattering in the full phase space without considering symmetry.

Our CUDA implementation significantly accelerates the computations by leveraging GPU parallel processing while maintaining all the functionality of the original code.

## Features

- Efficient calculation of three-phonon weighted scattering phase space
- Full phase space treatment without symmetry constraints
- GPU acceleration for computationally intensive tasks
- Support for various analysis scenarios:
  1. Total scattering phase space calculation
  2. Frequency adjustment analysis
  3. Mode-specific three-phonon process investigation

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit
- NVIDIA HPC SDK or compatible CUDA Fortran compiler
- Phonopy installation for mesh file generation

## Compilation

To compile the CUDA-parallelized version:

```bash
nvfortran -Mcuda -qopenmp -o WP3_phonopy_cuda WP3_phonopy_cuda.cuf
```

## Input Files

The program requires:

1. `mesh.yaml` - Generated using Phonopy
   - You can create this file by running: `phonopy --nomeshsym mesh.conf`
   - The `mesh.conf` file should contain:
     ```
     ATOM_NAME = Nb Fe Sb  # Example atoms
     DIM = 2 2 2           # Supercell dimensions
     MP = 8 8 8            # Mesh points
     ```

2. `WP3.input` - Configuration file with parameters for the calculation

## Usage Scenarios

### Scenario 1: Calculate Total Scattering Phase Space

This mode outputs the entire weighted scattering phase space to `WP3_plus.txt` and `WP3_minus.txt`.

Example `WP3.input` file:
```
0.15       # sigma in Gaussian function
300        # temperature (K)
F          # do not adjust frequency
T          # calculate total WP3 (T for total, F for specific mode)
```

Command:
```bash
./WP3_phonopy_cuda
```

Note: This calculation is computationally intensive and should be submitted as a batch job.

### Scenario 2: Adjust Frequencies and Analyze Impact

This mode allows you to analyze the effect of raising frequencies above a certain threshold.

Example `WP3.input` file:
```
0.15       # sigma in Gaussian function
300        # temperature (K)
T          # adjust frequency
7.25 3     # frequency threshold (THz) and amount to raise (THz)
T          # output total WP3
```

Note: This calculation is also computationally intensive and should be submitted as a batch job.

### Scenario 3: Analyze Specific Three-Phonon Processes

This mode analyzes specific phonon modes and their three-phonon processes. Output files are `special_WP3_minus.txt` and `special_WP3_plus.txt`.

Example `WP3.input` file:
```
0.15       # sigma in Gaussian function
300        # temperature (K)
F          # do not adjust frequency
T          # analyze specific mode
143        # mode number (example)
```

The output files contain:
- Scattering phase space value
- q' wavevector index
- q' frequency (rad/ps)
- q'' wavevector index
- q'' frequency (rad/ps)

You can sort the output to identify the most significant three-phonon processes:
```bash
sort special_WP3_minus.txt > 123.txt
```

## Performance

The CUDA-parallelized version achieves significant speedup compared to the original OpenMP version:
- Scenario 1 & 2: Multiple orders of magnitude faster, depending on the GPU
- Scenario 3: Fast enough to run interactively without job submission

## Output Files

- `WP3_plus.txt`, `WP3_minus.txt`: Total weighted scattering phase space
- `special_WP3_plus.txt`, `special_WP3_minus.txt`: Mode-specific three-phonon processes

## Tips

1. For large calculations, use job submission with appropriate resource requests
2. When analyzing specific modes, you can identify interesting modes by first examining the total WP3 output
3. Use `sort` command to order the three-phonon processes by their contribution

## CUDA Optimization

This implementation parallelizes the computationally intensive loops in the original code, particularly:
- The nested loops over q-points
- The Gaussian broadening calculations
- The phase space integration
