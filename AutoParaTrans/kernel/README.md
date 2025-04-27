# Fortran to CUDA Converter

A tool for automatically converting Fortran multi-loop code to CUDA parallel code.

## Introduction

This tool aims to simplify the GPU acceleration process for Fortran scientific computing code. It automatically analyzes loop structures and variable dependencies in Fortran source code and generates equivalent CUDA parallel code. This is particularly useful for accelerating compute-intensive scientific and engineering applications.

## Features

- **Automatic Fortran Parsing**: Supports standard Fortran 90/95/2003 syntax
- **Loop Structure Analysis**: Identifies various types of loops (standard DO loops, DO WHILE loops, unconditional DO loops)
- **Variable Dependency Analysis**: Detects loop variables, array access patterns, and data dependencies
- **Parallelism Detection**: Automatically identifies parallelizable loops
- **CUDA Code Generation**: Generates optimized CUDA kernels and host-side calling code
- **Extensibility**: Can be used in various scientific computing domains, including but not limited to:
  - Computational physics
  - Computational chemistry
  - Climate modeling
  - Fluid dynamics
  - Materials science
  - Quantum computing simulations

## How It Works

The conversion process includes the following major steps:

1. **Static Analysis**:
   - Lexical analysis: Processes source code, extracting tokens and syntactic elements
   - Variable detection: Identifies all variables, arrays, and their types
   - Loop analysis: Identifies all loops and their nested structure
   - AST construction: Builds an abstract syntax tree representation

2. **Parallelism Analysis**:
   - Identifies loop-carried dependencies
   - Analyzes array access patterns
   - Determines which loops can be parallelized

3. **Code Generation**:
   - Maps parallelizable loops to CUDA threads
   - Generates appropriate thread block and grid dimensions
   - Preserves serial structure for non-parallelizable loops
   - Generates complete CUDA kernel and host-side code

## Usage

### Basic Usage

```python
import fortran_to_cuda_converter as f2c

# Fortran source code
fortran_code = """
program example
  implicit none
  integer :: i, j, n = 100
  real :: a(100,100), b(100,100), c(100,100)
  
  do i = 1, n
    do j = 1, n
      c(i,j) = a(i,j) + b(i,j)
    end do
  end do
end program example
"""

# Convert to CUDA
result = f2c.convert_fortran_to_cuda(fortran_code)

# Output results
print(result['cuda_kernel'])
print(result['host_code'])
```

### Command Line Usage

```bash
python fortran_to_cuda.py input.f90 -o output.cu
```

## Installation

```bash
pip install fortran-to-cuda-converter
```

## Dependencies

- Python 3.6+
- re (regular expression library, Python standard library)

## Limitations

- Currently does not support complex preprocessor directives
- Does not handle dependencies between Fortran modules
- May require manual intervention for complex data dependency analysis
- Does not support all Fortran language features, primarily focusing on loop structures

## Optimization Suggestions

The converted CUDA code may need further optimization:

1. **Memory Optimization**: Consider using shared memory to reduce global memory access
2. **Thread Block Size Adjustment**: Adjust thread block size based on specific GPU architecture
3. **Loop Unrolling**: Consider manual unrolling for small inner loops to improve performance
4. **Atomic Operations**: May need to add atomic operations when handling write conflicts

## Examples

### Example 1: Matrix Multiplication

**Fortran Code**:

```fortran
! Matrix multiplication
do i = 1, n
  do j = 1, n
    do k = 1, n
      c(i,j) = c(i,j) + a(i,k) * b(k,j)
    end do
  end do
end do
```

**Generated CUDA Code**:

```cuda
__global__ void matrix_multiply_kernel(float* a, float* b, float* c, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (i < n && j < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
      sum += a[i * n + k] * b[k * n + j];
    }
    c[i * n + j] = sum;
  }
}
```

### Example 2: Jacobi Iteration

**Fortran Code**:

```fortran
! Jacobi iteration
do iter = 1, max_iter
  do i = 2, n-1
    do j = 2, n-1
      new_u(i,j) = 0.25 * (u(i+1,j) + u(i-1,j) + u(i,j+1) + u(i,j-1))
    end do
  end do
  
  do i = 2, n-1
    do j = 2, n-1
      u(i,j) = new_u(i,j)
    end do
  end do
end do
```

**Generated CUDA Code** (partial):

```cuda
__global__ void jacobi_kernel(float* u, float* new_u, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  
  if (i < n-1 && j < n-1) {
    new_u[i*n + j] = 0.25f * (u[(i+1)*n + j] + u[(i-1)*n + j] + 
                             u[i*n + (j+1)] + u[i*n + (j-1)]);
  }
}
```

