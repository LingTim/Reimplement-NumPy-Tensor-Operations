# Reimplement NumPy Tensor Operations

## URL https://github.com/LingTim/Reimplement-NumPy-Tensor-Operations

## Author GitHub Account: LingTim

## Motivation Since native Python is inherently slow for numerical computations, developers rely heavily on NumPy for acceleration. While NumPy's backend is traditionally built with C and Fortran, modern C++ has become the mainstream standard for system software. Therefore, this project aims to simulate the underlying logic of NumPy's matrix operations using purely C++.

## Operations and Routines I will implement the following core operations as C++ extensions for Python:

1.Test programs. 2. Define the C++ N-dimensional array. 3. Implement element-wise operations. 4. Implement broadcasting operations. 5. Implement reduction operations. 6. Implement tensor contraction.

Optimization & Interface: * All operations will first be implemented in standard C++ (naive baseline) to verify correctness. Subsequently, the AVX instruction set will be explicitly utilized to vectorize and accelerate the computations. * Pybind11 will be used to wrap and compile these C++ functions into .pyd dynamic libraries, making them directly callable within Python.

## Performance Goals Instead of attempting to beat the highly optimized production-level NumPy, the goal of this project is to benchmark and analyze the performance differences between pure Python implementations, the naive C++ baseline, and the AVX-optimized C++ versions.

## Schedule

Week 6-7(3/30~4/12): Write test programs, define the C++ N-dimensional array data
structure, and handle strides and shapes.

Week 8(4/13~4/19): Implement element-wise operations.
Week 9(4/20~4/26): Implement broadcasting and reduction operations.
Week 10(4/27~5/3): Implement tensor contraction.
Week 11(5/4~5/10): Apply AVX2 instructions and cache-friendly loop tiling to tensor contraction.
Week 12(5/11~5/17): Benchmark pure Python, naive C++, and optimized C++, and analyze memory and CPU bottlenecks.