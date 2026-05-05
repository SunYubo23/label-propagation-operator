# Fast Algorithms for the Label Propagation Operator on Signed Graphs

This repository contains the official Julia implementation for the paper **"Fast Algorithms for the Label Propagation Operator on Signed Graphs"**.

It provides the source code for our proposed fast algorithms, baseline comparisons, and ablation studies to efficiently compute the label propagation operator (Q-matrix) on signed networks.

## Algorithms and Baselines

In this paper, we focus on the development of a scalable label propagation approach. We introduce our proposed algorithm and evaluate its performance against competitive baselines and ablation variants.

### Proposed Method
* `MultiTreeQ`: Our primary contribution, an optimized multi-tree based approach designed for scalable label propagation. It leverages [mention key technique, e.g., forest sampling or structural balance] to achieve high computational efficiency on large-scale signed graphs.

### Baselines and Ablation Variants
To demonstrate the effectiveness of `MultiTreeQ`, we compare it with the following algorithms:

1.  `RandomWalkQ`: A fast approximation baseline based on random walks, used to provide a performance benchmark for accuracy and convergence speed.
2.  `MultiTreeQ_woRB`: An ablation variant of our proposed method without [e.g., Rao-Blackwellization], used to quantify the contribution of specific optimization components within `MultiTreeQ`.
3.  `RandomWalkQ_woT`: A simplified version of the random walk baseline without [e.g., Truncation], included to provide a comprehensive comparison across different algorithmic configurations.
##  Repository Structure

* `algorithm.jl`: Contains the core implementations of the four algorithms mentioned above, as well as the exact/baseline solver.
* `graph.jl`: Utilities for graph I/O, constructing adjacency, degree, Laplacian, and transition matrices.
* `small_dataset.jl`: Benchmark script for smaller datasets. Compares execution time, memory allocation, and Mean Absolute Error (MAE) against the exact solution.
* `large_dataset.jl`: Benchmark script for large-scale datasets. Focuses purely on execution time (scalability) as exact solutions are computationally prohibitive.
* `log/`: Directory where benchmark results and logs are automatically saved.

##  Prerequisites

This codebase is written in **Julia**. To run the scripts, you need to install Julia and the following required packages. 

Open the Julia REPL and run:

```julia
using Pkg
Pkg.add([
    "Arpack", 
    "LinearAlgebra", 
    "SparseArrays", 
    "Statistics", 
    "Random", 
    "Dates", 
    "IterativeSolvers", 
    "ProgressMeter"
])
```
##  Running the Benchmarks

### 1. Small Dataset Benchmark (Accuracy & Performance)
The `small_dataset.jl` script evaluates the algorithms on datasets where computing the exact Q-matrix is feasible. It reports:
* Execution Time (seconds)
* Memory Allocation (MB)
* Mean Absolute Error (MAE)

**Usage:**
Run the following command from the root directory of the project:
```bash
julia -t auto src/small_dataset.jl
```
*Results will be automatically saved in `log/<dataset_name>_results.txt`.*

### 2. Large Dataset Benchmark (Scalability)
The `large_dataset.jl` script is designed to test the scalability of the algorithms on massive graphs. Because the exact solution is too expensive to compute, this script **only tracks execution time**.

**Usage:**
Run the following command from the root directory of the project:
```bash
julia -t auto src/large_dataset.jl
```
