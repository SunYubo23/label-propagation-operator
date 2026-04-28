# Fast Algorithms for the Label Propagation Operator on Signed Graphs

This repository contains the official Julia implementation for the paper **"Fast Algorithms for the Label Propagation Operator on Signed Graphs"**.

It provides the source code for our proposed fast algorithms, baseline comparisons, and ablation studies to efficiently compute the label propagation operator (Q-matrix) on signed networks.

##  Algorithms Overview

In our paper, we propose two primary fast algorithms and compare them alongside their respective ablation variants:

**Main Algorithms:**
1. `RandomWalkQ`: A fast approximation algorithm based on random walks.
2. `MultiTreeQ`: An optimized multi-tree based approach for scalable label propagation.

**Ablation Studies:**
1. `RandomWalkQ_woT`: The random walk approach without [T , e.g., Truncation].
2. `MultiTreeQ_woRB`: The multi-tree approach without [RB, e.g., Rao-Blackwellization].

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
