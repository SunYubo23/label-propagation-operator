include("graph.jl")
include("algorithm.jl")

using Arpack, LinearAlgebra, SparseArrays
using Statistics
using Random
using Dates


function run_benchmark(func, QQ_exact; runs=5)
    times = Float64[]
    allocs = Float64[]
    maes = Float64[]
    
    for i in 1:runs
        GC.gc() 
        res = @timed func()
        ans_matrix = res.value
        
        push!(times, res.time)
        push!(allocs, res.bytes / 1024^2)
        push!(maes, mean_absolute_error(ans_matrix, QQ_exact))
    end
    
    return (
        time_avg = mean(times), time_std = std(times),
        alloc_avg = mean(allocs), alloc_std = std(allocs),
        mae_avg = mean(maes), mae_std = std(maes)
    )
end

dataset_name = "bitcoinalpha.txt"

dataset_basename = splitext(basename(dataset_name))[1] 

G = read_data(dataset_name)

println("Dataset: ", dataset_name)
@show G.n, G.m

A = adjacency_matrix(G)
D = degree_matrix(G)
L = laplacian_matrix(G)
_, P = transition_matrix(G)

num = round(Int, G.n * 0.010)
@show num

shuffled_nodes = randperm(G.n)
label = shuffled_nodes[1:num]
unlabel = shuffled_nodes[num+1:end]
epsilon = 0.05 
sample_num = round(Int, 2*epsilon^(-2) * log(G.n))
test_runs = 5

p_uu = P[unlabel, unlabel]
p_ul = P[unlabel, label]

println("\nCalculating exact solution Q...")
# exact_timed = @timed (inv(Matrix(I - p_uu)) * p_ul)
exact_timed = @timed baseline_gmres_solver(p_uu,p_ul)
QQ = exact_timed.value
time_exact = exact_timed.time

println("Exact solution took $(round(time_exact, digits=3)) seconds.")
println(Sys.CPU_THREADS, " CPU threads are available.")
println(nthreads(), " Julia threads are being used.\n")

println("Warming up JIT compiler, please wait a few seconds...")
dummy_unlabel = unlabel[1:min(10, length(unlabel))]
_ = RandomWalkQ_woT(G, 1, label, dummy_unlabel)
_ = RandomWalkQ(G, 1, label, dummy_unlabel, 2)
_ = MultiTreeQ_woRB(G, Threads.nthreads(), label, dummy_unlabel)
_ = MultiTreeQ(G, Threads.nthreads(), label, dummy_unlabel)
println("Warm-up finished! 100% compiled. Starting real benchmarks...\n")
println("-"^50)

println("Benchmarking Algo 1 (RandomWalkQ_woT) - $test_runs runs...")
res1 = run_benchmark(() -> RandomWalkQ_woT(G, sample_num, label, unlabel), QQ, runs=test_runs)

println("Benchmarking Algo 2 (RandomWalkQ) - $test_runs runs...")
res2 = run_benchmark(() -> RandomWalkQ(G, sample_num, label, unlabel, 30), QQ, runs=test_runs)

println("Benchmarking Algo 3 (MultiTreeQ_woRB) - $test_runs runs...")
res3 = run_benchmark(() -> MultiTreeQ_woRB(G, sample_num, label, unlabel), QQ, runs=test_runs)

println("Benchmarking Algo 4 (MultiTreeQ) - $test_runs runs...")
res4 = run_benchmark(() -> MultiTreeQ(G, sample_num, label, unlabel), QQ, runs=test_runs)

log_dir = joinpath(@__DIR__, "..", "log")
mkpath(log_dir)

log_file = joinpath(log_dir, "$(dataset_basename)_results.txt")

current_time = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
fmt(m, s) = "$(round(m, digits=6)) ± $(round(s, digits=6))"
log_content = """
======================================================================
Time          : $current_time
Dataset       : $dataset_name
Graph Size    : Nodes = $(G.n), Edges = $(G.m)
Split         : Labels = $(length(label)), Unlabels = $(length(unlabel))
Parameters    : Samples = $sample_num, Threads = $(nthreads())
Test Runs     : $test_runs times (Results: Mean ± Std)
Exact Sol Time: $(round(time_exact, digits=3)) s
----------------------------------------------------------------------
[1] RandomWalkQ_woT
    Time (s) : $(fmt(res1.time_avg, res1.time_std))
    Alloc(MB): $(fmt(res1.alloc_avg, res1.alloc_std))
    MAE      : $(fmt(res1.mae_avg, res1.mae_std))

[2] RandomWalkQ
    Time (s) : $(fmt(res2.time_avg, res2.time_std))
    Alloc(MB): $(fmt(res2.alloc_avg, res2.alloc_std))
    MAE      : $(fmt(res2.mae_avg, res2.mae_std))

[3] MultiTreeQ_woRB
    Time (s) : $(fmt(res3.time_avg, res3.time_std))
    Alloc(MB): $(fmt(res3.alloc_avg, res3.alloc_std))
    MAE      : $(fmt(res3.mae_avg, res3.mae_std))

[4] MultiTreeQ
    Time (s) : $(fmt(res4.time_avg, res4.time_std))
    Alloc(MB): $(fmt(res4.alloc_avg, res4.alloc_std))
    MAE      : $(fmt(res4.mae_avg, res4.mae_std))
======================================================================
"""

println("\n" * log_content)

open(log_file, "a") do f
    write(f, log_content)
end

println("All benchmarks finished! Results have been saved to: $(abspath(log_file))")