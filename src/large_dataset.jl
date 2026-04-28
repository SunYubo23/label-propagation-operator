include("graph.jl")
include("syb.jl")

using Arpack, LinearAlgebra, SparseArrays
using Random
using Dates

dataset_path = "youtube.txt" 
dataset_basename = splitext(basename(dataset_path))[1]

println("Loading Large Dataset: ", dataset_path)
G = read_data(dataset_path)
@show G.n, G.m

num = round(Int, G.n * 0.01)
shuffled_nodes = randperm(G.n)
label = shuffled_nodes[1:num]
unlabel = shuffled_nodes[num+1:end]

epsilon = 0.1
sample_num = round(Int, 2*epsilon^(-2) * log(G.n)) 
println("Samples: $sample_num | Labels: $(length(label)) | Threads: $(nthreads())")

println("\nWarming up JIT compiler with dummy data...")
dummy_unlabel = unlabel[1:min(5, length(unlabel))]
_ = RandomWalkQ_woT(G, 1, label, dummy_unlabel)
_ = RandomWalkQ(G, 1, label, dummy_unlabel, 2)
_ = MultiTreeQ_woRB(G, nthreads(), label, dummy_unlabel)
_ = MultiTreeQ(G, nthreads(), label, dummy_unlabel)
println("Warm-up finished!\n" * "-"^50)


println("Running Algo 1: RandomWalkQ_woT...")
GC.gc()
t1 = @timed RandomWalkQ_woT(G, sample_num, label, unlabel)


println("Running Algo 2: RandomWalkQ...")
GC.gc()
t2 = @timed RandomWalkQ(G, sample_num, label, unlabel, 60)

println("Running Algo 3: MultiTreeQ_woRB...")
GC.gc()
t3 = @timed MultiTreeQ_woRB(G, sample_num, label, unlabel)

println("Running Algo 4: MultiTreeQ...")
GC.gc()
t4 = @timed MultiTreeQ(G, sample_num, label, unlabel)

log_dir = joinpath(@__DIR__, "..", "log")
mkpath(log_dir)
log_file = joinpath(log_dir, "LARGE_SCALE_$(dataset_basename).txt")

current_time = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
fmt_time(t) = "$(round(t, digits=3)) s"
fmt_mb(b) = "$(round(b / 1024^2, digits=2)) MB"

log_content = """
======================================================================
LARGE SCALE PERFORMANCE TEST
Time          : $current_time
Dataset       : $dataset_path
Graph Size    : Nodes = $(G.n), Edges = $(G.m)
Config        : Samples = $sample_num, Threads = $(nthreads())
Note          : Single run, no ground truth comparison.
----------------------------------------------------------------------
[1] RandomWalkQ_woT
    Time : $(fmt_time(t1.time)) | Alloc : $(fmt_mb(t1.bytes))

[2] RandomWalkQ
    Time : $(fmt_time(t2.time)) | Alloc : $(fmt_mb(t2.bytes))

[3] MultiTreeQ_woRB
    Time : $(fmt_time(t3.time)) | Alloc : $(fmt_mb(t3.bytes))

[4] MultiTreeQ
    Time : $(fmt_time(t4.time)) | Alloc : $(fmt_mb(t4.bytes))
======================================================================
"""

println("\n" * log_content)

open(log_file, "a") do f
    write(f, log_content)
end

println("Large scale benchmark finished! Saved to: $log_file")