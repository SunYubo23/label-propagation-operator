using SparseArrays
using Base.Threads
using ProgressMeter
using IterativeSolvers

function mean_absolute_error(f1,f2)
    mask = abs.(f2) .> 1e-5
    return any(mask) ? mean(abs.(f1[mask] - f2[mask])) : 0.0
end

function RandomWalkQ_woT(G, sample_num, label, unlabel)
    v_u = length(unlabel)
    v_l = length(label)
    
    nbr, nbr_weights = neighbor_weight(G)
    
    max_node = G.n
    is_label = falses(max_node)
    for l in label; is_label[l] = true; end
    
    label_col_map = Dict(label[k] => k for k in 1:v_l)
    
    row_dicts = [Dict{Int, Float64}() for _ in 1:v_u]
    
    prog_bar = Progress(v_u, 1, "Random Walking: ")
    completed = Atomic{Int}(0)
    
    @threads for u_idx in 1:v_u
        u = unlabel[u_idx]
        local_dict = row_dicts[u_idx] 
        
        for r in 1:sample_num
            i = u
            path_sign = 1.0  
            while !is_label[i]
                neighbors = nbr[i]
                weights = nbr_weights[i]
                rand_idx = rand(1:length(neighbors))
                path_sign *= weights[rand_idx] 
                i = neighbors[rand_idx]
            end        
            l_idx = label_col_map[i]
            local_dict[l_idx] = get(local_dict, l_idx, 0.0) + path_sign
        end
        
        atomic_add!(completed, 1)
        if completed[] % 1000 == 0; update!(prog_bar, completed[]); end
    end
    update!(prog_bar, v_u)

    println("\nRandom walk finished. Parallelizing matrix assembly...")

    row_nnz = [length(row_dicts[i]) for i in 1:v_u]
    offsets = zeros(Int, v_u + 1)
    for i in 1:v_u
        offsets[i+1] = offsets[i] + row_nnz[i]
    end
    nnz_total = offsets[end]

    I = Vector{Int}(undef, nnz_total)
    J = Vector{Int}(undef, nnz_total)
    V = Vector{Float64}(undef, nnz_total)

    inv_sample_num = 1.0 / Float64(sample_num)
    @threads for u_idx in 1:v_u
        start_idx = offsets[u_idx] + 1
        curr = start_idx
        for (l_idx, val) in row_dicts[u_idx]
            @inbounds I[curr] = u_idx
            @inbounds J[curr] = l_idx
            @inbounds V[curr] = val * inv_sample_num
            curr += 1
        end
    end

    println("Constructing sparse matrix...")
    return sparse(I, J, V, v_u, v_l)
end

function RandomWalkQ(G, sample_num, label, unlabel, max_steps)
    v_u = length(unlabel)
    v_l = length(label)
    
    nbr, nbr_weights = neighbor_weight(G)
    
    max_node = G.n
    is_label = falses(max_node)
    for l in label
        is_label[l] = true
    end
    
    label_col_map = Dict(label[k] => k for k in 1:v_l)
    
    row_dicts = [Dict{Int, Float64}() for _ in 1:v_u]
    
    prog_bar = Progress(v_u, 1, "Random Walking (Truncated): ")
    completed = Atomic{Int}(0)
    @threads for u_idx in 1:v_u
        u = unlabel[u_idx]
        local_dict = row_dicts[u_idx] 
        
        for r in 1:sample_num
            i = u
            path_sign = 1.0  
            step_count = 0
            
            while !is_label[i] && step_count < max_steps
                neighbors = nbr[i]
                weights = nbr_weights[i]
            
                rand_idx = rand(1:length(neighbors))
                i = neighbors[rand_idx]
                path_sign *= weights[rand_idx]
                
                step_count += 1
            end
            
            if is_label[i]
                l_idx = label_col_map[i]
                local_dict[l_idx] = get(local_dict, l_idx, 0.0) + path_sign
            end
        end
        atomic_add!(completed, 1)
        if completed[] % 1000 == 0
            update!(prog_bar, completed[])
        end
    end
    
    update!(prog_bar, v_u) 
    println("\nRandom walk finished. Parallelizing matrix assembly...")
    
    row_nnz = [length(row_dicts[i]) for i in 1:v_u]
    offsets = zeros(Int, v_u + 1)
    for i in 1:v_u
        offsets[i+1] = offsets[i] + row_nnz[i]
    end
    nnz_total = offsets[end]
    
    I = Vector{Int}(undef, nnz_total)
    J = Vector{Int}(undef, nnz_total)
    V = Vector{Float64}(undef, nnz_total)
    inv_sample_num = 1.0 / Float64(sample_num)
    
    @threads for u_idx in 1:v_u
        start_idx = offsets[u_idx] + 1
        curr = start_idx
        
        for (l_idx, val) in row_dicts[u_idx]
            @inbounds I[curr] = u_idx
            @inbounds J[curr] = l_idx
            @inbounds V[curr] = val * inv_sample_num
            curr += 1
        end
    end
    
    println("Constructing sparse matrix...")
    return sparse(I, J, V, v_u, v_l)
end

function MultiTreeQ_woRB(G, sample_num, label, unlabel)
    v_u = length(unlabel)
    v_l = length(label)
    max_node = G.n 
    
    nbr, nbr_weights = neighbor_weight(G) 
    
    is_label = falses(max_node)
    for l in label; is_label[l] = true; end
    
    label_col_map = Dict(label[k] => k for k in 1:v_l)
    
    num_t = Threads.nthreads()
    chunk_size = cld(sample_num, num_t)
    chunks = collect(Iterators.partition(1:sample_num, chunk_size))
    num_chunks = length(chunks)

    chunk_dicts = [Dict{Int, Float64}() for _ in 1:num_chunks]
    chunk_weights = zeros(Float64, num_chunks)
    
    prog_bar = Progress(sample_num, 1, "Sampling Forests: ")
    completed = Atomic{Int}(0)
    
    max_walk_steps = 20 * G.n 

    @threads for chunk_idx in 1:num_chunks
        current_chunk = chunks[chunk_idx]
        local_dict = chunk_dicts[chunk_idx]
        local_weight = 0.0
        
        in_Tree = falses(max_node)
        root = fill(-1, max_node)
        next_node = fill(-1, max_node)
        path_pos = fill(0, max_node)
        current_path = Int[]
        sizehint!(current_path, min(G.n, 10000)) 
        
        edge_weight_out = fill(0.0, max_node)
        
        for k in current_chunk
            fill!(in_Tree, false); fill!(root, -1); fill!(next_node, -1); fill!(path_pos, 0)
            empty!(current_path)
            num_neg_cycles = 0 
            
            for l in label
                in_Tree[l] = true; root[l] = l
            end
            
            for src in unlabel
                if in_Tree[src]; continue; end
                u = src; empty!(current_path)
                step_count = 0
                
                while !in_Tree[u]
                    step_count += 1
                    if step_count > max_walk_steps
                        for n in current_path
                            in_Tree[n] = true; root[n] = -1; path_pos[n] = 0
                        end
                        break
                    end

                    push!(current_path, u); path_pos[u] = length(current_path)
                    
                    neighbors = nbr[u]
                    weights = nbr_weights[u]
                    
                    if isempty(neighbors)
                        in_Tree[u] = true; root[u] = -1; path_pos[u] = 0
                        break
                    end
                    
                    nxt_idx = rand(1:length(neighbors))
                    nxt = neighbors[nxt_idx]
                    w = weights[nxt_idx]
                    
                    next_node[u] = nxt
                    edge_weight_out[u] = w
                    
                    if path_pos[nxt] > 0
                        cycle_start_idx = path_pos[nxt]
                        cycle_sign = w 
                        
                        for idx in cycle_start_idx:(length(current_path)-1)
                            cycle_sign *= edge_weight_out[current_path[idx]]
                        end
                        
                        if cycle_sign > 0
                            for idx in cycle_start_idx:length(current_path)
                                path_pos[current_path[idx]] = 0
                            end
                            resize!(current_path, cycle_start_idx-1); u = nxt 
                        else
                            for n in current_path
                                in_Tree[n] = true; root[n] = -1; path_pos[n] = 0
                            end
                            if cycle_sign < 0; num_neg_cycles += 1; end
                            break 
                        end
                    else
                        u = nxt
                    end
                end 
                
                if !isempty(current_path) && in_Tree[u]
                    target_root = root[u]
                    for n in current_path
                        if path_pos[n] > 0 
                            in_Tree[n] = true; root[n] = target_root; path_pos[n] = 0
                        end
                    end
                end
            end 
            
            W = 2.0 ^ num_neg_cycles
            local_weight += W
            
            for u_idx in 1:v_u
                u = unlabel[u_idx]
                if root[u] != -1
                    path_sign = 1.0
                    curr = u
                    trace_steps = 0
                    
                    while !is_label[curr] && trace_steps < G.n
                        path_sign *= edge_weight_out[curr]
                        curr = next_node[curr]
                        trace_steps += 1
                    end
                    
                    l_idx = label_col_map[curr]
                    
                    linear_idx = (u_idx - 1) * v_l + l_idx
                    local_dict[linear_idx] = get(local_dict, linear_idx, 0.0) + W * path_sign
                end
            end
            
            atomic_add!(completed, 1)
            if completed[] % max(1, sample_num ÷ 100) == 0
                update!(prog_bar, completed[])
            end
        end 
        
        chunk_weights[chunk_idx] = local_weight
    end 
    update!(prog_bar, sample_num)

    println("\nSampling finished. Constructing sparse matrix directly...")
    
    total_weight = sum(chunk_weights)
    inv_weight = total_weight > 0 ? 1.0 / total_weight : 1.0
    

    offsets = zeros(Int, num_chunks + 1)
    for i in 1:num_chunks
        offsets[i+1] = offsets[i] + length(chunk_dicts[i])
    end
    nnz_total = offsets[end]
    
    I = Vector{Int}(undef, nnz_total)
    J = Vector{Int}(undef, nnz_total)
    V = Vector{Float64}(undef, nnz_total)
    
    @threads for chunk_idx in 1:num_chunks
        curr = offsets[chunk_idx] + 1
        for (linear_idx, val) in chunk_dicts[chunk_idx]
            l_idx = (linear_idx - 1) % v_l + 1
            u_idx = (linear_idx - 1) ÷ v_l + 1
            
            @inbounds I[curr] = u_idx
            @inbounds J[curr] = l_idx
            @inbounds V[curr] = val * inv_weight
            curr += 1
        end
    end
    
    return sparse(I, J, V, v_u, v_l)
end



function MultiTreeQ(G, sample_num, label, unlabel)
    v_u = length(unlabel)
    v_l = length(label)
    max_node = G.n 
    
    nbr, nbr_weights = neighbor_weight(G) 
    
    is_label = fill(false, max_node)
    for l in label; is_label[l] = true; end
    
    label_col_map = Dict(label[k] => k for k in 1:v_l)
    
    u_to_idx = fill(0, max_node)
    for (idx, u) in enumerate(unlabel)
        u_to_idx[u] = idx
    end
    
    B_ptr = zeros(Int, max_node + 1)
    B_ptr[1] = 1 
    for u in unlabel
        for nxt in nbr[u]
            if is_label[nxt]
                B_ptr[u+1] += 1
            end
        end
    end
    for i in 1:max_node
        B_ptr[i+1] += B_ptr[i]
    end
    
    total_B_edges = B_ptr[end] - 1
    B_col = Vector{Int}(undef, total_B_edges)
    B_val = Vector{Float64}(undef, total_B_edges)
    D_val = zeros(Float64, max_node)
    
    cur_ptr = copy(B_ptr)
    for u in unlabel
        neighbors = nbr[u]
        weights = nbr_weights[u]
        for i in 1:length(neighbors)
            nxt = neighbors[i]
            if is_label[nxt]
                idx = cur_ptr[u]
                B_col[idx] = label_col_map[nxt]
                B_val[idx] = Float64(weights[i])
                cur_ptr[u] += 1
                D_val[u] += abs(weights[i])
            end
        end
    end

    global_dicts = [Dict{Int, Float64}() for _ in 1:v_u]
    row_locks = [Threads.SpinLock() for _ in 1:4096]
    
    num_t = Threads.nthreads()
    chunk_size = cld(sample_num, num_t)
    chunks = collect(Iterators.partition(1:sample_num, chunk_size))
    num_chunks = length(chunks)
    chunk_weights = zeros(Float64, num_chunks)
    
    max_walk_steps = 20 * G.n 
    prog_bar = Progress(sample_num, 1, "Sampling RB Forests: ")
    completed = Atomic{Int}(0)

    @threads for chunk_idx in 1:num_chunks
        current_chunk = chunks[chunk_idx]
        local_weight = 0.0
        
        in_Tree = fill(false, max_node)
        root = fill(-1, max_node)
        next_node = fill(-1, max_node)
        path_pos = fill(0, max_node)
        current_path = Int[]
        sizehint!(current_path, min(G.n, 10000))
        
        edge_weight_out = fill(0.0, max_node)
        S_val = fill(1.0, max_node)
        head = fill(0, max_node)
        nxt_list = fill(0, max_node)
        active_boundaries = Int[]
        
        Sum_B = zeros(Float64, v_l)
        is_active_l = fill(false, v_l)
        active_l_indices = Int[]
        
        for k in current_chunk
            fill!(in_Tree, false); fill!(root, -1); fill!(next_node, -1); fill!(path_pos, 0)
            empty!(current_path)
            num_neg_cycles = 0 
            
            for l in label
                in_Tree[l] = true; root[l] = l
            end
            
            for src in unlabel
                if in_Tree[src]; continue; end
                u = src; empty!(current_path)
                step_count = 0
                
                while !in_Tree[u]
                    step_count += 1
                    if step_count > max_walk_steps
                        for n in current_path
                            in_Tree[n] = true; root[n] = -1; path_pos[n] = 0
                        end
                        break
                    end

                    push!(current_path, u); path_pos[u] = length(current_path)
                    
                    neighbors = nbr[u]
                    weights = nbr_weights[u]
                    
                    if isempty(neighbors)
                        in_Tree[u] = true; root[u] = -1; path_pos[u] = 0
                        break
                    end
                    
                    nxt_idx = rand(1:length(neighbors))
                    nxt = neighbors[nxt_idx]
                    w = weights[nxt_idx]
                    
                    next_node[u] = nxt
                    edge_weight_out[u] = w 
                    
                    if path_pos[nxt] > 0
                        cycle_start_idx = path_pos[nxt]
                        cycle_sign = Float64(w)
                        
                        for idx in cycle_start_idx:(length(current_path)-1)
                            cycle_sign *= edge_weight_out[current_path[idx]]
                        end
                        
                        if cycle_sign > 0
                            for idx in cycle_start_idx:length(current_path)
                                path_pos[current_path[idx]] = 0
                            end
                            resize!(current_path, cycle_start_idx-1); u = nxt 
                        else
                            for n in current_path
                                in_Tree[n] = true; root[n] = -1; path_pos[n] = 0
                            end
                            if cycle_sign < 0; num_neg_cycles += 1; end
                            break 
                        end
                    else
                        u = nxt
                    end
                end 
                
                if !isempty(current_path) && in_Tree[u]
                    target_root = root[u]
                    for n in current_path
                        if path_pos[n] > 0 
                            in_Tree[n] = true; root[n] = target_root; path_pos[n] = 0
                        end
                    end
                end
            end 
            
            W = 2.0 ^ num_neg_cycles
            local_weight += W
            
            for u in unlabel
                if root[u] == -1; continue; end
                curr = u; path_sign = 1.0
                rb_steps = 0
                
                while !is_label[next_node[curr]] && rb_steps < G.n
                    path_sign *= edge_weight_out[curr]
                    curr = next_node[curr]
                    rb_steps += 1
                end
                
                boundary_node = curr
                S_val[u] = path_sign
                
                if head[boundary_node] == 0
                    push!(active_boundaries, boundary_node)
                end
                nxt_list[u] = head[boundary_node]
                head[boundary_node] = u
            end
            
            for b in active_boundaries
                Sum_D = 0.0
                
                curr_u = head[b]
                while curr_u != 0
                    Sum_D += D_val[curr_u]
                    S_u = S_val[curr_u]
                    
                    for idx in B_ptr[curr_u]:(B_ptr[curr_u+1]-1)
                        l_idx = B_col[idx]
                        w_sgn = B_val[idx]
                        
                        if !is_active_l[l_idx]
                            is_active_l[l_idx] = true
                            push!(active_l_indices, l_idx)
                        end
                        Sum_B[l_idx] += S_u * w_sgn
                    end
                    curr_u = nxt_list[curr_u]
                end
                
                if Sum_D > 0
                    inv_Sum_D = 1.0 / Sum_D 
                    curr_u = head[b]
                    while curr_u != 0
                        S_u = S_val[curr_u]
                        u_idx = u_to_idx[curr_u]
                        factor = W * S_u * inv_Sum_D
                        lock_idx = (u_idx & 4095) + 1
                        lck = row_locks[lock_idx]
                        lock(lck)
                        dict_u = global_dicts[u_idx]
                        for l_idx in active_l_indices
                            val = Sum_B[l_idx]
                            if val != 0.0
                                dict_u[l_idx] = get(dict_u, l_idx, 0.0) + factor * val
                            end
                        end
                        unlock(lck)
                        
                        curr_u = nxt_list[curr_u]
                    end
                end
                
                for l_idx in active_l_indices
                    Sum_B[l_idx] = 0.0
                    is_active_l[l_idx] = false
                end
                empty!(active_l_indices)
                
                head[b] = 0
            end
            empty!(active_boundaries)
            
            atomic_add!(completed, 1)
            if completed[] % max(1, sample_num ÷ 100) == 0
                update!(prog_bar, completed[])
            end
        end 
        
        chunk_weights[chunk_idx] = local_weight
    end 
    update!(prog_bar, sample_num)
    println("\nRB Sampling finished. Assembling final sparse matrix...")
    
    total_weight = sum(chunk_weights)
    inv_weight = total_weight > 0 ? 1.0 / total_weight : 1.0
    
    row_nnz = [length(d) for d in global_dicts]
    offsets = zeros(Int, v_u + 1)
    for i in 1:v_u
        offsets[i+1] = offsets[i] + row_nnz[i]
    end
    nnz_total = offsets[end]
    
    I = Vector{Int}(undef, nnz_total)
    J = Vector{Int}(undef, nnz_total)
    V = Vector{Float64}(undef, nnz_total)
    @threads for u_idx in 1:v_u
        curr = offsets[u_idx] + 1
        for (l_idx, val) in global_dicts[u_idx]
            @inbounds I[curr] = u_idx
            @inbounds J[curr] = l_idx
            @inbounds V[curr] = val * inv_weight
            curr += 1
        end
    end
    
    return sparse(I, J, V, v_u, v_l)
end

function baseline_label_propagation(p_uu, p_ul; max_iter=1000, tol=1e-4)
    v_u, v_l = size(p_ul)
    Q = copy(p_ul) 
    println("Running Baseline 1: Label Propagation...")
    prog_bar = Progress(max_iter, 1, "Label Prop: ")
    for iter in 1:max_iter
        Q_next = p_uu * Q .+ p_ul
        diff = norm(Q_next - Q) / norm(Q)
        Q = Q_next
        ProgressMeter.update!(prog_bar, iter, showvalues = [(:Current_Diff, diff), (:Target_Tol, tol)])
        if diff < tol
            ProgressMeter.finish!(prog_bar)
            println("Label Propagation converged at iteration $iter")
            break
        end
        if iter == max_iter
            println("\nWarning: Label Propagation reached max iterations ($max_iter) without full convergence.")
        end
    end
    return Q
end


function baseline_gmres_solver(p_uu, p_ul; tol=1e-4)
    v_u, v_l = size(p_ul)
    Q = zeros(Float64, v_u, v_l)
    A = sparse(I, v_u, v_u) - p_uu 
    println("Running Baseline 2: GMRES Solver (Multi-threaded)...")
    prog_bar = Progress(v_l, 1, "GMRES Solving: ")
    completed = Atomic{Int}(0)
    @threads for i in 1:v_l
        b_i = Vector(p_ul[:, i])
        if norm(b_i) > 0.0
            x_i = gmres(A, b_i, reltol=tol, log=false)
            Q[:, i] = x_i
        end
        atomic_add!(completed, 1)
        if completed[] % max(1, v_l ÷ 100) == 0
            update!(prog_bar, completed[])
        end
    end
    update!(prog_bar, v_l) 
    println("GMRES Solver finished!")
    return Q
end