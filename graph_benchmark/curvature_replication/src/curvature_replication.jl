#=
This is a replication of the optimised linear algebra code from the paper
"Understanding over-squashing and bottlenecks on graphs via curvature",
the code is found at https://github.com/jctops/understanding-oversquashing/blob/main/gdl/src/gdl/curvature/numba.py

We changed the code to use sparse matrices so large datasets could fit in memory.
=#

using MLDatasets; PubMed
using SparseArrays

# c is our curvature matrix out
# A is our adjacency matrix
function balanced_forman_curvature(A)
    N = size(A, 1)
    A2 = A^2
    d_in = sum(A, dims=1)
    d_out = sum(A, dims=2)
   
    C = spzeros(N, N)
    
     @inbounds for (i, j) in zip(findnz(A)...)
        if A[i, j] == 0
            continue
        end
        d_min = min(d_in[i], d_out[j])
        d_max = max(d_in[i], d_out[j])
        sharp_ij = 0
        lambda_ij = 0

        A_ij= A[i, j]
        @inbounds @simd for k in 1:N
            tmp = A[k, j] * (A2[i, k] - A[i, k]) * A_ij
            if tmp > 0
                sharp_ij += 1
                lambda_ij = max(lambda_ij, tmp)
            end
            tmp = A[i, k] * (A2[k, j] - A[k, j]) * A_ij
            if tmp > 0
                sharp_ij += 1

                lambda_ij = max(lambda_ij, tmp)
            end
        end
        C[i, j] =  ((2 / d_max) + (2 / d_min) - 2 + (2 / d_max + 1 / d_min) * A2[i, j] * A_ij)
        if lambda_ij > 0
            C[i, j] += sharp_ij / (d_max * lambda_ij) 
        end
    end
    
    C
end

data = PubMed()

function edge_index_to_adj(edge_index)
    N = maximum(edge_index)
    A = spzeros(Bool, N, N)
    @inbounds for (i, j) in zip(edge_index...)
        A[i, j] = true
    end
    return A
end

A = edge_index_to_adj(data.graphs[1].edge_index)

start = time()
C = balanced_forman_curvature(A)

end_time = time()
println("Time taken to compute BFC for the whole graph: ", end_time - start)
