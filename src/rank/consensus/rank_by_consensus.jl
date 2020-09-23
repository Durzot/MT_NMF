"""
Created on Sun Apr 26 2020

@author: Yoann Pradat

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Rank selection method in NMF based on multiple metrics computed from the consensus matrices. These metrics include
    - the cophenetic correlation
        J-P. Brunet et al, Metagenes and molecular pattern discovery using matrix factorization. 2009, PNAS.
    - the dispersion
        H. Kim and H. Park, Sparse non-negative matrix factorizations via alternating non-negativity-constrained least 
        squares for microarray data analysis. 2007, Bioinformatics
    - the silhouette width 
        Rousseeuw PJ. Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Jour- nal of Computational and Applied Mathematics. 1987
"""

abstract type AbstractRCParams <: AbstractNMFParams end

"""
    RCParams([,K_min, K_max, n_iter, pert, seed])

Parameters passed to `rank_by_consensus`.

# Arguments
- `K_min::Integer`     : Minimum rank evaluated.
- `K_max::Integer`     : Maximum rank evaluated.
- `n_iter::Integer`    : Total number of iterations, each iteration corresponding to one run of NMF on one random 
    initialization of the matrices W and H.
- `seed::Integer`      : Seed for the random number generator.
"""

mutable struct RCParams <: AbstractRCParams
    K_min::Integer
    K_max::Integer
    n_iter::Integer
    seed::Integer

    function RCParams(;
        K_min  = 2,
        K_max  = 10,
        n_iter = 100,
        seed   = 0)

        K_min > 1 || throw(ArgumentError("K_min must be superior or equal to 2"))
        K_max >= K_min || throw(ArgumentError("K_max must be superior or equal to K_min"))
        n_iter > 0 || throw(ArgumentError("n_iter must be superior or equal to 1"))

        new(K_min, K_max, n_iter, seed)
    end
end

abstract type AbstractRCNMFResults <: AbstractNMFResults end

"""
    RCNMFResults(grid_nmf_results, list_rank, list_coph, list_idxs, list_cons)

Struct for storing results returned by `rank_by_consensus`.
"""
mutable struct RCNMFResults <: AbstractRCNMFResults
    df_metrics::DataFrame

    list_idxs::Vector{Vector{Int64}}
    list_cons::Vector{Array{Float64, 2}}

    function RCNMFResults(
        df_metrics::DataFrame,
        list_idxs::Vector{Vector{Int64}},
        list_cons::Vector{Array{Float64, 2}})

        new(df_metrics, list_idxs, list_cons)
    end
end

"""
    _rank_by_consensus_one((V, nmf))

Helper function for `rank_by_consensus`.
"""
function _rank_by_consensus_one((V, nmf))
    nmf_results = nmf.solver(V, nmf.global_params, nmf.local_params)
end


"""
    rank_by_consensus(V, rc_params, nmf)

To run in parallel the arguments must have been made available to all workers using `@everywhere` macro. 
Returns in a tuple the ranks, cophenetics correlations, ordering indices and consensus matrices.
"""

function rank_by_consensus(V::Matrix{T}, rc_params::RCParams, nmf::NMF) where T <: Real
    #### fix random number generator
    rng = MersenneTwister(rc_params.seed)

    #### build grid
    list_nmf    = Vector{NMF}()

    for rank in rc_params.K_min:rc_params.K_max
        for seed in randperm(rng, rc_params.n_iter)
            nmf_rank_seed = copy(nmf)
            nmf_rank_seed.global_params.rank = rank
            nmf_rank_seed.global_params.seed = seed

            push!(list_nmf, nmf_rank_seed)
        end
    end

    grid = vec(collect(Iterators.product([V], list_nmf)))

    #### run in parallel
    grid_res = @showprogress pmap(grid) do g
        _rank_by_consensus_one(g)
    end

    grid_rank     = (x -> x.global_params.rank).(grid_res)
    grid_W        = (x -> x.W).(grid_res)
    grid_H        = (x -> x.H).(grid_res)
    list_rank     = Vector{Int64}()
    list_coph     = Vector{Float64}()
    list_disp     = Vector{Float64}()
    list_silh     = Vector{Float64}()
    list_idxs     = Vector{Vector{Int64}}()
    list_cons     = Vector{Array{Float64, 2}}()
    list_sparse_W = Vector{Float64}()
    list_sparse_H = Vector{Float64}()

    for rank in rc_params.K_min:rc_params.K_max
        #### compute consensus matrix
        cons = mean((H -> _connectivity_matrix(H)).(grid_H[grid_rank .== rank]))
    
        #### perform hierarchical clustering
        hc = hclust(1 .- cons, linkage=:average, branchorder=:r)

        #### compute cophenetic correlation
        coph = cons_cophenetic(cons, hc)

        #### compute dispersion
        disp = cons_dispersion(cons)

        #### compute avg silhouette width
        silh = cons_avg_silhouette(cons, hc, rank)

        #### reorder consensus matrix
        cons = cons[hc.order, hc.order]

        ### compute average sparseness of rows of W
        sparse_W = mean((W -> matr_avg_row_sparseness(W)).(grid_W[grid_rank .== rank]))

        ### compute average sparseness of columns of H
        sparse_H = mean((H -> matr_avg_col_sparseness(H)).(grid_H[grid_rank .== rank]))

        push!(list_rank, rank)
        push!(list_coph, coph)
        push!(list_disp, disp)
        push!(list_silh, silh)
        push!(list_idxs, hc.order)
        push!(list_cons, cons)
        push!(list_sparse_W, sparse_W)
        push!(list_sparse_H, sparse_H)
    end

    df_metrics = DataFrame(
        rank       = list_rank, 
        cophenetic = list_coph, 
        dispersion = list_disp, 
        silhouette = list_silh, 
        sparse_W   = list_sparse_W,
        sparse_H   = list_sparse_H
       )

    RCNMFResults(
        df_metrics,
        list_idxs, 
        list_cons
    )
end
