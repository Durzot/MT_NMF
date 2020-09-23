"""
Created on Fri May 01 2020

@author: Yoann Pradat

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Rank selection method in NMF based on average stability of latent factor across perturbed versions of the input matrix.
matrix V.
Alexandrov et al, Deciphering Signatures of Mutational Processes Operative in Human Cancer. 2013, Cell Reports.
"""

abstract type AbstractRSParams <: AbstractNMFParams end

"""
    RSParams([,K_min, K_max, n_iter, pert, seed])

Parameters passed to `rank_by_stability`.

# Arguments
- `K_min::Integer`     : Minimum rank evaluated.
- `K_max::Integer`     : Maximum rank evaluated.
- `n_iter::Integer`    : Total number of iterations, each iteration corresponding to one run of NMF on one perturbed 
    version of the matrix `V`.
- `pert_meth::Symbol`  : Perturbation method of the matrix `V` at each iteration. Choose :multinomial for random draws 
    from a Multinomial distribution or :noise for another distribution as specified by `pert_dist`. 
- `pert_dist::Distribution` : Distribution from  which additive random noise will be drawn in case `pert_meth` is :noise.
- `seed::Integer`      : Seed for the random numbner generator.
"""
mutable struct RSParams <: AbstractRSParams
    K_min::Integer
    K_max::Integer
    n_iter::Integer
    pert_meth::Symbol
    pert_dist::Union{Nothing, Distribution}
    seed::Integer

    function RSParams(;
        K_min::Integer                         = 2,
        K_max::Integer                         = 10,
        n_iter::Integer                        = 250,
        pert_meth::Symbol                      = :multinomial,
        pert_dist::Union{Nothing,Distribution} = nothing,
        seed::Integer                          = 0)

        K_min > 0 || throw(ArgumentError("K_min must be superior or equal to 0"))
        K_max >= K_min || throw(ArgumentError("K_max must be superior or equal to K_min"))
        n_iter > 0 || throw(ArgumentError("n_iter must be superior or equal to 1"))

        new(K_min, K_max, n_iter, pert_meth, pert_dist, seed)
    end
end

abstract type AbstractRSNMFResults <: AbstractNMFResults end

"""
    RSNMFResults(grid_nmf_results, list_rank, list_clu_results, list_fid_results)

Struct for storing results returned by `rank_by_stability`.
"""
mutable struct RSNMFResults{T} <: AbstractRSNMFResults
    grid_nmf_results::Vector{NMFResults{T}}
    df_metrics::DataFrame
    list_clu_results::Vector{RSCluNMFResults{T}}
    list_fid_results::Vector{RSFidNMFResults{T}}

    function RSNMFResults{T}(
        grid_nmf_results::Vector{NMFResults{T}},
        df_metrics::DataFrame,
        list_clu_results::Vector{RSCluNMFResults{T}},
        list_fid_results::Vector{RSFidNMFResults{T}}) where T <: Real

        new{T}(grid_nmf_results, df_metrics, list_clu_results, list_fid_results)
    end
end

"""
    _rank_by_stability_one((V_pert, nmf))

Helper function for `rank_by_stability`.
"""
function _rank_by_stability_one((V, nmf)) where T <: Real
    nmf.solver(V, nmf.global_params, nmf.local_params)
end

"""
    rank_by_stability(V, rs_params, nmf)

Rank selection method in NMF based on average stability of latent factor across perturbed versions of the input matrix
V.

# Arguments
- `V::Matrix{T}` : matrix to be factorised
- `rs_params::RSParams` : see `RSParams`(@ref)
- `nmf::NMF` : see `NMF`(@ref)

# References
- Alexandrov et al, Deciphering Signatures of Mutational Processes Operative in Human Cancer. 2013, Cell Reports.
"""
function rank_by_stability(V::Matrix{T}, rs_params::RSParams, nmf::NMF) where T <: Real
    #### fix the rng
    rng = Random.seed!(rs_params.seed)

    K_min  = rs_params.K_min
    K_max  = rs_params.K_max
    n_iter = rs_params.n_iter
    pert_meth = rs_params.pert_meth
    pert_dist = rs_params.pert_dist

    #### fix seeds for W and H initializations
    seeds    = randperm(rng, n_iter * (K_max - K_min + 1))
    grid_nmf = Array{Tuple{Matrix{T}, NMF}, 1}()

    for (i, K) in enumerate(K_min:K_max)
        for j in 1:n_iter
            V_pert = VNMF._get_perturbed_matrix(V, pert_meth, pert_dist, rng)
            nmf_K = copy(nmf)
            nmf_K.global_params.rank = K
            nmf_K.global_params.seed = seeds[(i-1) * n_iter + j]
            push!(grid_nmf, (V_pert, nmf_K))
        end
    end

    #### run in parallel
    grid_nmf_results = @showprogress pmap(grid_nmf) do g
        VNMF._rank_by_stability_one(g)
    end
        
    #### get reconstruction and stability results by rank
    grid_V_pert  = (x -> x[1]).(grid_nmf)
    grid_rank    = (x -> x[2].global_params.rank).(grid_nmf)

    list_rank = Vector{Int64}()
    list_clu_results = Vector{RSCluNMFResults{T}}()
    list_fid_results = Vector{RSFidNMFResults{T}}()
    list_stab_avg    = Vector{Float64}()
    list_stab_std    = Vector{Float64}()
    list_fid_avg     = Vector{Float64}()
    list_fid_std     = Vector{Float64}()

    for rank = K_min:K_max
        clu_results = evaluate_stability(grid_nmf_results[grid_rank .== rank], dist_func=CosineDist(), rng=rng)
        fid_results = evaluate_fidelity(grid_nmf_results[grid_rank .== rank], grid_V_pert[grid_rank .== rank])

        push!(list_rank, rank)
        push!(list_clu_results, clu_results)
        push!(list_fid_results, fid_results)

        push!(list_stab_avg, mean(clu_results.stab_avg))
        push!(list_stab_std, std(clu_results.stab_avg))
        push!(list_fid_avg, mean(fid_results.list_fid))
        push!(list_fid_std, std(fid_results.list_fid))
    end

    df_metrics = DataFrame(
        rank     = list_rank,
        stab_avg = list_stab_avg,
        stab_std = list_stab_std,
        fid_avg  = list_fid_avg,
        fid_std  = list_fid_std,
       )

    RSNMFResults{T}(
        grid_nmf_results, 
        df_metrics,
        list_clu_results, 
        list_fid_results
    )
end
