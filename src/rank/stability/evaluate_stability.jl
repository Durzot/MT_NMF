"""
Created on Fri May 04 2020

@author: Yoann Pradat

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Algorithm for evaluating factors stability in rank selection method in NMF based on average stability of latent factor 
across perturbed versions of the input matrix V.
"""

abstract type AbstractRSCluNMFResults <: AbstractNMFResults end

"""
    RSCluNMFResults(cluster_W_avg, cluster_W_std, cluster_H_avg, cluster_H_std, cluster_indices, cluster_stab_avg, cluster_intra_dist)

Struct for storing results returned by the clustering algorithm `evaluate_stability`.
"""
mutable struct RSCluNMFResults{T} <: AbstractRSCluNMFResults
    W_avg::Matrix{T}   
    W_std::Matrix{T}
    H_avg::Matrix{T}   
    H_std::Matrix{T}
    labels::Vector{Int64}
    stab_avg::Vector{Float64}
    stab_std::Vector{Float64}

    function RSCluNMFResults{T}(
        W_avg::Matrix{T},
        W_std::Matrix{T},
        H_avg::Matrix{T},  
        H_std::Matrix{T},
        labels::Vector{Int64},
        stab_avg::Vector{Float64},
        stab_std::Vector{Float64}) where T <: Real

        new{T}(W_avg, W_std, H_avg, H_std, labels, stab_avg, stab_std)
    end
end


"""
    evaluate_stability(list_nmf_results[, dist_func, n_init, max_iter, stopping_tol, stopping_iter, rng])

Evaluates the stability of multiple runs of a NMF factorisation by clustering in a as many clusters
as the rank of the factorisation and compute the average silhouette index within each cluster.

#Arguments
- list_nmf_results::Vector{NMFResults{T}} : vector containing results of each NMF run
- dist_func::SemiMetric : SemiMetric from the `Distances` library used for computing distances between vectors.
- rng::AbstractRNG : random number generator, for the random initialization of the centroids positions. 
- n_init::Integer : number of random initialization
- n_repl::Integer : maxium number of centroids update for each initialization.
"""

function evaluate_stability(list_nmf_results::Array{NMFResults{T},1}; dist_func::SemiMetric=CosineDist(), n_init::Integer=5, max_iter::Integer=100, stopping_crit::Symbol=:dist, stopping_tol::Real=0.005, stopping_iter::Integer=10, rng::AbstractRNG=MersenneTwister(0)) where T <: Real

    W_all = hcat([nmf_results.W for nmf_results in list_nmf_results]...)
    H_all = vcat([nmf_results.H for nmf_results in list_nmf_results]...)

    #### recover parameters of the algorithm
    n_nmf = length(list_nmf_results)
    n_tot = size(W_all, 2)
    n_fac = n_tot ÷ n_nmf

    list_centroids   = Array{Matrix{T}, 1}()
    list_labels      = Array{Array{Int64}, 1}()
    list_intra_dists = Array{Array{Float64,1}, 1}()
    list_cost        = Array{Float64,1}()

    perm_nmf = randperm(rng, n_nmf)

    for i_init = 1:min(n_init, n_nmf)
        i_iter = perm_nmf[i_init]
        init   = convert(Array{Int64,1}, ((i_iter-1) * n_fac + 1):(i_iter * n_fac))

        centroids, labels, cost = kmeans_init(
            X             = W_all,
            init          = init,
            max_iter      = max_iter,
            dist_func     = dist_func,
            stopping_crit = stopping_crit,
            stopping_tol  = stopping_tol,
            stopping_iter = stopping_iter
        )

        push!(list_centroids, centroids)
        push!(list_labels, labels)
        push!(list_cost, cost)
    end

    #### index of best initialization
    i_init_best = argmin(list_cost)

    centroids   = list_centroids[i_init_best]
    labels      = list_labels[i_init_best]

    #### compute silhouette indices per cluster
    if n_fac == 1
        stab_avg = [1.0]
        stab_std = [0.0]
    else
        stab_all = silhouettes(labels, pairwise(dist_func, W_all, dims=2))
        stab_avg = [mean(stab_all[findall(x -> x == label, labels)]) for label in sort(unique(labels))]
        stab_std = [std(stab_all[findall(x -> x == label, labels)]) for label in sort(unique(labels))]
    end

    #### rename labels in decreasing stab avg
    sorted_order = sortperm(stab_avg, rev=true)
    stab_avg = stab_avg[sorted_order]
    stab_std = stab_std[sorted_order]
    centroids = centroids[:,sorted_order]

    sorting_dict  = Dict( j => i for (i,j) in enumerate(sortperm(stab_avg, rev=true)))
    labels = map(x -> sorting_dict[x], labels)

    #### compute cluster standard deviations
    W_avg = zeros(Float64, size(centroids))
    W_std = zeros(Float64, size(W_avg))

    for label = 1:n_fac
        W_avg[:, label] = centroids[:, label]
        W_std[:, label] = std(W_all[:, labels .== label], dims=2)
    end
    
    #### compute cluster H avg and std
    H_avg = zeros(Float64, n_fac, size(H_all, 2))
    H_std = zeros(Float64, n_fac, size(H_all, 2))

    for label = 1:n_fac
        H_avg[label, :] = mean(H_all[labels .== label, :], dims = 1)
        H_std[label, :] = std(H_all[labels .== label, :], dims = 1)
    end

    RSCluNMFResults{T}(
        W_avg,
        W_std,
        H_avg,
        H_std,
        labels,
        stab_avg,
        stab_std
    )
end

#function evaluate_stability(list_nmf_results::Array{NMFResults{T},1}; dist_func::SemiMetric=CosineDist(), 
#    rng::AbstractRNG=MersenneTwister(0), n_init::Integer=5, n_repl::Integer=100) where T <: Real
#    #### constants
#    convergence_cutoff = 0.005
#    convergence_iter = 10  
#
#    W_all = hcat([nmf_results.W for nmf_results in list_nmf_results]...)
#    H_all = vcat([nmf_results.H for nmf_results in list_nmf_results]...)
#
#    #### recover parameters of the algorithm
#    n_clus = list_nmf_results[1].global_params.rank
#    n_iter = size(list_nmf_results, 1)
#    n_tot  = n_clus * n_iter
#
#    centroids_final = zeros(T, size(W_all, 1), n_clus)
#    cluster_indices = zeros(Int64, n_tot)
#    cluster_indices_final = zeros(Int64, n_tot)
#    cluster_intra_dist = zeros(Float64, n_iter, n_clus)
#    cluster_intra_dist_final = zeros(Float64, n_iter, n_clus)
#    min_cluster_intra_dist = Inf 
#
#    #### random permutation for selection of initial centroids
#    #### i.e random selection of one matrix W among the n_iter
#    #### W matrices.
#    perm_iter = randperm(rng, n_iter)
#
#    for i_init = 1:min(n_init, n_iter)
#        centroids_cur = list_nmf_results[perm_iter[i_init]].W
#        centroids_pre = rand(rng, T, size(centroids_cur))
#
#        count_i_repl = 0
#
#        for i_repl = 1:n_repl
#            dist_all = pairwise(dist_func, hcat(centroids_cur, W_all), dims=2)
#            centroids_dist = transpose(dist_all[1:n_clus, (n_clus+1):(n_clus+n_tot)])
#
#            #### update cluster labels
#            perm_n_clus = randperm(rng, n_clus)
#            for j_n_clus in perm_n_clus 
#                for i = 1:n_clus:n_tot
#                    i_range =  i:(i+n_clus-1)
#                    _, index_min = findmin(centroids_dist[i_range, j_n_clus], dims=1)
#                    centroids_dist[i_range[index_min],:] .= Inf 
#                    cluster_indices[i_range[index_min]] .= j_n_clus
#                end
#            end
#
#            #### update centroids
#            dist_max_to_new_centroids = 0
#            for j_n_clus = 1:n_clus
#                centroids_cur[:, j_n_clus] = mean(W_all[:, cluster_indices .== j_n_clus], dims=2)
#                dist_to_new_centroids = dist_func(centroids_cur[:, j_n_clus], centroids_pre[:, j_n_clus])
#                dist_max_to_new_centroids = max(dist_max_to_new_centroids, dist_to_new_centroids)
#            end
#
#            #### check convergence
#            if dist_max_to_new_centroids < convergence_cutoff
#                count_i_repl += 1
#            else 
#                count_i_repl = 0
#                centroids_pre .= centroids_cur
#            end
#
#            if count_i_repl == convergence_iter
#                break
#            end
#        end
#
#        #### update intra cluster distances
#        for j = 1:n_clus
#            cluster_intra_dist_one = pairwise(dist_func, hcat(centroids_cur[:,j], W_all[:, cluster_indices .== j]), dims=2)
#            cluster_intra_dist[:, j] = cluster_intra_dist_one[1, 2:size(cluster_intra_dist_one, 2)]
#        end
#
#        #### update final cluster centroids, indices and dist if lower
#        #### mean intra cluster distance is achieved
#        if min_cluster_intra_dist > mean(cluster_intra_dist)
#            centroids_final .= centroids_cur
#            cluster_indices_final .= cluster_indices
#            cluster_intra_dist_final .= cluster_intra_dist
#
#            min_cluster_intra_dist = mean(cluster_intra_dist)
#        end
#    end
#
#    centroids = centroids_final
#    cluster_indices = cluster_indices_final
#    cluster_intra_dist = cluster_intra_dist_final
#    
#    #### reorder according to mean intra cluster distances 
#    cluster_intra_dist_avg = vec(mean(cluster_intra_dist, dims=1))
#    cluster_intra_dist_avg_indices = sortperm(cluster_intra_dist_avg)
#
#    centroids = centroids[:, cluster_intra_dist_avg_indices]
#    cluster_intra_dist = cluster_intra_dist[:, cluster_intra_dist_avg_indices]
#    cluster_indices_new = cluster_indices[:]
#
#    for j = 1:n_clus
#        cluster_indices_new[cluster_indices .== cluster_intra_dist_avg_indices[j]] .= j
#    end
#    cluster_indices = cluster_indices_new[:]   
#
#    #### compute silhouette indices per cluster
#    stab_all = silhouettes(cluster_indices, pairwise(dist_func, W_all, dims=2))
#    cluster_stab_avg = zeros(Float64, n_clus)
#
#    for j = 1:n_clus
#        cluster_stab_avg[j] = mean(stab_all[cluster_indices .== j])
#    end
#
#    #### compute cluster standard deviations
#    cluster_W_avg = zeros(Float64, size(centroids))
#    cluster_W_std = zeros(Float64, size(cluster_W_avg))
#
#    for j = 1:n_clus
#        cluster_W_avg[:, j] = centroids[:, j]
#        cluster_W_std[:, j] = std(W_all[:, cluster_indices .== j], dims=2)
#    end
#    
#    #### compute cluster H avg and std
#    cluster_H_avg = zeros(Float64, n_clus, size(H_all, 2))
#    cluster_H_std = zeros(Float64, n_clus, size(H_all, 2))
#
#    for j = 1:n_clus
#        cluster_H_avg[j, :] = mean(H_all[cluster_indices .== j, :], dims = 1)
#        cluster_H_std[j, :] = std(H_all[cluster_indices .== j, :], dims = 1)
#    end
#
#    RSCluNMFResults{T}(
#        cluster_W_avg,
#        cluster_W_std,
#        cluster_H_avg,
#        cluster_H_std,
#        cluster_indices,
#        cluster_stab_avg,
#        cluster_intra_dist
#    )
#end


#mutable struct RSCluNMFResults{T} <: AbstractRSCluNMFResults
#    cluster_W_avg::Matrix{T}   
#    cluster_W_std::Matrix{T}
#    cluster_H_avg::Matrix{T}   
#    cluster_H_std::Matrix{T}
#    cluster_indices::Vector{Int64}
#    cluster_stab_avg::Vector{Float64}
#    cluster_intra_dist::Matrix{Float64}
#
#    function RSCluNMFResults{T}(
#        cluster_W_avg::Matrix{T},
#        cluster_W_std::Matrix{T},
#        cluster_H_avg::Matrix{T},  
#        cluster_H_std::Matrix{T},
#        cluster_indices::Vector{Int64},
#        cluster_stab_avg::Vector{Float64},
#        cluster_intra_dist::Matrix{Float64}) where T <: Real
#
#        new{T}(cluster_W_avg, cluster_W_std, cluster_H_avg, cluster_H_std, cluster_indices, cluster_stab_avg, cluster_intra_dist)
#    end
#end
