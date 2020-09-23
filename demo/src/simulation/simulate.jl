"""
Created on Fri 15 May 2020

@author: Yoann Pradat

Functions for simulation different matrices for the factorization V =  WH + E
"""

using Combinatorics
using LinearAlgebra

function scale_col(X; p=1)
    norms_X = mapslices(x -> norm(x, p), X, dims=1)
    X .* repeat(norms_X .^ -1, size(X, 1), 1)
end

function random_fixed_total(n::Integer, total::Integer, rng::AbstractRNG)
    μ = total÷n
    T = zeros(Integer, n)
    T[1] = total

    while sum(T[1:n-1]) >= total
        T[1:n-1] .= rand(rng, 1:2μ,  n-1)
    end
    
    T[n] = total - sum(T[1:n-1])

    T
end

"""
    simulate_X_clustered_a(K, N, n_clusters, dist_X, rng)

Exaclty one cluster of rows is active in one cluster of columns. Number of clusters of rows is equal to the number of 
clusters of columns.
"""
function simulate_X_clustered_a(K::Integer, N::Integer, n_clusters::Integer, dist_X::Distribution, rng::AbstractRNG)
    X = zeros(Float64, K,N)

    lims_K = cumsum(random_fixed_total(n_clusters, K, rng))
    lims_N = cumsum(random_fixed_total(n_clusters, N, rng))
    pushfirst!(lims_K, 0)
    pushfirst!(lims_N, 0)

    for i = 1:n_clusters
        range_K = (lims_K[i]+1):lims_K[i+1]
        range_N = (lims_N[i]+1):lims_N[i+1]
        X[range_K, range_N] .= rand(rng, dist_X, length(range_K), length(range_N))
    end

    X, lims_K, lims_N, Matrix{Int}(I,n_clusters, n_clusters)
end


# """
#     simulate_X_clustered_b(K, N, n_clusters, dist_X, rng)
# 
# One or more clusters of rows are active in one cluster of columns. The number of clusters of rows is equal to the number
# of clusters of columns.
# """
# function simulate_X_clustered_b(K::Integer, N::Integer, n_clusters::Integer, dist_X::Distribution, rng::AbstractRNG)
#     X = zeros(Float64, K,N)
# 
#     lims_K = cumsum(random_fixed_total(n_clusters, K, rng))
#     lims_N = cumsum(random_fixed_total(n_clusters, N, rng))
#     pushfirst!(lims_K, 0)
#     pushfirst!(lims_N, 0)
# 
#     active_clusters = zeros(Int, n_clusters, n_clusters)
#     combinations_K  = collect(combinations(1:n_clusters))
#     randperm_comb_K = randperm(rng, 2^n_clusters-1)
#     for j = 1:n_clusters
#         active_clusters[combinations_K[randperm_comb_K[j]], j] .= 1
#     end
# 
#     for j = 1:n_clusters
#         range_N = (lims_N[j]+1):lims_N[j+1]
#         for i in 1:n_clusters
#             range_K = (lims_K[i]+1):lims_K[i+1]
#             if active_clusters[i,j] == 1
#                 X[range_K, range_N] .= rand(rng, dist_X, length(range_K), length(range_N))
#             end
#         end
#     end
# 
#     X, lims_K, lims_N, active_clusters
# end

"""
    simulate_X_clustered_b(K, N, n_clusters_K, n_clusters_N, dist_X, rng)

One or more clusters of rows are active in one cluster of columns. Number of clusters of rows is inferior or equal to 
the number of clusters of columns.
"""
function simulate_X_clustered_b(K::Integer, N::Integer, n_clusters_K::Integer, n_clusters_N::Integer, dist_X::Distribution, rng::AbstractRNG)
    X = zeros(Float64, K, N)

    n_clusters_K <= n_clusters_N || throw(ArgumentError("Please choose n_clusters_K <= n_clusters_N"))

    lims_K = cumsum(random_fixed_total(n_clusters_K, K, rng))
    lims_N = cumsum(random_fixed_total(n_clusters_N, N, rng))
    pushfirst!(lims_K, 0)
    pushfirst!(lims_N, 0)

    active_clusters = zeros(Int, n_clusters_K, n_clusters_N)
    combinations_K  = collect(combinations(1:n_clusters_K))
    randperm_comb_K = randperm(rng, 2^n_clusters_K-1)
    for j = 1:n_clusters_N
        active_clusters[combinations_K[randperm_comb_K[j]], j] .= 1
    end

    for j = 1:n_clusters_N
        range_N = (lims_N[j]+1):lims_N[j+1]
        for i in 1:n_clusters_K
            range_K = (lims_K[i]+1):lims_K[i+1]
            if active_clusters[i,j] == 1
                X[range_K, range_N] .= rand(rng, dist_X, length(range_K), length(range_N))
            end
        end
    end

    X, lims_K, lims_N, active_clusters
end

"""
    simulate_WH_clustered(; F, N, K, n_clusters_K, n_clusters_N, dist_W, dist_H, H_cluster_mode, rng)

Simulate W and H with clustering on H.
"""
function simulate_WH_clustered(;F::Integer, N::Integer, K::Integer, n_clusters_K::Integer=0, n_clusters_N::Integer, dist_W::Distribution, dist_H::Distribution, H_cluster_mode::Symbol, rng::AbstractRNG)
    K = K

    W = rand(rng, dist_W, F, K)
    W = scale_col(W, p=1)

    if H_cluster_mode == :a
        H, lims_K, lims_N, active_clusters = simulate_X_clustered_a(K, N, n_clusters_N, dist_H, rng)
    elseif H_cluster_mode == :b
        H, lims_K, lims_N, active_clusters = simulate_X_clustered_c(K, N, n_clusters_K, n_clusters_N, dist_H, rng)
    end

    W, H, lims_K, lims_N, active_clusters
end

"""
    simulate_WH(F, N; n_factors, dist_W, dist_H, rng)

Simulate W and H.
"""
function simulate_WH(;F::Integer, N::Integer, n_factors::Integer, dist_W::Distribution, dist_H::Distribution, rng::AbstractRNG)
    K = n_factors

    W = rand(rng, dist_W, F, K)
    W = scale_col(W, p=1)

    H = rand(rng, dist_H, K, N)

    W, H
end
