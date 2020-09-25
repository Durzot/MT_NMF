"""
Created on Fri Jun 19 2020

@author: Yoann Pradat

Test clustering functions.
"""

using Distances
using Test
using VariantsNMF

@testset "build_conn" begin
    labels = [1, 0, 1, 2, 1]
    conn_true = [
        [1 0 1 0 1]
        [0 1 0 0 0]
        [1 0 1 0 1]
        [0 0 0 1 0]
        [1 0 1 0 1]
       ]
    
    conn = VariantsNMF.build_conn(labels)
    conn = convert(Matrix{Int64}, conn)
    @test conn == conn_true 
end


@testset "kmeans_init" begin
    p_centroids = 96
    n_centroids = 16
    n_repeats = 400

    # 1. test on synthetic data where centroids in each repetition are identical
    # each cluster is then reduced to one point and the clustering is perfect
    centroids_true = rand(p_centroids, n_centroids)
    labels_true = repeat(collect(1:n_centroids), n_repeats)
    X = repeat(centroids_true, 1, n_repeats)

    # 1.1 stop when relative change in cost remains below 5e-2 over 10 iter
    centroids_ref, labels_ref, cost_ref = VariantsNMF.kmeans_init(
        X             = X,
        init          = collect(1:n_centroids),
        max_iter      = 100,
        dist_func     = CosineDist(),
        stopping_crit = :cost,
        stopping_tol  = 5e-2,
        stopping_iter = 10,
        verbose       = true
    )

    @test centroids_ref ≈ centroids_true atol = 1e-9
    @test cost_ref ≈ 0 atol = 1e-9
    @test labels_ref == labels_true

    # 1.2 stop when cosine distance between new centroids and ref centroids remains below 5e-2 for 10 iter
    centroids_ref, labels_ref, cost_ref = VariantsNMF.kmeans_init(
        X             = X,
        init          = collect(1:n_centroids),
        max_iter      = 100,
        dist_func     = CosineDist(),
        stopping_crit = :dist,
        stopping_tol  = 5e-2,
        stopping_iter = 10,
        verbose       = true
    )

    @test centroids_ref ≈ centroids_true atol = 1e-9
    @test cost_ref ≈ 0 atol = 1e-9
    @test labels_ref == labels_true

end

# using BenchmarkTools
# 
# centroids = rand(96, 16)
# X = repeat(centroids, 1, 400)
# init = [i for i in 1:16]
# labels = repeat(init, 400)
# dist_func = CosineDist()
# 
# @benchmark centroids_ref, labels_ref, cost_ref = VNMF.kmeans_init(
#     X = X,
#     init = init,
#     max_iter = 100,
#     dist_func = dist_func,
#     stopping_crit = :cost,
#     stopping_tol  = 5e-2,
#     stopping_iter = 10,
#     verbose = true
# )
# 
# @benchmark dist_to_centroids = pairwise($dist_func, hcat($centroids, $X), dims=2)
