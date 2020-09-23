"""
Created on Fri Jun 19 2020

@author: Yoann Pradat

Test clustering functions of VNMF module.
"""

@testset "kmeans_init" begin
    centroids = rand(96, 16)
    X = repeat(centroids, 1, 400)
    init = [i for i in 1:16]
    labels = repeat(init, 400)

    centroids_ref, labels_ref, cost_ref = VNMF.kmeans_init(
        X = X,
        init = init,
        max_iter = 100,
        dist_func = CosineDist(),
        stopping_crit = :cost,
        stopping_tol  = 5e-2,
        stopping_iter = 10,
        verbose = true
    )

    @test centroids_ref ≈ centroids atol = 1e-9
    @test cost_ref ≈ 0 atol = 1e-9
    @test labels_ref == labels
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
