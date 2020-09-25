"""
Created on Fri May 28 2020

@author: Yoann Pradat

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Algorithms for clustering data.
"""

"""
    build_conn(labels)

Build the connectivity matrix C of a list of labels of size N. C has size N x N and C_ij is 1 if i and j are have the
same label, 0 otherwise.
"""
function build_conn(labels::Array{Int64,1})
    indices_1 = repeat(transpose(labels), length(labels), 1)
    indices_2 = repeat(labels, 1, length(labels))
    convert(Matrix{Float64}, indices_1 .== indices_2)
end

"""
    _kmeans_cost(X, centroids, labels, dist_func)

Auxiliary function for kmeans_init.
"""
function _kmeans_cost(X::Matrix{T}, centroids::Matrix{T}, labels::Vector{Int64}, dist_func::SemiMetric) where T <: Real
    list_intra_dist = Array{Float64, 1}()

    for label in unique(labels)
        intra_dist = pairwise(dist_func, hcat(centroids[:,label], X[:, labels .== label]), dims=2)[1, 2:end]
        push!(list_intra_dist, mean(intra_dist))
    end

    mean(list_intra_dist)
end

"""
    kmeans_init(X, init, max_iter, dist_func, stopping_crit, stopping_tol, stopping_iter)

Run kmeans algorithm for a specified set of initial centroids. If `stopping_crit` is :dist, convergenced is reached when 
the maximum distance between the new centroids and the ref centroids remains below `stopping_tol` for `stopping_iter` 
consecutive iterations. If `stopping_crit` is :cost, convergence is reached  when  the relative change in the cost 
remains below `stopping_tol` for `stopping_iter` consecutive iterations.
"""
function kmeans_init(;X::Matrix{T}, init::Vector{Int64}, max_iter::Integer, dist_func::SemiMetric, stopping_crit::Symbol, stopping_tol::Real, stopping_iter::Real, verbose::Bool=false) where T <: Real
    n_clus        = length(init)
    n_tot         = size(X, 2)
    centroids_ref = rand(T, size(X,1), n_clus)
    labels_ref    = zeros(Int64, n_tot)
    cost_ref      = 1e9
    n_iter_tol    = 0

    #### initial centroids
    centroids     = X[:, init]

    for i_iter = 1:max_iter
        #### compute distance of each data point to each centroid
        dist_to_centroids = pairwise(dist_func, hcat(centroids, X), dims=2)
        dist_to_centroids = transpose(dist_to_centroids[1:n_clus, (n_clus+1):(n_clus+n_tot)])

        #### update cluster labels using the specific structure of the matrix X
        #### The matrix X is an assembly of submatrices, each with n_clus columns, and
        #### labels are assigned so that for each submatrix, each column is assigned
        #### to one distinct label.
        labels = zeros(Int64, n_tot)

        for j in randperm(n_clus)
            for i in 1:n_clus:n_tot
                i_range = i:(i+n_clus-1)
                i_index = i + argmin(dist_to_centroids[i_range, j]) - 1
                labels[i_index] = j
                dist_to_centroids[i_index, :] .= Inf
            end
        end

        # labels = vec((x -> x[2]).(argmin(dist_to_centroids, dims=2)))

        #### update cluster centroids
        for label = 1:n_clus
            centroids[:, label] = mean(X[:, labels .== label], dims=2)
        end

        #### compute max distance between new and ref centroids
        dist_max = maximum(diag(pairwise(dist_func, centroids_ref, centroids, dims=2)))

        #### compute new cost
        cost = _kmeans_cost(X, centroids, labels, dist_func)

        #### check convergence
        if stopping_crit == :cost
            if abs((cost_ref-cost)/cost_ref) < stopping_tol
                n_iter_tol += 1
            else 
                n_iter_tol = 0
                centroids_ref .= centroids
                labels_ref .= labels
                cost_ref = cost 
            end
        elseif stopping_crit == :dist
            if dist_max < stopping_tol
                n_iter_tol += 1
            else 
                n_iter_tol = 0
                centroids_ref .= centroids
                labels_ref .= labels
                cost_ref = cost 
            end
        else 
            throw(ArgumentError("Invalid `stopping_crit`"))
        end

        if n_iter_tol == stopping_iter
            if verbose
                @info "convergence at iter" i_iter
            end
            break
        end

        if i_iter == max_iter
            if verbose
                @info "maximum number of iterations reached" i_iter
            end
        end
    end

    centroids_ref, labels_ref, cost_ref
end

"""
    kmeans_cons(X, k_min, k_max, n_init, best_init, max_iter, dist_func, stopping_crit, stopping_tol, stopping_iter)

Build a consensus matrix from application of kmeans algorithm for a varying number of clusters. For each value of k, 
kmeans is run for multiple initializations. If `best_init` is true, only the initialization resulting in the lowest cost 
is used for building the consensus matrix for each value of k. Otherwise every run of kmeans is used for building the 
consensus matrix. The X matrix should be in d × n format with d n the number of observations of a d-dimension vector of 
characteristics.
"""
function kmeans_cons(;X::Matrix{T}, k_min::Integer, k_max::Integer, n_init::Integer, best_init::Bool, max_iter::Integer, dist_func::SemiMetric, stopping_crit::Symbol, stopping_tol::Real, stopping_iter::Real) where T <: Real
    list_conn = Array{Matrix{Float64}, 1}()

    for k = k_min:k_max
        list_cost = Array{Float64,1}()
        list_labels = Array{Array{Int64, 1}, 1}()

        for i_init = 1:n_init
            init = sample(1:size(X,2), k, replace=false)
            centroids, labels, cost = kmeans_init(
                X             = X,
                init          = init,
                max_iter      = max_iter,
                dist_func     = dist_func,
                stopping_crit = stopping_crit,
                stopping_tol  = stopping_tol,
                stopping_iter = stopping_iter
            )

            push!(list_cost, cost)
            push!(list_labels, labels)
        end

        if best_init
            labels_best = list_labels[argmin(list_cost)]
            conn = build_conn(labels_best)
            push!(list_conn, conn)
        else
            for labels in list_labels
                conn = build_conn(labels)
                push!(list_conn, conn)
            end
        end
    end

    mean(list_conn)
end


# using PyPlot
# using PyCall
# const plt = PyPlot
# const cm  = pyimport("matplotlib.cm")
# 
# py"""
# from sklearn import datasets
# 
# def noisy_circles(n_samples):
#     return datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
#     
# def noisy_moons(n_samples):
#     return datasets.make_moons(n_samples=n_samples, noise=.05)
# """
# 
# X_moons, y_moons = py"noisy_moons"(100)
# X_circs, y_circs = py"noisy_circles"(100)
# 
# fig, ax = plt.subplots(figsize=(14,10))
# ax.scatter(X_circs[y_circs .== 0, 1], X_circs[y_circs .== 0, 2], color="purple", label="true 0")
# ax.scatter(X_circs[y_circs .== 1, 1], X_circs[y_circs .== 1, 2], color="orange", label="true 0")
# 
# X = convert(Array{Float64, 2}, transpose(X_circs))
# 
# cons = kmeans_cons(
#     X             = X,
#     k_min         = 2,
#     k_max         = 15,
#     n_init        = 20,
#     best_init     = false,
#     max_iter      = 100,
#     dist_func     = Euclidean(),
#     stopping_crit = :dist,
#     stopping_tol  = 0.05,
#     stopping_iter = 100
# )
# 
# hc = hclust(1 .- cons, linkage=:average, branchorder=:r)
# labels = cutree(hc, k=2)
# 
# fig, ax = plt.subplots(figsize=(14,10))
# cmap = cm.get_cmap("tab20")
# for label in unique(labels)
#     ax.scatter(X[1, labels .== label], X[2, labels .== label], color=cmap(label))
# end
# 
# plt.close()
