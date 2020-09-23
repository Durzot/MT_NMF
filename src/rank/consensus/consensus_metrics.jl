"""
Created on Sun Apr 26 2020

@author: Yoann Pradat

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Code for the functions cophcorr, dispersion and silhouette.
"""

"""
    _build_lists_ancestors!(list_nodes, lists_ancestors, node, list_ancestors, merges)

Helper recursive function for `_get_lists_ancestors`. Recursively go through a binary tree to get the list of ancestors 
of each node.
"""
function _build_lists_ancestors!(list_nodes::Vector{Int64}, lists_ancestors::Vector{Vector{Int64}}, node::Int64, list_ancestors::Vector{Int64}, merges::Array{Int64,2})
    if node < 0
        push!(list_nodes, node)
        push!(lists_ancestors, list_ancestors)

        nothing
    else
        push!(list_ancestors, node)

        node_left = merges[node,1]
        node_right = merges[node,2]
        
        #### recursive search
        _build_lists_ancestors!(list_nodes, lists_ancestors, node_left, list_ancestors[:], merges)
        _build_lists_ancestors!(list_nodes, lists_ancestors, node_right, list_ancestors[:], merges)
    end
end

"""
    _get_lists_ancestors(merges)

Helper function for `_cophenetic_distances`. Retrieve the list of ancestors of each node in a binary tree.
"""
function _get_lists_ancestors(merges::Array{Int64, 2})
    n = size(merges, 1)
    list_nodes      = Vector{Int64}()
    lists_ancestors = Vector{Vector{Int64}}()

    node_left = merges[n,1]
    node_right = merges[n,2]
    list_ancestors  = Vector{Int64}()

    _build_lists_ancestors!(list_nodes, lists_ancestors, node_left, list_ancestors[:], merges)
    _build_lists_ancestors!(list_nodes, lists_ancestors, node_right, list_ancestors[:], merges)

    list_nodes, lists_ancestors
end

"""
    _get_lowest_common_ancestor(ancestors_a, ancestors_b)

Find lowest common element from two vectors ordered in decreasing order.
"""
function _get_lowest_common_ancestor(ancestors_a::Vector{Int64}, ancestors_b::Vector{Int64})
    n_a = length(ancestors_a)
    n_b = length(ancestors_b)

    if n_a == 0 || n_b == 0
        #### -1 to mean the tree's root node
        return -1
    end

    #### iterate backwards
    for i = n_a:-1:1
        for j = n_b:-1:1
            if ancestors_b[j] > ancestors_a[i]
                break
            elseif ancestors_b[j] == ancestors_a[i]
                return ancestors_a[i]
            end
        end
    end

    #### no common ancestor was found
    return -1
end

"""
    _compute_cophenetic_distances(merges, heights)

Compute matrix of cophenetic distances from the array of merges and the array of heights as produced in output of the hclust function.
"""
function _compute_cophenetic_distances(merges::Array{Int64, 2}, heights::Array{Float64,1})
    #### get the list of ancestors for each node
    list_nodes, lists_ancestors = _get_lists_ancestors(merges)

    #### init matrix of distances
    n = length(list_nodes)
    D = zeros(Float64, n, n)

    for i = 2:n
        for j=1:i-1
            node_i = -list_nodes[i]
            node_j = -list_nodes[j]

            lca = _get_lowest_common_ancestor(lists_ancestors[i], lists_ancestors[j])

            if lca == -1
                D[node_i, node_j] = heights[n-1]
            else
                D[node_i, node_j] = heights[lca]
            end
        end
    end

    D + transpose(D)
end


"""
    cons_cophenetic(cons, hc)

Computes the cophenetic correlation coefficient from the consensus matrix and hierarchical clustering result.
"""
function cons_cophenetic(cons::Matrix{T}, hc::Hclust{T}) where T <: Real
    #### matrix of distances
    D_con = 1 .- cons

    #### compute matrix of cophenetic distances
    D_cop = _compute_cophenetic_distances(hc.merges, hc.heights)

    #### compute cophenetic correlation
    D_con_lower_part = D_con[[x > y ? true : false for x in 1:size(D_con,1), y in 1:size(D_con,2)]]
    D_cop_lower_part = D_cop[[x > y ? true : false for x in 1:size(D_cop,1), y in 1:size(D_cop,2)]]

    #### Pearson correlation
    cor(D_con_lower_part, D_cop_lower_part)
end

"""
    cons_dispersion(cons)

Compute the dispersion coefficient for the consensus matrix.
"""
function cons_dispersion(cons::Matrix{T}) where T <: Real
    mean(4 * (cons .- 0.5) .^ 2)
end

"""
    cons_avg_silhouette(cons, hc, rank)

Compute the average silhouette width from the hierarchical clustering of the consensus matrix cut with the number
of clusters equal to the rank.
"""
function cons_avg_silhouette(cons::Matrix{T}, hc::Hclust{T}, rank::Integer) where T <: Real
    idxs = cutree(hc, k=rank)
    silh = silhouettes(idxs, 1 .- cons)
    silh_avg = 0

    for lab in 1:rank
        silh_avg += mean(silh[idxs .== lab])
    end

    silh_avg/rank
end

"""
    matr_avg_col_sparseness(X)

Compute the average sparseness of the columns of the input matrix using Hoyer's definition of sparseness.
"""
function matr_avg_col_sparseness(X::Matrix{T}) where T <: Real
    mean(mapslices(x -> (sqrt(length(x)) - norm(x, 1)/norm(x,2))/(sqrt(length(x))-1), X; dims=1))
end

"""
    matr_avg_row_sparseness(X)

Compute the average sparseness of the rows of the input matrix using Hoyer's definition of sparseness.
"""
function matr_avg_row_sparseness(X::Matrix{T}) where T <: Real
    mean(mapslices(x -> (sqrt(length(x)) - norm(x, 1)/norm(x,2))/(sqrt(length(x))-1), X; dims=2))
end
