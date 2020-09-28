"""
Created on Tue May 19 2020

@author: Yoann Pradat

Functions used for assessing convergence of NMF algorithms.
"""

"""
    _update_metrics!(metrics, cur_iter, V, W, H)

Helper function for tracking cost evolution in NMF algorithms.
"""
function _update_metrics!(metrics::NMFMetrics, V::Matrix{T}, W::Matrix{T}, H::Matrix{T}, stopping_crit::Symbol) where T <: Real
    #### compute connectivity matrices only if required
    if stopping_crit == :conn
        prv_conn = metrics.cur_conn
        cur_conn  = _connectivity_matrix(H)
        push!(metrics.unchange_conn, prv_conn == cur_conn)
        metrics.cur_conn = cur_conn
    end

    #### compute indices and cost 
    prv_cost = metrics.cur_cost

    #### compute new cost and new connectivity matrix
    cur_cost  = metrics.func_cost(V, W, H) 
    cur_divg  = metrics.func_divg(V, W, H) 
    rel_cost  = abs((cur_cost - prv_cost)/prv_cost)

    #### update metrics
    push!(metrics.absolute_cost, cur_cost)
    push!(metrics.absolute_divg, cur_divg)
    push!(metrics.relative_cost, rel_cost)

    metrics.cur_iter += 1
    metrics.cur_cost = cur_cost
    metrics.cur_divg = cur_divg
end


"""
    _build_conn(labels)

Helper function for `_connectivity_matrix`.
"""
function _build_conn(labels::Array{Int64,1})
    indices_1 = repeat(transpose(labels), length(labels), 1)
    indices_2 = repeat(labels, 1, length(labels))

    convert(Matrix{Float64}, indices_1 .== indices_2)
end

"""
    _connectivity_matrix(H)

Build connectivity matrix from H in which rows are latent factors and columns individuals.
"""
function _connectivity_matrix(H::Matrix{T}; clustering::Symbol=:max) where T <: Real
    if clustering == :max
        labels = argmax(H, dims=1)
        labels = (c -> c[1]).(labels)
        labels = vec(labels)
    elseif clustering == :kmeans
        throw(ArgumentError("clustering method not implemented")) 
    end
    _build_conn(labels)
end

"""
    _check_stopping_crit(metrics, global_params)

Helper function for checking convergence in NMF algorithms.
"""
function _check_stopping_crit(metrics::NMFMetrics, global_params::AbstractNMFParams)
    stopping_crit_met = false
    
    if global_params.stopping_crit == :conn
        #### stop when the connectivity matrix remains constant over successive iterations.
        if metrics.cur_iter > global_params.stopping_iter
            range_iter = (metrics.cur_iter-global_params.stopping_iter):(metrics.cur_iter-1)
            stopping_crit_met = all(metrics.unchange_conn[range_iter])
        end
        
    elseif global_params.stopping_crit == :rel_cost
        #### stop when relative change in cost is below tolerance over successive iterations
        if metrics.cur_iter > global_params.stopping_iter
            range_iter = (metrics.cur_iter-global_params.stopping_iter):(metrics.cur_iter-1)
            stopping_crit_met = maximum(metrics.relative_cost[range_iter]) < global_params.stopping_tol
        end

    elseif global_params.stopping_crit == :cost
        #### stop when absolute change in cost is below tolerance over successive iterations
        if metrics.cur_iter > global_params.stopping_iter
            range_iter = (metrics.cur_iter-global_params.stopping_iter):(metrics.cur_iter-1)
            stopping_crit_met = maximum(metrics.absolute_cost[range_iter]) < global_params.stopping_tol
        end
    end

    stopping_crit_met || metrics.cur_iter == global_params.max_iter
end

"""
    _check_sanity_iter(W, H)

Helper function for checking iteration sanity in NMF algorithms.
"""

function _check_sanity_iter(W::Matrix{T}, H::Matrix{T}) where T <: Real
    sane = true
    cs_W = sum(W, dims=1)
    cs_H = sum(H, dims=1)
    rs_W = sum(W, dims=2)
    rs_H = sum(H, dims=2)
    
    if any(cs_W .== 0)
        @warn "Sanity check: One or more columns of W are null vectors, leading to division by zero in the update of H and W. Try reducing regularization if any, removing the scaling of W or modifying the initiazation method. The problem might as well be ill-posed."
        sane = false
    end

    if any(rs_W .== 0)
        @warn "Sanity check: One or more rows of W are null vectors, leading to division by zero in the update of H and W. Try reducing regularization if any, removing the scaling of W or modifying the initiazation method. The problem might as well be ill-posed."
        sane = false
    end

    if any(cs_H .== 0)
        @warn "Sanity check: One or more columns of H are null vectors, leading to division by zero in the update of H and W. Try reducing regularization if any, removing the scaling of W or modifying the initiazation method. The problem might as well be ill-posed."
        sane = false
    end

    if any(rs_H .== 0)
        @warn "Sanity check: One or more rows of H are null vectors, leading to division by zero in the update of H and W. Try reducing regularization if any, removing the scaling of W or modifying the initiazation method. The problem might as well be ill-posed."
        sane = false
    end

    sane
end
