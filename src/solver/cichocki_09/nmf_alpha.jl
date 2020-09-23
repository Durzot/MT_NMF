"""
Created on Wed Apr 22 2020

@author: Yoann Pradat

Implementation of the alpha NMF algorithm under various constraints as described in Chapter 3 of 
Andrzej CICHOCKI, et. al., Nonnegative Matrix and Tensor Factorizations. John Wiley & Sons, 2009.
"""

"""
    _normalize(X[, p])

Column-wise normalization.
"""
function _normalize(X::Matrix{T}; p::Integer=1) where T <: Real
    norms = sum(X .^ p, dims=1) .^ (1. / p) 
    X .* repeat(norms .^ -1, size(X, 1), 1)
end

"""
    _update_nmf_α!(V, V_a, W, H, params)

Update the H and W matrices in that order using the update rules using algorithm 3.3 page 147 
Andrzej CICHOCKI, et. al., Nonnegative Matrix and Tensor Factorizations. John Wiley & Sons, 2009.
"""
function _update_nmf_α!(V::Matrix{T}, V_a::Matrix{T}, W::Matrix{T}, H::Matrix{T}, params::CParams) where T <: Real
    #### compute regularization
    #### for J_H = ||H||₁, ∇J_H = J_{K,N} with J the matrix filled with ones
    #### for J_H = 0.5*||H||₂, ∇J_H = H
    regu_H = H -> params.α_H * (params.l₁ratio_H * ones(size(H)) + (1. - params.l₁ratio_H) * H) / length(H)
    regu_W = W -> params.α_W * (params.l₁ratio_W * ones(size(W)) + (1. - params.l₁ratio_W) * W) / length(W)

    #### orthonormalization term
    orth_H = H -> params.μ_H * (H - repeat(sum(H, dims=1), size(H)[1], 1))
    orth_W = W -> params.μ_W * (W - repeat(sum(W, dims=2), 1, size(W)[2]))

    if params.α == 0
        #### update H
        H .= (H .* exp.(params.ω * transpose(_normalize(W, p=1)) * log.(V ./ V_a))) .^ (1 + params.α_sparse_H)
        V_a .= W * H

        #### update W
        W .= (W .* exp.(params.ω * log.(V ./ V_a) * transpose(_normalize(H, p=1)))) .^ (1 + params.α_sparse_W)
        V_a .= W * H
    else         
        #### update H
        H .= (H .* (transpose(W) * ((V ./ (V_a .+ eps(T))) .^ params.α) - regu_H(H) + orth_H(H)) .^ (params.ω / params.α)) .^ 
        (1 + params.α_sparse_H)
        V_a .= W * H

        #### update W (incl. normalize columns of W)
        W .= (W .* (((V ./ (V_a .+ eps(T))) .^ params.α) * transpose(H) - regu_W(W) + orth_W(W)) .^ (params.ω / params.α)) .^ 
        (1 + params.α_sparse_W)
        W .= _normalize(W, p=1)
        V_a .= W * H
    end
end
 

"""
    nmf_α(V, W_init, H_init, params)

Run the NMF α algorithm using the set of input parameters. For the parameters, see [`CParams`](@ref)

# References
- Andrzej CICHOCKI, et. al., Nonnegative Matrix and Tensor Factorizations. John Wiley & Sons, 2009.
"""
function nmf_α(V::Matrix{T}, W_init::Matrix{T}, H_init::Matrix{T}, params::CParams) where T <: Real
    W       = W_init[:,:]
    H       = H_init[:,:]
    V_a     = W * H

    #### metrics and stopping_crit
    metrics            = NMFMetrics(V, W_init, H_init, _α_cost, params)
    stopping_crit_met  = false

    while !stopping_crit_met
        #### update V_a, W, H
        _update_nmf_α!(V, V_a, W, H, params)

        #### update metrics 
        _update_metrics!(metrics, V, W, H, params)

        #### check the stopping criterion
        stopping_crit_met = _check_stopping_crit(metrics, params)

        #### print intermediate if verbose
        if params.verbose && (metrics.cur_iter % (metrics.max_iter ÷ 10) == 0)
            @printf("iter %d/%d | cost: %.5g\n", metrics.cur_iter, metrics.max_iter, metrics.absolute_cost[metrics.cur_iter])
        end
    end

    #### convergence information
    if metrics.cur_iter == metrics.max_iter && params.stopping_crit != :none
        @warn "maximum number of iterations reached"
    end
    if metrics.cur_iter < metrics.max_iter
        @info "stopping criterion met at" metrics.cur_iter
    end

    NMFResults{T}(W, H, metrics, params)
end
