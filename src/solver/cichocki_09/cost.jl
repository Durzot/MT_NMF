"""
Created on Thu Apr 23 2020

@author: Yoann Pradat

Cost function for nmf_α
"""

"""
    _α_cost(V, W, H, params)

Compute the cost function that is minimized by the NMF alogrithm running with parameters params. It adds up
the α_divergence and regularization terms.

See also: [`α_divergence`](@ref), [`CParams`](@ref)
"""
function _α_cost(V::Matrix{T}, W::Matrix{T}, H::Matrix{T}, params::CParams) where T <: Real
    regu_H = params.α_H * (params.l₁ratio_H * norm(H, 1) + (1. - params.l₁ratio_H) * norm(H, 2)) / length(H)
    regu_W = params.α_W * (params.l₁ratio_W * norm(W, 1) + (1. - params.l₁ratio_W) * norm(W, 2)) / length(W)
    α_div  = α_divergence(V, W*H, params.α)

    α_div + regu_H + regu_W
end
