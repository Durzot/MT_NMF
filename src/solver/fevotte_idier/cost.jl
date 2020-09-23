"""
Created on Sat Apr 15 2020

@author: Yoann Pradat

Cost function for nmf_MM, nmf_HH and nmf_ME.
"""

"""
    _β_cost(V, W, H, params)

Compute the cost function that is minimized by the NMF alogrithm running with parameters params. It adds up
the β_divergence and regularization terms.

See also: [`β_divergence`](@ref), [`FIParams`](@ref)
"""
function _β_cost(V::Matrix{T}, W::Matrix{T}, H::Matrix{T}, params::FIParams) where T <: Real
    regu_H = params.α_H * (params.l₁ratio_H * norm(H, 1) + (1. - params.l₁ratio_H) * norm(H, 2)) / length(H)
    regu_W = params.α_W * (params.l₁ratio_W * norm(W, 1) + (1. - params.l₁ratio_W) * norm(W, 2)) / length(W)
    β_div  = β_divergence(V, W*H, params.β)

    β_div + regu_H + regu_W
end
