"""
Created on Wed Apr 22 2020

@author: Yoann Pradat

Struct for specifying parameters of FI NMF algorithms.
"""

abstract type AbstractFIParams <: AbstractNMFParams end

"""
    FIParams([,max_iter, stopping_crit, stopping_tol, verbose, β, scale_W, α_H, α_W, l₁ratio_H, l₁ratio_W])

Local parameters passed to `nmf_MM`, `nmf_H` and `nmf_ME`.

# Arguments
- `β::Real`               : parameter of the β-divergence
- `scale_W::Bool`         : scale the components of W to have unit l₁ norm
- `α_H::Real`             : constant that multiplies the regularization term on H
- `α_W::Real`             : constant that multiplies the regularization term on W
- `l₁ratio_H::Real`       : l₁/l₂ regularization balance for H (in [0; 1]). 1 is l₁ only, 0 is l₂ only
- `l₁ratio_W::Real`       : l₁/l₂ regularization balance for W (in [0; 1]). 1 is l₁ only, 0 is l₂ only
- `θ::Real`               : θ for mixed update between ME and H. Used only in `nmf_ME`.
"""
mutable struct FIParams <: AbstractFIParams
    β::Real
    scale_W::Bool
    α_H::Real
    α_W::Real
    l₁ratio_H::Real
    l₁ratio_W::Real
    θ::Real

    function FIParams(;
        β::Real               = 2,
        scale_W::Bool         = true,
        α_H::Real             = 0,
        α_W::Real             = 0,
        l₁ratio_H::Real       = 0,
        l₁ratio_W::Real       = 0,
        θ::Real               = 0.95)

        valid_α_H = α_H >= 0
        valid_α_W = α_W >= 0
        valid_l₁ratio_H = l₁ratio_H >= 0 && l₁ratio_H <= 1
        valid_l₁ratio_W = l₁ratio_W >= 0 && l₁ratio_W <= 1
        valid_θ = θ >= 0 && θ <= 1

        valid_α_H || throw(ArgumentError("α_H must be > 0"))
        valid_α_W || throw(ArgumentError("α_W must be > 0"))
        valid_l₁ratio_H || throw(ArgumentError("l₁ratio_H must be in [0,1]"))
        valid_l₁ratio_W || throw(ArgumentError("l₁ratio_W must be in [0,1]"))
        valid_θ  || throw(ArgumentError("θ must be between 0 and 1"))

        new(β, scale_W, α_H, α_W, l₁ratio_H, l₁ratio_W, θ)
    end
end
