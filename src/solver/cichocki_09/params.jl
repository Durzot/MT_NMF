"""
Created on Wed Apr 22 2020

@author: Yoann Pradat

Parameters for nmf algorithms in the subfolder.
"""

"""
    CParams([,α, max_iter, stopping_crit, stopping_tol, α_H, α_W, l₁ratio_H, l₁ratio_W, μ_H, μ_W, α_sparse_H, 
    α_sparse_W, ω, verbose])

Parameters passed to `nmf_α`.

# Arguments
- `α::Real`               : parameter of the α-divergence
- `max_iter::Integer`     : maximum number of iterations (in main procedure)
- `stopping_crit::Symbol` : criterion for stopping iterations. One of :none or :cost
- `stopping_tol::Real`    : tolerance for the stopping crit
- `scale_W::Bool`         : scale the components of W to have unit l₁ norm
- `α_H::Real`             : constant that multiplies the regularization term on H
- `α_W::Real`             : constant that multiplies the regularization term on W
- `l₁ratio_H::Real`       : l₁/l₂ regularization balance for H (in [0; 1]). 1 is l₁ only, 0 is l₂ only
- `l₁ratio_W::Real`       : l₁/l₂ regularization balance for W (in [0; 1]). 1 is l₁ only, 0 is l₂ only
- `μ_H::Real`             : constant that multiplies the orthogonalisation term on H
- `μ_W::Real`             : constant that multiplies the orthogonalisation term on W
- `α_sparse_H::Real`      : (1+α_sparse_H) is used as global exponent in the update of H
- `α_sparse_W::Real`      : (1+α_sparse_W) is used as global exponent in the update of W
- `ω::Real`               : overrelaxation parameter, typically in (0,2)
- `verbose::Bool`         : whether to show intermediate information
"""
mutable struct CParams 
    α::Real
    max_iter::Integer
    stopping_crit::Symbol
    stopping_tol::Real
    α_H::Real
    α_W::Real
    l₁ratio_H::Real
    l₁ratio_W::Real
    μ_H::Real
    μ_W::Real
    α_sparse_H::Real
    α_sparse_W::Real
    ω::Real
    verbose::Bool

    function CParams(;
        α::Real               = 2,
        max_iter::Integer     = 1000,
        stopping_crit::Symbol = :none,
        stopping_tol::Real    = 1e-5,
        α_H::Real             = 0,
        α_W::Real             = 0,
        l₁ratio_H::Real       = 0,
        l₁ratio_W::Real       = 0,
        μ_H::Real             = 0,
        μ_W::Real             = 0,
        α_sparse_H::Real      = 0,
        α_sparse_W::Real      = 0,
        ω::Real               = 1,
        verbose::Bool         = false)

        stopping_criteria = [:none, :cost]

        valid_iter = max_iter > 1
        valid_stopping_crit = stopping_crit in stopping_criteria
        valid_stopping_tol = stopping_tol > 0
        valid_α_H = α_H >= 0
        valid_α_W = α_W >= 0
        valid_l₁ratio_H = l₁ratio_H >= 0 && l₁ratio_H <= 1
        valid_l₁ratio_W = l₁ratio_W >= 0 && l₁ratio_W <= 1
        valid_μ_H = μ_H >= 0
        valid_μ_W = μ_W >= 0
        valid_α_sparse_H = α_sparse_H >= 0
        valid_α_sparse_W = α_sparse_W >= 0
        valid_ω = ω > 0

        valid_iter  || throw(ArgumentError("max_iter must be at least 2"))
        valid_stopping_crit || throw(ArgumentError(string("stopping_crit should be one of ", stopping_criteria)))
        valid_stopping_tol || throw(ArgumentError("stopping_tol must be > 0"))
        valid_α_H || throw(ArgumentError("α_H must be > 0"))
        valid_α_W || throw(ArgumentError("α_W must be > 0"))
        valid_l₁ratio_H || throw(ArgumentError("l₁ratio_H must be in [0,1]"))
        valid_l₁ratio_W || throw(ArgumentError("l₁ratio_W must be in [0,1]"))
        valid_μ_H || throw(ArgumentError("μ_H must be > 0"))
        valid_μ_W || throw(ArgumentError("μ_W must be > 0"))
        valid_α_sparse_H || throw(ArgumentError("α_sparse_H must be > 0"))
        valid_α_sparse_W || throw(ArgumentError("α_sparse_W must be > 0"))
        valid_ω || throw(ArgumentError("ω must be > 0"))

        new(α, max_iter, stopping_crit, stopping_tol, α_H, α_W, l₁ratio_H, l₁ratio_W, μ_H, μ_W, α_sparse_H, α_sparse_W, ω, verbose)
    end
end
