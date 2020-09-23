"""
Created on Mon Apr 13 2020

@author: Yoann Pradat

Demo of Fevotte and Idier Multiplicative Update NMF algorithms.
"""

include("../../Users/ypradat/Documents/projects/dev_ypradat/fac/nmf/vnmf/src/VNMF.jl")
using .VNMF
using LaTeXStrings
using Plots
using Random

gr() # set the backend to GR

#### # Fevotte-Idier NMF algorithms
#### #################################################################################################################

function run_fi_nmf(V::Matrix{T}, W_init::Matrix{T}, H_init::Matrix{T}, params::Params, local_params::FIParams) where T<:Real
    #### maximization-minimization algorithm
    println("running MM ...")
    res_MM = nmf_MM(
        V, 
        W_init,
        H_init,
        params,
        local_params,
    )
    println("MM over")

    #### heuristic algorithm
    println("running  H...")
    res_H = nmf_H(
        V, 
        W_init,
        H_init,
        params,
        local_params,
    )
    println("H over")

    #### maximization-equalization algorithm
    println("running ME ...")
    res_ME = nmf_ME(
        V, 
        W_init,
        H_init,
        params,
        local_params,
    )
    println("ME over")

    res_MM, res_H, res_ME
end

function plot_cost(res::Array{NMFResults{T},1}, lab::Array{String,1}, ls::Array{Symbol,1}, fn::String; cst::Float64=1e-12) where T <: Real
    n_res = length(res)
    n_lab = length(lab)
    n_ls  = length(ls)

    n_res == n_lab || throw(ArgumentError("array lab must have same size as array res"))
    n_res == n_ls || throw(ArgumentError("array ls must have same size as array res"))

    p_obj = plot(dpi=100, size=(1000, 400))

    for n = 1:n_res
        #### plot cost for each iter
        cost = res[n].metrics.absolute_cost[1:res[n].metrics.cur_iter] .+ cst
        plot!(
            p_obj,
            cost,
            label = lab[n],
            ls    = ls[n],
            lw    = 1,
            color = "black",
        )
    end

    #### plot settings
    plot!(
        p_obj,
        xlabel = "",
        xscale = :log,
        ylabel = "",
        yscale = :log,
    )

    #### save plot
    savefig(p_obj, fn)
    println("plot saved at $fn")
end

#### simulate matrice for testing NMF
F = 10
N = 25
K = 5
rng = MersenneTwister(123) 

#### truncated-normal random matrices
V = abs.(randn(rng, Float64, F, K)) * abs.(randn(rng,Float64, K, N))
W_init = abs.(randn(rng, Float64, F, K)) + ones(Float64, F, K)
H_init = abs.(randn(rng, Float64, K, N)) + ones(Float64, K, N)

params = Params(
    max_iter      = convert(Int, 1e4),
    stopping_crit = :none,
    stopping_tol  = 1e-6,
    verbose       = true,
)

####
#### params for β = 0.5, no regularization
####

local_params = FIParams(
    β             = 0.5,
    scale_W       = true,
    α_H           = 0,
    α_W           = 0,
    θ             = 0.95,
)

#### run
res_MM, res_H, res_ME = run_fi_nmf(V, W_init, H_init, params, local_params)

#### plot
res = [res_MM, res_H, res_ME]
lab = ["MM", "H", "ME"]
ls  = [:dash, :dot, :solid]
fn  = "../plot/nmf_MM_H_ME_$(local_params.β)_$(local_params.α_W)_$(local_params.α_H).pdf"
plot_cost(res, lab, ls, fn)

####
#### params for β = 1.5, l1 regularization
####
#
local_params = FIParams(
    β             = 1.5,
    scale_W       = true,
    α_H           = 10,
    α_W           = 0,
    l₁ratio_H     = 1,
    l₁ratio_W     = 1,
    θ             = 0.95,
)

#### run
res_MM, res_H, res_ME = run_fi_nmf(V, W_init, H_init, params, local_params)

#### plot
res = [res_MM, res_H, res_ME]
lab = ["MM", "H", "ME"]
ls  = [:dash, :dot, :solid]
fn  = "../plot/nmf_MM_H_ME_$(local_params.β)_$(local_params.α_W)_$(local_params.α_H)_ratios_$(local_params.l₁ratio_H)_$(local_params.l₁ratio_W).pdf"
plot_cost(res, lab, ls, fn)
