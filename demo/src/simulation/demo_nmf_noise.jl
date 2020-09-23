"""
Created on Mon Apr 13 2020

@author: Yoann Pradat

Code supporting presentation on NMF.
"""

include("../../../src/VNMF.jl")
using .VNMF
using Distributions
using Random
include("../util/plot_cost.jl")

gr() # set the backend to GR

#### # 1. SIMULATE DATA
#### # ################################################################################################################

#### simulate matrix for testing NMF
F = 10
N = 25
K = 5
rng = MersenneTwister(123) 

#### truncated-normal factors with noise
μ = 2
σ = √2
dist_W = truncated(Normal(μ, σ), 0, Inf)
dist_H = truncated(Normal(μ, σ), 0, Inf)

dist_E = Normal(μ, 0.2*σ)
dist_E_type = "Normal"

# α = μ
# θ = 1
# dist_E = Gamma(α, θ)
# dist_E_type = "Gamma"

W = rand(rng, dist_W, F, K)
H = rand(rng, dist_H, K, N)
E = rand(rng, dist_E, F, N)

V = W * H + E

fo = "../../plot/simulation/nmf_noise"
mkpath(fo)

#### global params
global_params = NMFParams(
    init          = :random,
    dist          = Uniform(0,1),
    rank          = K,
    max_iter      = 20_000,
    stopping_crit = :none,
    verbose       = true,
)

#### # 2. FEVOTTE-IDIER NMF ALGORITHMS WITH KNOWN RANK 
#### # ################################################################################################################

list_res = Array{NMFResults{Float64}, 1}()
list_lab = Array{String, 1}()
list_ls  = Array{Symbol, 1}()
list_col = Array{String, 1}()

local_params = FIParams(
    scale_W = true,
    α_H     = 0,
    α_W     = 0,
)

####
#### 2.1 β divergences, FI algorithms
####

for (β, col) in zip([2, 1, 0], ["black", "red", "green"])
    local_params.β = β

    run_mm = true
    run_me = true
    run_h  = true

    if β == 2
        run_h = false
    elseif β == 1
        run_h  = false
        run_me = false
    end

    if run_mm
        local_params.alg = :mm

        push!(list_res, nmf_FI(V, global_params, local_params))
        push!(list_lab, raw"$\mathrm{MM} \quad \beta =" * "$(β)" * raw"$")
        push!(list_ls, :dash)
        push!(list_col, col)
    end

    if run_me
        local_params.alg = :me

        push!(list_res, nmf_FI(V, global_params, local_params))
        push!(list_lab, raw"$\mathrm{ME} \quad \beta =" * "$(β)" * raw"$")
        push!(list_ls, :solid)
        push!(list_col, col)
    end

    if run_h
        local_params.alg = :h

        push!(list_res, nmf_FI(V, global_params, local_params))
        push!(list_lab, raw"$\mathrm{H} \quad \beta =" * "$(β)" * raw"$")
        push!(list_ls, :dot)
        push!(list_col, col)
    end

end

####
#### 2.2 PLOT
####

fn = joinpath(fo, "nmf_FI_noise_$(dist_E_type).pdf")
plot_cost(list_res, list_lab, list_ls, list_col, fn, relative=true)

#### # 3. MU NMF ALGORITHMS WITH KNOWN RANK 
#### # ################################################################################################################

list_res = Array{NMFResults{Float64}, 1}()
list_lab = Array{String, 1}()
list_ls  = Array{Symbol, 1}()
list_col = Array{String, 1}()

####
#### 2.1 β divergences
####

for (β, col) in zip([2, 1, 0], ["black", "red", "green"])
    local_params = MUParams(
        β       = β,
        scale_W = true,
        div     = :β,
        alg     = :mu
    )

    push!(list_res, nmf_MU(V, global_params, local_params))
    push!(list_lab, raw"$\mathrm{MU} \quad \beta =" * "$(β)" * raw"$")
    push!(list_ls, :solid)
    push!(list_col, col)
end

####
#### 2.2 α divergences
####

for (α, col) in zip([2, 1, 0], ["black", "red", "green"])
    local_params = MUParams(
        α       = α,
        scale_W = true,
        div     = :α,
        alg     = :mu
    )

    push!(list_res, nmf_MU(V, global_params, local_params))
    push!(list_lab, raw"$\mathrm{MU} \quad \alpha =" * "$(α)" * raw"$")
    push!(list_ls, :dash)
    push!(list_col, col)
end

####
#### 2.3 PLOT
####

fn = joinpath(fo, "nmf_MU_noise_$(dist_E_type).pdf")
plot_cost(list_res, list_lab, list_ls, list_col, fn, relative=true)

#### # Cichocki NMF algorithms
#### #################################################################################################################

#params = CParams(
#    α             = 2,
#    max_iter      = convert(Int, 1e4),
#    stopping_crit = :none,
#    stopping_tol  = 1e-6,
#    α_H           = 0,
#    α_W           = 0,
#    l₁ratio_H     = 1,
#    l₁ratio_W     = 1,
#    μ_H           = 0,
#    μ_W           = 0,
#    α_sparse_H    = 0,
#    α_sparse_W    = 0,
#    ω             = 1,
#    verbose       = true,
#)
#
#
#res_α = nmf_α(
#    V,
#    W_init,
#    H_init,
#    params
#)
