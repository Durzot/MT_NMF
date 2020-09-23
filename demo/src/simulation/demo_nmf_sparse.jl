"""
Created on Mon May 22 2020

@author: Yoann Pradat

Evaluate NMF algorithms with sparsity.
"""

include("./src/VNMF.jl")
using .VNMF
using DataFrames
using Distributions
using Random

include("./demo/src/util/plot_cost.jl")
gr() # set the backend to GR

#### # 1. SIMULATE DATA
#### # ################################################################################################################

#### simulate matrix for testing NMF
F = 10
N = 25
K = 5
rng = MersenneTwister(123) 

#### truncated-normal factors with normal noise
dist_W = truncated(Normal(5, 2), 0, Inf)
dist_H = truncated(Normal(5, 2), 0, Inf)
dist_E = Normal(1, 0.1)

W = rand(rng, dist_W, F, K)
H = rand(rng, dist_H, K, N)
E = rand(rng, dist_E, F, N)

V = W * H + E

#### # 2. FEVOTTE-IDIER NMF ALGORITHMS WITH KNOWN RANK 
#### # ################################################################################################################

fo = "demo/plot/simulation/nmf_sparse"
mkpath(fo)

#### global params
global_params = NMFParams(
    init          = :random,
    dist          = truncated.(Normal(0, 1), 0, Inf),
    rank          = K,
    max_iter      = 10_000,
    stopping_crit = :cost,
    stopping_tol  = 1e-6,
    verbose       = false,
)


#### # 2.1 L1 Regularization on H, MM algo
list_local_params = Array{FIParams, 1}()
list_sparse_W = Array{Real, 1}()
list_sparse_H = Array{Real, 1}()
list_cur_cost = Array{Real, 1}()
list_cur_divg = Array{Real, 1}()

list_β = [-1, 0, 1, 2]
list_e = [-1, 0, 1, 2, 3, 4]

#### MM alg
for β in list_β
    for e in list_e
        println(repeat("=", 80))
        #### MM alg
        local_params = FIParams(
            β         = β,
            scale_W   = false,
            α_H       = 10.0^e,
            α_W       = 0,
            l₁ratio_H = 1,
            l₁ratio_W = 1,
            alg       = :mm,
        )
        push!(list_local_params, local_params)
        @info "params" local_params.β local_params.α_W local_params.α_H

        #### run
        res_FI_MM = nmf_FI(V, global_params, local_params)

        cur_divg = res_FI_MM.metrics.cur_divg
        cur_cost = res_FI_MM.metrics.cur_cost
        sparse_W = VNMF.matr_avg_col_sparseness(res_FI_MM.W)
        sparse_H = VNMF.matr_avg_col_sparseness(res_FI_MM.H)

        push!(list_cur_divg, cur_divg)
        push!(list_cur_cost, cur_cost)
        push!(list_sparse_W, sparse_W)
        push!(list_sparse_H, sparse_H)

        @info "at convergence" cur_divg, cur_cost
        @info "sparsity level" sparse_W sparse_H
    end
end

#### get results in a dataframe
df_metrics_MM = DataFrame(
   β = (x -> x.β).(list_local_params),
   α_W = (x -> x.α_W).(list_local_params),
   α_H = (x -> x.α_H).(list_local_params),
   cur_divg = list_cur_divg,
   cur_cost = list_cur_cost,
   sparse_W = list_sparse_W,
   sparse_H = list_sparse_H,
)

#### # 2.2 L1 Regularization on H, H algo
list_local_params = Array{FIParams, 1}()
list_sparse_W = Array{Real, 1}()
list_sparse_H = Array{Real, 1}()
list_cur_cost = Array{Real, 1}()
list_cur_divg = Array{Real, 1}()

list_β = [-1, 0, 1, 2]
list_e = [-1, 0, 1, 2, 3, 4]

#### MM alg
for β in list_β
    for e in list_e
        println(repeat("=", 80))
        #### MM alg
        local_params = FIParams(
            β         = β,
            scale_W   = false,
            α_H       = 10.0^e,
            α_W       = 0,
            l₁ratio_H = 1,
            l₁ratio_W = 1,
            alg       = :h,
        )
        push!(list_local_params, local_params)
        @info "params" local_params.β local_params.α_W local_params.α_H

        #### run
        res_FI_MM = nmf_FI(V, global_params, local_params)

        cur_divg = res_FI_MM.metrics.cur_divg
        cur_cost = res_FI_MM.metrics.cur_cost
        sparse_W = VNMF.matr_avg_col_sparseness(res_FI_MM.W)
        sparse_H = VNMF.matr_avg_col_sparseness(res_FI_MM.H)

        push!(list_cur_divg, cur_divg)
        push!(list_cur_cost, cur_cost)
        push!(list_sparse_W, sparse_W)
        push!(list_sparse_H, sparse_H)

        @info "at convergence" cur_divg, cur_cost
        @info "sparsity level" sparse_W sparse_H
    end
end

#### get results in a dataframe
df_metrics_H = DataFrame(
   β = (x -> x.β).(list_local_params),
   α_W = (x -> x.α_W).(list_local_params),
   α_H = (x -> x.α_H).(list_local_params),
   cur_divg = list_cur_divg,
   cur_cost = list_cur_cost,
   sparse_W = list_sparse_W,
   sparse_H = list_sparse_H,
)
