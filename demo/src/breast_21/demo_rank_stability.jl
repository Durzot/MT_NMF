"""
Created on Fri 15 May 2020

@author: Yoann Pradat

Performs NMF rank selection by the method of factor stability across sligth perturbations of the V matrix. 
"""

using Distances
using Distributed
using Distributions
using DelimitedFiles
using PyCall

@everywhere include("./src/VNMF.jl")
@everywhere using .VNMF

#### folder for saving plots and other files
folder   = "./demo/plot/breast_21"
mkpath(folder)

#### # 1. LOAD DATA
#### # ################################################################################################################

row_types    = vec(readdlm(pwd() * "/../../../data/breast_21/types.txt", ',', String, '\n'))
row_subtypes = vec(readdlm(pwd() * "/../../../data/breast_21/subtypes.txt", ',', String, '\n'))
V = readdlm(pwd() * "/../../../data/breast_21/originalGenomes.txt", ',', Float64, '\n')

# new_order = sortperm(row_types)
# row_types = row_types[new_order]
# row_subtypes = row_subtypes[new_order]
# V = V[new_order, :]
# 
# py"""
# import sys
# if "." not in sys.path:
#     sys.path.append(".")
# 
# from demo.src.util.plot_VWH import *
# 
# def make_plot_V(V, row_types, row_subtypes, filepath):
#     row_groups_colors = {
#         "C>A": "cornflowerblue",
#         "C>G": "lightseagreen",
#         "C>T": "coral",
#         "T>A": "sandybrown",
#         "T>C": "forestgreen",
#         "T>G": "deeppink"
#     }
# 
#     fig = plot_V(
#         V                 = V,
#         row_groups        = row_types,
#         row_groups_colors = row_groups_colors,
#         row_labels        = row_subtypes,
#         row_labelsize     = 4
#     )
#     fig.savefig(filepath, bbox_inches="tight")
# """
# 
# filepath = joinpath(folder, "plot_V_obs.pdf")
# py"make_plot_V"(V, row_types, row_subtypes, filepath)

#### # 2. SELECT ALGORITHM
#### # ################################################################################################################

#### params for rank selection
rs_params = RSParams(
    K_min  = 1,
    K_max  = 10,
    n_iter = 50,
    pert   = :Multinomial,
    seed   = 123
)

#### nmf algorithm
nmf_global_params = NMFParams(
    init          = :random,
    dist          = Uniform(0, 1),
    max_iter      = 10_000,
    stopping_crit = :conn,
    stopping_iter = 100,
    verbose       = false,
)

nmf_local_params = MUParams(
    β       = 1,
    div     = :β,
    scale_W = false,
    alg     = :mu
)

nmf = NMF(
    solver        = nmf_MU,
    global_params = nmf_global_params,
    local_params  = nmf_local_params
)

#### run rank selection by stability procedure
@time rs_results = rank_by_stability(V, rs_params, nmf)

#### # 3. PLOT RESULTS
#### # ################################################################################################################

#### plot metrics
include(pwd() * "/demo/src/util/plot_metrics.jl")
filepath = joinpath(folder, "rank_stability_metrics.pdf")
plot_metrics_stability(rs_results.df_metrics, filepath)

#### plot factors

py"""
import matplotlib.cm as cm
import sys
if "." not in sys.path:
    sys.path.append(".")

from demo.src.util.plot_VWH import *

def make_plot_W(W_avg, W_std, stab_avg, stab_std, row_types, row_subtypes, filepath):
    row_groups_colors = {
        "C>A": "cornflowerblue",
        "C>G": "lightseagreen",
        "C>T": "coral",
        "T>A": "sandybrown",
        "T>C": "forestgreen",
        "T>G": "deeppink"
    }
    hmap_boundaries = [0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    hmap_cmap = "Blues"

    fig = plot_W_stab(
        W_avg             = W_avg,
        W_std             = W_std,
        stab_avg          = stab_avg,
        stab_std          = stab_std,
        row_groups        = row_types,
        row_groups_colors = row_groups_colors,
        row_labels        = row_subtypes,
        hmap_boundaries   = hmap_boundaries,
        hmap_cmap         = hmap_cmap
    )
    fig.savefig(filepath, bbox_inches="tight")
"""

for K = rs_params.K_min:rs_params.K_max
    filepath = joinpath(folder, "plot_W_rank_$(K).pdf")
    clu_results = rs_results.list_clu_results[K]
    py"make_plot_W"(clu_results.W_avg, clu_results.W_std, clu_results.stab_avg, clu_results.stab_std, row_types, row_subtypes, filepath)
end


#### DEBUG

K = 1

W_init = readdlm(pwd() * "/../../../data/breast_21/W_init_rank$K.txt", ',', Float64, '\n')
H_init = readdlm(pwd() * "/../../../data/breast_21/H_init_rank$K.txt", ',', Float64, '\n')

W_fit = readdlm(pwd() * "/../../../data/breast_21/W_fit_rank$K.txt", ',', Float64, '\n')
H_fit = readdlm(pwd() * "/../../../data/breast_21/H_fit_rank$K.txt", ',', Float64, '\n')

V = readdlm(pwd() * "/../../../data/breast_21/originalGenomes.txt", ',', Float64, '\n')


#### nmf algorithm
nmf_global_params = NMFParams(
    init          = :random,
    dist          = Uniform(0, 1),
    max_iter      = 10_000,
    stopping_crit = :conn,
    stopping_iter = 100,
    verbose       = true,
)

# nmf_local_params = MUParams(
#     β       = 1,
#     div     = :β,
#     scale_W = false,
#     alg     = :mu
# )

nmf_local_params = FIParams(
    β       = 1,
    scale_W = false,
    alg     = :mm
)

W_fit_julia = zeros(Float64, size(W_fit))
H_fit_julia = zeros(Float64, size(H_fit))

for j =  1:10
    V_iter = readdlm(pwd() * "/../../../data/breast_21/V$(j)_rank$(K).txt", ',', Float64, '\n')

    w_init = reshape(W_init[:,j], size(W_init,1), 1)
    h_init = reshape(H_init[j,:], 1, size(H_init,2))

    res = VNMF._nmf_FI(V_iter, w_init, h_init, nmf_global_params, nmf_local_params)

    VNMF.scale_col_W!(res.W, res.H)

    W_fit_julia[:, j] .= reshape(res.W, length(res.W))
    H_fit_julia[j, :] .= reshape(res.H, length(res.H))
end

@test maximum(abs.(W_fit_julia - W_fit)) ≈ 0 atol = 1e-10
@test maximum(abs.(H_fit_julia - H_fit)) ≈ 0 atol = 1e-10

##### ANALYSE W_ALL AND H_ALL
W_all = W_fit_julia
H_all = H_fit_julia

##### REPRODUCE FOR RANK = 5

nmf_global_params = NMFParams(
    init          = :random,
    dist          = Uniform(0, 1),
    rank          = 5,
    max_iter      = 20_000,
    stopping_crit = :cost,
    stopping_tol  = 1e-6,
    verbose       = false,
)

nmf_local_params = MUParams(
    β       = 2,
    div     = :β,
    scale_W = true,
    alg     = :mu
)

nmf = NMF(
    solver        = nmf_MU,
    global_params = nmf_global_params,
    local_params  = nmf_local_params
)

#### build grid
list_V_pert = VNMF._get_list_perturbed_matrix(V, 50, :Multinomial, rng=rng)
list_nmf    = Vector{NMF}()
for (rank, seed) in zip(5:5, 5:5)
    #### using a different seed for each run of NMF
    nmf_rank = copy(nmf)
    nmf_rank.global_params.rank = rank
    nmf_rank.global_params.seed = seed

    push!(list_nmf, nmf_rank)
end

grid = vec(collect(Iterators.product(list_V_pert, list_nmf)))

#### run in parallel
grid_nmf_results = @showprogress pmap(grid) do g
    VNMF._rank_by_stability_one(g)
end


W_all = hcat([nmf_results.W for nmf_results in grid_nmf_results]...)
H_all = vcat([nmf_results.H for nmf_results in grid_nmf_results]...)

#### recover parameters of the algorithm
n_fac = nmf_global_params.rank
n_nmf = size(grid_nmf_results, 1)
n_tot = n_fac * n_nmf
