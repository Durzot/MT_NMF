"""
Created on Fri 15 May 2020

@author: Yoann Pradat

Performs NMF rank selection by the method of factor stability across sligth perturbations of the V matrix. 
"""

using Distances
using Distributed
using Distributions
using Random
using PyCall

@everywhere include("./src/VNMF.jl")
@everywhere using .VNMF

include(pwd() * "/demo/src/simulation/simulate.jl")

#### folder for saving plots and other files
folder   = "./demo/plot/simulation/rank_stability_a"
mkpath(folder)

#### # 1. SIMULATE DATA
#### # ################################################################################################################

#### fix rng
rng = MersenneTwister(123)

#### simulate clusters of columns (individuals) and clusters of rows (variables)
W, H, lims_K, lims_N, active_clusters = simulate_WH_clustered(
    F              = 96,
    N              = 200,
    K              = 5,
    n_clusters_N   = 5,
    dist_W         = truncated(Normal(0, 3), 0, Inf),
    dist_H         = Poisson(1000),
    H_cluster_mode = :a,
    rng            = rng
)

#### get col cluster indices
col_indices = zeros(Int, size(H,2))
for i in 1:(length(lims_N)-1)
    range_indices = (lims_N[i]+1):lims_N[i+1]
    col_indices[range_indices] .= i
end

#### simulate noise
E = rand(rng, Poisson(2), size(W,1), size(H, 2))

#### V input matrix (observations)
V = round.(W * H) + E

py"""
import sys
if "." not in sys.path:
    sys.path.append(".")

from demo.src.util.plot_VWH import *

def make_plot_V(V, col_indices, filepath):

    row_groups_colors = {
        "C>A": "cornflowerblue",
        "C>G": "lightseagreen",
        "C>T": "coral",
        "T>A": "sandybrown",
        "T>C": "forestgreen",
        "T>G": "deeppink"
    }
    fig = plot_V(V, col_indices)
    fig.savefig(filepath, bbox_inches="tight")
"""

filepath = joinpath(folder, "plot_V_true.pdf")
py"make_plot_V"(V, col_indices, filepath)

#### remove cluster order by shuffling
rand_order_V = randperm(rng, size(V,2))
V_obs = V[:, rand_order_V]
col_indices_obs = col_indices[rand_order_V]

filepath = joinpath(folder, "plot_V_obs.pdf")
py"make_plot_V"(V_obs, col_indices_obs, filepath)

#### save V to test on SigProfiler
folder_results   = "./demo/results/simulation/rank_stability_a"
mkpath(folder_results)

using DelimitedFiles
writedlm(joinpath(folder_results, "V_a_simulated.txt"), V, '\t')

#### # 2. SELECT ALGORITHM
#### # ################################################################################################################

#### params for rank selection
rs_params = RSParams(
    K_min  = 2,
    K_max  = 8,
    n_iter = 10,
    pert   = :Multinomial,
    seed   = 123
)

#### nmf algorithm
nmf_global_params = NMFParams(
    init          = :random,
    dist          = Uniform(0, 1),
    max_iter      = 20_000,
    stopping_crit = :cost,
    stopping_tol  = 1e-6,
    verbose       = false,
)

nmf_local_params = MUParams(
    β       = 2,
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
@time rs_results = rank_by_stability(V_obs, rs_params, nmf)


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
