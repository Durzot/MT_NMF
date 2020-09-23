"""
Created on Fri 15 May 2020

@author: Yoann Pradat

Performs NMF rank selection by the method of consensus matrices and related metrics.
"""

#using Pkg
#ENV["Python"] = "/usr/local/anaconda3/bin/python"
#Pkg.build("PyCall")

println("working dir: " *  pwd())

using Distances
using Distributed
using Distributions
using PyCall
using Random

@everywhere include("./src/VNMF.jl")
@everywhere using .VNMF

include(pwd() * "/demo/src/simulation/simulate.jl")

#### folder for saving plots and other files
folder   = "./demo/plot/simulation/rank_consensus_a"
mkpath(folder)

#### # 1. SIMULATE DATA
#### # ################################################################################################################

#### fix rng
rng = MersenneTwister(123)

#### simulate clusters of columns (individuals) and clusters of rows (variables)
W, H, lims_K, lims_N, active_clusters = simulate_WH_clustered(
    F              = 40,
    N              = 200,
    K              = 6,
    n_clusters_N   = 6,
    dist_W         = Uniform(0,1),
    dist_H         = Poisson(500),
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
E = rand(rng, Poisson(1), size(W,1), size(H, 2))

#### V matrix 
V = round.(W * H) + E

py"""
import matplotlib.cm as cm
import numpy as np
import sys
if "." not in sys.path:
    sys.path.append(".")

from demo.src.util.plot_VWH import *

cmap = cm.get_cmap("Paired")

def make_plot_V(V, col_indices, filepath):
    col_indices_colors = {i: cmap(i) for i in np.unique(col_indices)}

    fig = plot_V(V, col_groups=col_indices, col_groups_colors=col_indices_colors)
    fig.savefig(filepath, bbox_inches="tight")
"""

filepath = joinpath(folder, "plot_V_true.pdf")
py"make_plot_V"(V, col_indices, filepath)

#### remove cluster order by shuffling
rand_order_V = randperm(size(V,2))
V_obs = V[:, rand_order_V]
col_indices_obs = col_indices[rand_order_V]

filepath = joinpath(folder, "plot_V_obs.pdf")
py"make_plot_V"(V_obs, col_indices_obs, filepath)

#### # 2. SELECT ALGORITHM
#### # ################################################################################################################

#### params for rank selection
rc_params = RCParams(
    K_min  = 2,
    K_max  = 10,
    n_iter = 50,
    seed   = 123
)

#### nmf algorithm
nmf_global_params = NMFParams(
    init          = :random,
    dist          = Uniform(0, 1),
    max_iter      = 20_000,
    stopping_crit = :conn,
    stopping_iter = 100,
    verbose       = false,
    seed          = 0
)

nmf_local_params = FIParams(
    β            = 2,
    scale_W_iter = false,
    scale_W_last = true,
    alg          = :mm
)

nmf = NMF(
    solver        = nmf_FI,
    global_params = nmf_global_params,
    local_params  = nmf_local_params
)

#### get cophenetic correlation per rank
@time rc_results = rank_by_consensus(V_obs, rc_params, nmf)

#### # 3. PLOT CONSENSUS METRICS
#### # ################################################################################################################

include(pwd() * "/demo/src/util/plot_metrics.jl")

filepath = joinpath(folder, "rank_consensus_metrics.pdf")
plot_metrics_consensus(rc_results.df_metrics, filepath)

#### # 4. PLOT CONSENSUS MATRICES
#### # ################################################################################################################

py"""
from demo.src.util.plot_consensus import *

def make_plot_consensus(cons, filepath):
    config   = ConsensusConfig()
    fig = plot_consensus(cons, config)
    fig.savefig(filepath, bbox_inches='tight')
"""

for (cons, rank) in zip(rc_results.list_cons, rc_results.df_metrics.rank)
    filepath = joinpath(folder, "rank_consensus_$(rank).png")
    py"make_plot_consensus"(cons, filepath)
end

#### # 5. RUN NMF AT BEST RANK
#### # ################################################################################################################

best_rank = rc_results.df_metrics[argmax(rc_results.df_metrics[:,:dispersion]),:rank]

nmf_global_params.rank    = best_rank
nmf_global_params.verbose = true
nmf.global_params = nmf_global_params
nmf.local_params  = nmf_local_params

nmf_res = nmf.solver(V_obs, nmf.global_params, nmf.local_params)

#### # 6. PLOT NMF AT BEST RANK
#### # ################################################################################################################

#### plot W * H with order from consensus at best rank
cons_order = rc_results.list_idxs[rc_results.df_metrics.rank .== best_rank][1]
V_ord = V_obs[:, cons_order]
col_indices_ord = col_indices_obs[cons_order]

filepath = joinpath(folder, "plot_V_ord.pdf")
py"make_plot_V"(V_ord, col_indices_ord, filepath)

#### plot W and and cosine similarity matrix with columns of true W

py"""
from demo.src.util.plot_VWH import *

def make_plot_W_alone(W_fit, W_ref, filepath):
    col_ref_labels = ["ref %d" % i for i in range(W_ref.shape[1])]

    fig = plot_W_alone(
        W_fit          = W_fit,
        W_ref          = W_ref,
        hmap_cmap      = "Blues",
        col_ref_labels = col_ref_labels,
        figsize        = (16,10)

    )
    fig.savefig(filepath, bbox_inches='tight')
"""

filepath = joinpath(folder, "plot_W_alone.pdf")
py"make_plot_W_alone"(nmf_res.W, W, filepath)
