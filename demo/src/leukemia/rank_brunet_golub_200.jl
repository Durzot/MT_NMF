"""
Created on Fri 15 May 2020

@author: Yoann Pradat

Performs NMF rank selection by the method of consensus matrices and cophenetic correlation on the
the first 200 genes of the 5000 x 38 leukemia dataset used in the article of Brunet et al.
"""

using CSV
using Distributed
using DelimitedFiles
using Distributions

@everywhere include("./src/VNMF.jl")
@everywhere using .VNMF

df_pos = CSV.read("../../../data/all_aml_38/golub_nmf_R_200.txt", delim='\t')
mt_pos = convert(Matrix{Float64}, df_pos[:, 2:end])

#### # 2. PARAMETERS ALGORITHM
#### # ##############################################################################################################

#### params for rank selection
rc_params = RCParams(
    K_min = 2,
    K_max = 4,
    n_iter = 10
)

#### nmf β-divergence, multiplicative updates
nmf_global_params = NMFParams(
    init          = :random,
    dist          = truncated(Normal(1, 1), 0, Inf),
    max_iter      = convert(Int, 1e4),
    stopping_crit = :conn,
    stopping_iter = 100,
    verbose       = false,
)

nmf_local_params = FIParams(
    β         = 1,
    l₁ratio_H = 0,
    α_H       = 1,
    alg       = :mm
)


#### NMF struct
nmf = NMF(
    solver        = nmf_FI,
    global_params = nmf_global_params,
    local_params  = nmf_local_params
)

#### get consensus metrics for each rank
@time rc_results = rank_by_consensus(mt_pos, rc_params, nmf)

#### # 3. PLOT WITH PYCALL
#### # ################################################################################################################

#using Pkg
#ENV["Python"] = "/usr/local/anaconda3/bin/python"
#Pkg.build("PyCall")

using PyCall

py"""
import os
import sys
sys.path.append(".")

from demo.src.plot_consensus import *

def make_plot(X, rank):
    folder   = "demo/plot/simulation/rank_cophenetic"
    os.makedirs(folder, exist_ok=True)
    filename = "rank_brunet_consensus_%s.png"  % rank
    filepath = os.path.join(folder, filename)

    config   = ConsensusConfig()

    fig = plot_consensus(X, config)
    fig.savefig(filepath, bbox_inches='tight')
    print("file saved at %s" % filepath)
"""

for (cons, rank) in zip(rc_results.list_cons, rc_results.list_rank)
    py"make_plot"(cons, rank)
end
