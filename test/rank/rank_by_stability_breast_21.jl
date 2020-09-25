"""
Created on Wed 03 June 2020

@author: Yoann Pradat

Test rank by stability as implement in SigProfiler matlab package. Compare VNMF results to that of the package on the 
same matrices.
"""

#### function for plotting factors
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
        hmap_cmap         = hmap_cmap,
        figsize           = (16,10),
    )
    fig.savefig(filepath, bbox_inches="tight")
"""

#### nmf algorithm with identical parameters
nmf_global_params = NMFParams(
    max_iter      = 10_000,
    dist          = Uniform(0,1),
    stopping_crit = :conn,
    stopping_iter = 100,
    verbose       = false,
    seed          = 123
)

nmf_local_params = FIParams(
    β            = 1,
    scale_W_iter = false,
    scale_W_last = true,
    alg          = :mm
)

nmf_alg = VNMF._nmf_FI

#### read row types and row subtypes
row_types    = vec(readdlm(pwd() * "/tests/data/breast_21/types.txt", ',', String, '\n'))
row_subtypes = vec(readdlm(pwd() * "/tests/data/breast_21/subtypes.txt", ',', String, '\n'))
row_types_order = sortperm(row_types)
V_breast_21 = readdlm(pwd() * "/tests/data/breast_21/originalGenomes.txt", ',', Float64, '\n')

K_min  = 1
K_max  = 5
n_iter = 20

#### # 1. USE PERTURBED AND INIT MATRICES PRODUCED BY SIGPROFILER
#### # ################################################

folder = pwd() * "/tests/plot/breast_21/"
mkpath(folder)

rng = MersenneTwister(123)

@testset "breast_21" begin
    #### loop over candidate ranks
    for K = K_min:K_max
        nmf_global_params.rank =  K

        W_init = readdlm(pwd() * "/tests/data/breast_21/W_init_rank_$(K).txt", ',', Float64, '\n')
        H_init = readdlm(pwd() * "/tests/data/breast_21/H_init_rank_$(K).txt", ',', Float64, '\n')

        W_fit = readdlm(pwd() * "/tests/data/breast_21/W_fit_rank_$(K).txt", ',', Float64, '\n')
        H_fit = readdlm(pwd() * "/tests/data/breast_21/H_fit_rank_$(K).txt", ',', Float64, '\n')

        W_fit_julia = zeros(Float64, size(W_fit))
        H_fit_julia = zeros(Float64, size(H_fit))

        list_nmf_results = Array{NMFResults{Float64},1}()

        #### loop over bootstrap iterations
        for j = 1:n_iter
            v_pert = readdlm(pwd() * "/tests/data/breast_21/V_iter_$(j)_rank_$(K).txt", ',', Float64, '\n')

            w_init = reshape(W_init[:,((j-1)*K+1):j*K], size(W_init,1), K)
            h_init = reshape(H_init[((j-1)*K+1):j*K,:], K, size(H_init,2))

            nmf_results = nmf_alg(v_pert, w_init, h_init, nmf_global_params, nmf_local_params)
            push!(list_nmf_results, nmf_results)

            W_fit_julia[:, ((j-1)*K+1):j*K] .= nmf_results.W
            H_fit_julia[((j-1)*K+1):j*K, :] .= nmf_results.H 
        end

        @test maximum(abs.(W_fit_julia - W_fit)) ≈ 0 atol = 1e-8
        @test maximum(abs.(H_fit_julia - H_fit)) ≈ 0 atol = 1e-8

        #### cluster factors across iterations
        clu_results = VNMF.evaluate_stability(list_nmf_results, dist_func = CosineDist(), rng=rng)

        filepath = joinpath(folder, "plot_W_rank_$(K)_sigpro.pdf")
        py"make_plot_W"(
            clu_results.W_avg[row_types_order,:], 
            clu_results.W_std[row_types_order,:], 
            clu_results.stab_avg, 
            clu_results.stab_std, 
            row_types[row_types_order], 
            row_subtypes[row_types_order], 
            filepath
           )
    end
end

#### # 2. USE PERTURBED MATRICES PRODUCED BY VNMF
#### # ################################################

#### params for rank selection
rs_params = RSParams(
    K_min     = K_min,
    K_max     = K_max,
    n_iter    = 200,
    pert_meth = :multinomial,
    seed      = 123
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

nmf_local_params = FIParams(
    β            = 1,
    scale_W_iter = false,
    scale_W_last = true,
    alg          = :mm
)

nmf = NMF(
    solver        = nmf_FI,
    global_params = nmf_global_params,
    local_params  = nmf_local_params
)

#### run rank selection by stability procedure
@time rs_results = rank_by_stability(V_breast_21, rs_params, nmf)

folder = pwd() * "/tests/plot/breast_21/"

for clu_results in rs_results.list_clu_results
    K = length(clu_results.stab_avg)

    filepath = joinpath(folder, "plot_W_rank_$(K)_VNMF.pdf")
    py"make_plot_W"(
        clu_results.W_avg[row_types_order,:], 
        clu_results.W_std[row_types_order,:], 
        clu_results.stab_avg, 
        clu_results.stab_std, 
        row_types[row_types_order], 
        row_subtypes[row_types_order], 
        filepath
    )
end
