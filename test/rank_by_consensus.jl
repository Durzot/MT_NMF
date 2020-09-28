using VariantsNMF
using Test

#### simulate
V = get_one_simulated_V()

@testset "nmf_FI" begin
    #### params for rank selection
    rc_params = RCParams(
        K_min = 2,
        K_max = 6,
        n_iter = 10
    )

    #### nmf β-divergence, multiplicative updates
    nmf_global_params = NMFParams(
        init          = :random,
        dist          = truncated(Normal(1, 1), 0, Inf),
        max_iter      = 1_000,
        stopping_crit = :conn,
        stopping_iter = 10,
        verbose       = false,
    )

    nmf_local_params = FIParams(
        β         = 1,
        l₁ratio_H = 0,
        α_H       = 0,
        alg       = :mm
    )

    #### NMF struct
    nmf = NMF(
        solver        = nmf_FI,
        global_params = nmf_global_params,
        local_params  = nmf_local_params
    )

    #### get consensus metrics for each rank
    @time rc_results = rank_by_consensus(V, rc_params, nmf)

    @test "rank" in names(rc_results.df_metrics)
    @test "cophenetic" in names(rc_results.df_metrics)
    @test "dispersion" in names(rc_results.df_metrics)
    @test "silhouette" in names(rc_results.df_metrics)
    @test "sparse_W" in names(rc_results.df_metrics)
    @test "sparse_H" in names(rc_results.df_metrics)
end
