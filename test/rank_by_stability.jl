using Distributions
using VariantsNMF
using Test

#### simulate
V = get_one_simulated_V()
V = round.(V .* 100)

@test "nmf_MU" begin

    #### params for rank selection
    rs_params = RSParams(
        K_min     = 1,
        K_max     = 10,
        n_iter    = 50,
        pert_meth = :multinomial,
        seed      = 123
    )

    #### nmf algorithm
    nmf_global_params = NMFParams(
        init          = :random,
        dist          = Uniform(0, 1),
        max_iter      = 1_000,
        stopping_crit = :conn,
        stopping_iter = 10,
        verbose       = false,
    )

    nmf_local_params = MUParams(
        β            = 1,
        div          = :β,
        scale_W_iter = false,
        scale_W_last = false,
        alg          = :mu
    )

    nmf = NMF(
        solver        = nmf_MU,
        global_params = nmf_global_params,
        local_params  = nmf_local_params
    )

    #### run rank selection by stability procedure
    @time rs_results = rank_by_stability(V, rs_params, nmf)

    @test "rank" in names(rs_results.df_metrics)
    @test "stab_avg" in names(rs_results.df_metrics)
    @test "stab_std" in names(rs_results.df_metrics)
    @test "fid_avg" in names(rs_results.df_metrics)
    @test "fid_std" in names(rs_results.df_metrics)
end
