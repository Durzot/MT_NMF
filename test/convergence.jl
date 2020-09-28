using Test
using VariantsNMF

@testset "nmf_FI" begin
    # simulate one factorisation
    V = get_one_simulated_V()

    # parameters of solvers
    βs = [-1, -0.5, 0, 0.5, 1, 1.5, 2, 3]
    stopping_crits = [:none, :conn, :rel_cost, :abs_cost] 
    stopping_tols  = [1, 1, 1e-6, 1]

    for (stopping_crit, stopping_tol) in zip(stopping_crits, stopping_tols)
        global_params = NMFParams(
            init          = :random,
            dist          = Uniform(0, 1),
            max_iter      = 1_000,
            stopping_crit = stopping_crit,
            stopping_tol  = stopping_tol,
            verbose       = 0,
            seed          = 0
        )

        for β in βs
            local_params = FIParams(β = β)

            local_params.alg = :mm
            res = nmf_FI(V, global_params, local_params)
            @test res.converged

            local_params.alg = :h
            res = nmf_FI(V, global_params, local_params)
            @test res.converged

            if β in [-1, 0, 0.5, 1.5, 2, 3]
                local_params.alg = :me
                res = nmf_FI(V, global_params, local_params)
                @test res.converged
            end
        end
    end
end
