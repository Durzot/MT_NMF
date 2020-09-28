using Distributions
using DataFrames
using Test
using VariantsNMF

function test_convergence(V, β, local_params)
    params = copy(local_params)
    params.β = β
    res = nmf_FI(V, global_params, params)
    issorted(res.metrics.absolute_cost[1:end-global_params.stopping_iter], rev=true)
end

function test_reg_H(V, β, αs, local_params)
    params = copy(local_params)
    params.β = β
    s_H = Array{Float64,1}()
    for α_H in αs
        params.α_H = α_H
        res = nmf_FI(V, global_params, params)
        push!(s_H, VariantsNMF.matr_avg_row_sparseness(res.H))
    end
    issorted(s_H)
end

function test_reg_W(V, β, αs, local_params)
    params = copy(local_params)
    params.β = β
    s_W = Array{Float64,1}()
    for α_W in αs
        params.α_W = α_W
        res = nmf_FI(V, global_params, params)
        push!(s_W, VariantsNMF.matr_avg_col_sparseness(res.W))
    end
    issorted(s_W)
end

function run_FI_tests(global_params, local_params, βs, αs)
    # record test results
    df_test_results = DataFrame(
        β = Float64[], 
        noise = String[],
        test_cvg_passed = Bool[], 
        test_reg_H_passed = Bool[], 
        test_reg_W_passed = Bool[]
    )

    # get simulations of factorisations for test
    noise_Vs, noise_names = get_many_simulated_V()

    for β in βs
        for (V, name) in zip(noise_Vs, noise_names)
            # cvg test: check monotonicity of the cost function
            # reg_H test: check monotonicity of matrix sparseness
            # reg_W test: check monotonicity of matrix sparseness

            if β ≤ 0 && minimum(V) == 0
                cvg_passed = test_convergence(V .+ 1e-4, β, local_params)
                reg_H_passed = test_reg_H(V .+ 1e-4, β, αs, local_params)
                reg_W_passed = test_reg_W(V .+ 1e-4, β, αs, local_params)
            else
                cvg_passed = test_convergence(V, β, local_params)
                reg_H_passed = test_reg_H(V, β, αs, local_params)
                reg_W_passed = test_reg_W(V, β, αs, local_params)
            end
            @test cvg_passed
            push!(df_test_results, [β name cvg_passed reg_H_passed reg_W_passed])
        end
    end
    println(df_test_results)
end

#### parameters
global_params = NMFParams(
    init          = :random,
    dist          = Uniform(0, 1),
    max_iter      = 1_000,
    stopping_crit = :rel_cost,
    stopping_tol  = 1e-6,
    verbose       = 0,
    seed          = 0
)

#### # 1. MM ALGORITHM
####################################################################################################################

@testset "nmf_MM" begin
    βs = [-1, -0.5, 0, 0.5, 1, 1.5, 2, 3]
    αs = [0, 0.1, 1, 10, 100]

    local_params = FIParams(
        scale_W_iter = false,
        scale_W_last = false,
        l₁ratio_H    = 1,
        l₁ratio_W    = 1,
        α_W          = 0,
        α_H          = 0,
        alg          = :mm
    )

    run_FI_tests(global_params, local_params, βs, αs)
end

#### # 2. H ALGORITHM
####################################################################################################################

@testset "nmf_H" begin
    βs = [-1, -0.5, 0, 0.5, 1, 1.5, 2, 3]
    αs = [0, 0.1, 1, 10, 100]

    local_params = FIParams(
        scale_W_iter = false,
        scale_W_last = false,
        l₁ratio_H    = 1,
        l₁ratio_W    = 1,
        α_W          = 0,
        α_H          = 0,
        alg          = :h
    )

    run_FI_tests(global_params, local_params, βs, αs)
end

#### # 3. ME ALGORITHM
####################################################################################################################

@testset "nmf_ME" begin
    βs = [-1, 0, 0.5, 1.5, 2]
    αs = [0, 0.1, 1, 10, 100]

    local_params = FIParams(
        scale_W_iter = false,
        scale_W_last = false,
        l₁ratio_H    = 1,
        l₁ratio_W    = 1,
        α_W          = 0,
        α_H          = 0,
        alg          = :h
    )

    run_FI_tests(global_params, local_params, βs, αs)
end
