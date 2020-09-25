"""
Created on Thu Apr 16 2020

@author: Yoann Pradat

Test Fevotte and Idier NMF algoritms.
"""

function copy_FIParams(params)
    FIParams(
        β            = params.β,
        scale_W_iter = params.scale_W_iter,
        scale_W_last = params.scale_W_iter,
        α_W          = params.α_W,
        α_H          = params.α_H,
        l₁ratio_H    = params.l₁ratio_H,
        l₁ratio_W    = params.l₁ratio_W,
        θ            = params.θ,
        alg          = params.alg
    )
end

function test_convergence(β, local_params)
    params = copy_FIParams(local_params)
    params.β = β
    res = nmf_FI(V, global_params, params)
    issorted(res.metrics.absolute_cost, rev=true)
end

function test_reg_H(β, αs, local_params)
    params = copy_FIParams(local_params)
    params.β = β
    s_H = Array{Float64,1}()
    for α_H in αs
        params.α_H = α_H
        res = nmf_FI(V, global_params, params)
        push!(s_H, VNMF.matr_avg_row_sparseness(res.H))
    end
    issorted(s_H)
end

function test_reg_W(β, αs, local_params)
    params = copy_FIParams(local_params)
    params.β = β
    s_W = Array{Float64,1}()
    for α_W in αs
        params.α_W = α_W
        res = nmf_FI(V, global_params, params)
        push!(s_W, VNMF.matr_avg_col_sparseness(res.W))
    end
    issorted(s_W)
end

function run_FI_tests(global_params, local_params, βs, αs)
    
    df_test_results = DataFrame(
        β = Float64[], 
        test_cvg_passed = Bool[], 
        test_reg_H_passed = Bool[], 
        test_reg_W_passed = Bool[]
    )

    for β in βs
        cvg_passed = test_convergence(β, local_params)
        reg_H_passed = test_reg_H(β, αs, local_params)
        reg_W_passed = test_reg_W(β, αs, local_params)
        push!(df_test_results, [β cvg_passed reg_H_passed reg_W_passed])
    end

    println(df_test_results)

    @test all(df_test_results.test_cvg_passed)
    @test all(df_test_results.test_reg_H_passed)
    @test all(df_test_results.test_reg_W_passed)
end


#### parameters
global_params = NMFParams(
    init          = :random,
    dist          = Uniform(0, 1),
    max_iter      = 1_000,
    stopping_crit = :none,
    verbose       = false,
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
