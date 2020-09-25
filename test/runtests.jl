include("../src/VNMF.jl")

using DataFrames
using Distances
using Distributions
using DelimitedFiles
using PyCall
using Random
using Test
using .VNMF

println("Julia version: ", VERSION)

function run_test(test)
    println("=" ^ 50)
    println("TEST: $test")
    include(test)
end

#### simulate matrices for testing NMF
F = 10
N = 25
K = 5
rng = MersenneTwister(123) 

#### truncated-normal random matrices
V = abs.(randn(rng, Float64, F, K)) * abs.(randn(rng,Float64, K, N)) + rand(rng, Uniform(0,1), F, N)

@time begin
@time @testset "divg tests" begin run_test("aux/divg_test.jl") end
@time @testset "cluster tests" begin run_test("aux/cluster_test.jl") end
@time @testset "rank stability" begin run_test("rank/rank_by_stability_breast_21.jl") end
@time @testset "nmf FI tests" begin run_test("solver/nmf_FI_test.jl") end
println("=" ^ 50)
end
