# Utilities for tests.

using Distributions
using LinearAlgebra: norm
using Random
using Test: @test

# Add noise on top of one factorisation WH
function add_matrix_noise(;WH::Matrix{T},
                        dist_E::Distribution=Uniform(0,1), 
                        rng::AbstractRNG=MersenneTwister(1234)) where T <: Real
    max.(WH + rand(rng, dist_E, size(WH,1), size(WH, 2)), 0)
end

# Simulate one matrix factorisation V = W*H + E
function get_one_simulated_V()
    F, K, N, rng = 10, 5, 25, MersenneTwister(1234)
    W, H = rand(rng, Uniform(0,1), F, K), rand(rng, Uniform(0,1), K, N)

    add_matrix_noise(WH=W*H, dist_E=Uniform(0,0.1), rng=rng)
end

# Simulate different noises on top of one factorisation WH
function get_many_simulated_V()
    F, K, N = 10, 5, 25
    rng = MersenneTwister(1234)
    W, H = rand(rng, Uniform(0,1), F, K), rand(rng, Uniform(0,1), K, N)

    #### simulate different factorisation
    noise_Vs = Matrix{Float64}[] 
    noise_names = String[] 

    push!(noise_Vs, add_matrix_noise(WH=W*H, dist_E=Uniform(0,0.01), rng=rng))
    push!(noise_names, "uniform_small")
    push!(noise_Vs, add_matrix_noise(WH=W*H, dist_E=Uniform(0,0.2), rng=rng))
    push!(noise_names, "uniform_large")
    push!(noise_Vs, add_matrix_noise(WH=W*H, dist_E=Normal(0,0.01), rng=rng))
    push!(noise_names, "normal_small")
    push!(noise_Vs, add_matrix_noise(WH=W*H, dist_E=Normal(0,0.2), rng=rng))
    push!(noise_names, "normal_large")
    push!(noise_Vs, add_matrix_noise(WH=W*H, dist_E=Gamma(0.05,1), rng=rng))
    push!(noise_names, "gamma_small")
    push!(noise_Vs, add_matrix_noise(WH=W*H, dist_E=Gamma(0.05,4), rng=rng))
    push!(noise_names, "gamma_large")

    noise_Vs, noise_names
end


# Simulate V, W and H.
function simulate_VWH(;F::Integer, 
                      N::Integer,
                      K::Integer,
                      dist_W::Distribution=Uniform(0,1), 
                      dist_H::Distribution=Uniform(0,1), 
                      dist_E::Distribution=Uniform(0,0.1), 
                      rng::AbstractRNG=MersenneTwister(1234))
    W = rand(rng, dist_W, F, K)
    H = rand(rng, dist_H, K, N)
    E = rand(rng, dist_E, F, N)

    W*H+E, W, H
end
