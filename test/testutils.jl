# Utilities for tests.

using Distributions
using Random
using Test: @test

function scale_col(X; p=1)
    norms_X = mapslices(x -> norm(x, p), X, dims=1)
    X .* repeat(norms_X .^ -1, size(X, 1), 1)
end

# Simulate W and H.
function simulate_WH(;F::Integer, N::Integer, n_factors::Integer, dist_W::Distribution, dist_H::Distribution, rng::AbstractRNG)
    K = n_factors

    W = rand(rng, dist_W, F, K)
    W = scale_col(W, p=1)

    H = rand(rng, dist_H, K, N)

    W, H
end
