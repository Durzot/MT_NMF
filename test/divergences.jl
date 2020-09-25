using Test
using VariantsNMF

X1 = [0.21 0.40 0.4; 0.01 0.4 0.45]
Y1 = [0.12 0.48 0.4; 0.1 0.55 0.45]

α_values = [-0.5, 0, 0.33, 1, 1.5, 2, 2.8]
α_divg_values = [0.283580, 0.195769, 0.162286, 0.124183, 0.109648, 0.101371, 0.095786]
β_values = [-0.5, 0, 0.33, 1, 1.5, 2, 2.8]
β_divg_values = [7.81200, 1.65435, 0.64633, 0.12418, 0.04732, 0.02255, 0.00952]

# test α-divergence function
@testset "α_divergence" begin
    for (α, α_divg) in zip(α_values, α_divg_values)
        @test α_divergence(X1, X1, α) ≈ 0  atol = 1e-9
        @test α_divergence(Y1, Y1, α) ≈ 0  atol = 1e-9
        @test α_divergence(X1, Y1, α) ≈ α_divg atol = 1e-5
    end
end

# test β-divegence function
@testset "β_divergence" begin
    for (β, β_divg) in zip(β_values, β_divg_values)
        @test β_divergence(X1, X1, β) ≈ 0  atol = 1e-9
        @test β_divergence(Y1, Y1, β) ≈ 0  atol = 1e-9
        @test β_divergence(X1, Y1, β) ≈ β_divg atol = 1e-5
    end
end
