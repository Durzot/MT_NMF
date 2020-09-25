"""
Created on Sat Apr 1 2020

@author: Yoann Pradat

Test cost functions of VNMF module.
"""

X1 = [0.21 0.40 0.4; 0.01 0.4 0.45]
Y1 = [0.12 0.48 0.4; 0.1 0.55 0.45]

β_divg_test = [7.81200, 1.65435, 0.64633, 0.12418, 0.04732, 0.02255, 0.00952]

@testset "β_divergence" begin
    i = 0
    for β in [-0.5, 0, 0.33, 1, 1.5, 2, 2.8]
        i += 1
        @test β_divergence(X1, Y1, β) ≈ β_divg_test[i] atol = 1e-5
	@test β_divergence(X1, X1, β) ≈ 0  atol = 1e-9
    end
end

α_divg_test = [0.283580, 0.195769, 0.162286, 0.124183, 0.109648, 0.101371, 0.095786]

@testset "α_divergence" begin
    i = 0
    for α in [-0.5, 0, 0.33, 1, 1.5, 2, 2.8]
        i += 1
        @test α_divergence(X1, Y1, α) ≈ α_divg_test[i] atol = 1e-5
	@test α_divergence(X1, X1, α) ≈ 0  atol = 1e-9
    end
end
