using Test
using VariantsNMF

printstyled("Julia version: $VERSION\n", color=:blue)
printstyled("Running tests: \n", color=:blue)

const tests = [
    "convergence",
    "clustering",
    "divergences",
    "nmf_FI",
    "rank_by_stability",
    "rank_by_consensus",
]

# test utilities to avoid redundancies
include("testutils.jl")

for t in tests
    @testset "Test $t" begin
        include("$t.jl")
    end
end
