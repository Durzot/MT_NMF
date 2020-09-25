module VariantsNMF

using Clustering
using DataFrames
using Distances
using Distributed
using Distributions
using LinearAlgebra: diag, norm, svd, transpose
using LinearAlgebra.BLAS: dot
using Printf: @printf
using ProgressMeter
using Random
using Statistics: mean, cor

export β_divergence
export α_divergence

#### common
export NMFMetrics
export NMF
export NMFParams
export NMFResults
export initialize_nmf

include("aux/structs.jl")
include("aux/cluster.jl")
include("aux/convergence.jl")
include("aux/divergences.jl")
include("aux/initialize.jl")
include("aux/scale.jl")

### cichocki_09
export CParams
export nmf_α

include("solver/cichocki_09/params.jl")
include("solver/cichocki_09/cost.jl")
include("solver/cichocki_09/nmf_alpha.jl")

#### base MU, FI
export MUParams
export FIParams
export nmf_MU
export nmf_FI

include("solver/base/nmf_MU.jl")
include("solver/base/nmf_FI.jl")

####
#### rank
####

#### stability: alexandrov et al
export RSParams
export RSNMFResults
export rank_by_stability

include("rank/stability/bootstrap.jl")
include("rank/stability/evaluate_stability.jl")
include("rank/stability/evaluate_fidelity.jl")
include("rank/stability/rank_by_stability.jl")

#### consensus: brunet et al
export RCParams
export RCNMFResults
export rank_by_consensus

include("rank/consensus/consensus_metrics.jl")
include("rank/consensus/rank_by_consensus.jl")

end
