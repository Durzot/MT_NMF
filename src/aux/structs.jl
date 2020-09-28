"""
Created on Sun May 03 2020

@author: Yoann Pradat

Structs used throughout NMF code.
"""

#### # PARAMETERS
#### # #########################################################################################################

abstract type AbstractNMFParams end

"""
    NMFParams([,rank, init, seed, max_iter, stopping_crit, stopping_tol, verbose])

Set of basic parameters passed to the various NMF algorithms.

# Arguments
- `rank::Integer`         : rank of the factorisation
- `init::Symbol`          : initialization method. See [`_initialize_nmf`](@ref)
- `dist::Distribution`    : distribution for random initialization. See [`_initialize_nmf`](@ref)
- `seed::Integer`         : seed for random initializations
- `max_iter::Integer`     : maximum number of iterations (in main procedure)
- `stopping_crit::Symbol` : criterion for stopping iterations. One of :none, :conn, :rel_cost or :abs_cost
- `stopping_iter::Symbol` : number of successive iterations with criterion satisfied for calling convergence
- `stopping_tol::Real`    : tolerance for the stopping crit if crit is :cost
- `verbose::Integer`      : 0 for silent, 1 for warning and 2 and more for to show intermediate information
"""
mutable struct NMFParams <: AbstractNMFParams
    rank::Integer
    init::Symbol
    dist::Distribution
    seed::Integer
    max_iter::Integer
    stopping_crit::Symbol
    stopping_iter::Integer
    stopping_tol::Real
    verbose::Integer

    function NMFParams(;
        rank::Integer          = 5,
        init::Symbol           = :random,
        dist::Distribution     = truncated(Normal(0,1), 0, Inf),
        seed::Integer          = 0,
        max_iter::Integer      = 1000,
        stopping_crit::Symbol  = :none,
        stopping_iter::Integer = 20,
        stopping_tol::Real     = 1e-5,
        verbose::Integer       = 1)

        valid_init = [:default, :random, :nndsvd, :nndsvda, :nndsvdar]
        valid_stopping_crit = [:none, :conn, :rel_cost, :abs_cost] 

        rank > 1  || throw(ArgumentError("rank must be at least 2"))
        max_iter > 1  || throw(ArgumentError("max_iter must be at least 2"))
        stopping_tol > 0 || throw(ArgumentError("stopping_tol must be > 0"))
        stopping_iter > 0 || throw(ArgumentError("stopping_iter must be > 0"))
        init in valid_init  || throw(ArgumentError(string("init should be one of ", valid_init)))
        stopping_crit in valid_stopping_crit || throw(ArgumentError(string("stopping_crit should be one of ", valid_stopping_crit)))

        new(rank, init, dist, seed, max_iter, stopping_crit, stopping_iter, stopping_tol, verbose)
    end
end

#### # METRICS
#### # #########################################################################################################

"""
    NMFMetrics(V, W_init, H_init, func_cost, params, local_params)

Struct for tracking cost-related metrics.
"""

mutable struct NMFMetrics
    cur_iter::Integer
    cur_cost::Real
    cur_divg::Real
    cur_conn::Matrix{Int64}
    unchange_conn::Array{Bool,1}
    absolute_cost::Array{Real,1}
    absolute_divg::Array{Real,1}
    relative_cost::Array{Real,1}
    func_cost::Function
    func_divg::Function

    function NMFMetrics(
        V::Matrix{T},
        W_init::Matrix{T},
        H_init::Matrix{T},
        func_cost::Function,
        func_divg::Function) where T <: Real

        unchange_conn = Array{Bool,1}()
        absolute_cost = Array{Real,1}()
        absolute_divg = Array{Real,1}()
        relative_cost = Array{Real,1}()

        cur_iter = 1
        cur_cost = func_cost(V, W_init, H_init)
        cur_divg = func_divg(V, W_init, H_init)
        cur_conn = ones(Int64, size(V, 2), size(V, 2))
        push!(absolute_cost, cur_cost)
        push!(absolute_divg, cur_divg)

        new(cur_iter, cur_cost, cur_divg, cur_conn, unchange_conn, absolute_cost, absolute_divg, relative_cost, func_cost, func_divg)
    end
end

#### # RESULTS
#### # #########################################################################################################

abstract type AbstractNMFResults end

"""
    NMFResults(W, H, metrics, global_params, local_params)

NMFResults returned by NMF algorithms.

# Arguments
- `W             :  : Matrix{T}`         : W matrix after last iteration
- `H             :  : Matrix{T}`         : H matrix after last iteration
- `metrics       :  : NMFMetrics`           : see [`NMFMetrics`](@ref)
- `global_params :  : AbstractNMFParams` : see [`NMFParams`](@ref)
- `local_params  :  : AbstractNMFParams` : see [`FIParams`](@ref), [`CParams`](@ref), [`MUParams`](@ref)
"""
mutable struct NMFResults{T} <: AbstractNMFResults
    W::Matrix{T}   
    H::Matrix{T}
    metrics::NMFMetrics
    global_params::AbstractNMFParams
    local_params::AbstractNMFParams
    converged::Bool

    function NMFResults{T}(
        W::Matrix{T}, 
        H::Matrix{T}, 
        metrics::NMFMetrics, 
        global_params::AbstractNMFParams, 
        local_params::AbstractNMFParams,
        converged::Bool) where T <: Real

        new{T}(W, H, metrics, global_params, local_params, converged)
    end
end

#### # STRUCT FOR STORING NMF
#### # #########################################################################################################

mutable struct NMF
    solver::Function
    global_params::AbstractNMFParams
    local_params::AbstractNMFParams

    function NMF(;solver::Function, global_params::AbstractNMFParams, local_params::AbstractNMFParams)
        new(solver, global_params, local_params)
    end
end

#### overload copy function for NMF struct
Base.copy(nmf::NMF) = NMF(;Dict(n => deepcopy(getfield(nmf, n)) for n ∈ fieldnames(NMF))...)
