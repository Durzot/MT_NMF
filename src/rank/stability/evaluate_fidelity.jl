"""
Created on Fri May 04 2020

@author: Yoann Pradat

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Algorithm for evaluating factorisation fidelity  in rank selection method in NMF based on average stability of latent 
factor across perturbed versions of the input matrix V.
"""

abstract type AbstractRSFidNMFResults <: AbstractNMFResults end

"""
    RSFidNMFResults(list_err, list_fid)

NMFResults returned by `evaluate_fidelity`.
"""
mutable struct RSFidNMFResults{T} <: AbstractRSFidNMFResults
    list_V_err::Vector{Matrix{T}}
    list_fid::Vector{Float64}

    function RSFidNMFResults{T}(
        list_V_err::Vector{Matrix{T}},
        list_fid::Vector{Float64}) where T <: Real

        new{T}(list_V_err, list_fid)
    end
end

"""
    evaluate_fidelity(list_nmf_results, list_V_pert)

Evaluates the fidelity of multiple runs of a NMF factorisation by compute the reconstruction error across
runs. Note that the reconstruction error is different from the cost minimized in the NMF algorithm in case
the latter includes regularization terms.

#Arguments
- list_nmf_results::Vector{NMFResults{T}} : vector containing results of each NMF run
- list_V_pert::Vector{Matrix{T}}       : vector containing the perturbed V matrix of each NMF run
"""
function evaluate_fidelity(list_nmf_results::Vector{NMFResults{T}}, list_V_pert::Vector{Matrix{T}}) where T <: Real
    list_V_err = Vector{Matrix{T}}()
    list_fid   = Vector{Float64}()

    for (nmf_results, V_pert) in zip(list_nmf_results, list_V_pert)
        V_appr = nmf_results.W * nmf_results.H
        V_err  = V_pert - V_appr

        if typeof(nmf_results.local_params) == FIParams
            fid = β_divergence(V_pert, V_appr, nmf_results.local_params.β)
        else
            if nmf_results.local_params.div == :α
                fid = α_divergence(V_pert, V_appr, nmf_results.local_params.α)
            elseif nmf_results.local_params.div == :β
                fid = β_divergence(V_pert, V_appr, nmf_results.local_params.β)
            end
        end

        push!(list_V_err, V_err)
        push!(list_fid, fid)
    end

    RSFidNMFResults{T}(
        list_V_err,
        list_fid
    )
end
