"""
Created on Thu Apr 30 2020

@author: Yoann Pradat

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Rank selection method in NMF based on average stability of latent factor across perturbed versions of the input matrix.
matrix V.
Alexandrov et al, Deciphering Signatures of Mutational Processes Operative in Human Cancer. 2013, Cell Reports.
"""

"""
    _get_perturbed_matrix_multi(V, rng)

Draws a perturbed version of the input matrix V using multinomial distributions.
"""
function _get_perturbed_matrix_multi(V::Matrix{T}, rng::AbstractRNG) where T <: Real
    #### normalize each column to sum 1
    V_norm = V .* repeat(sum(V, dims=1) .^ -1, size(V, 1), 1)
    
    #### draw from multinomial
    #### WARNING: incorrect for input V that does not contain integer entries
    V_pert = zeros(Int64, size(V))

    for j = 1:size(V_pert, 2)
        dist = Multinomial(convert(Integer, sum(V[:,j])), V_norm[:, j])
        V_pert[:, j] = rand(rng, dist)
    end

    convert.(T, V_pert)
end

"""
    _get_perturbed_matrix_noise(V, dist, rng[, force_positivity])

Draws a perturbed version of the input matrix V adding random noise drawn from the specified distribution.
"""
function _get_perturbed_matrix_noise(V::Matrix{T}, dist::Distribution, rng::AbstractRNG; force_positivity::Bool=true) where T <: Real
    V_nois = rand(rng, dist, size(V))
    V_sign = sign.(rand(rng, size(V)...) .- 0.5)
    V_pert = V + V_sign .* V_nois

    force_positivity ? max.(V_pert, 0) : V_pert
end 

"""
    _get_perturbed_matrix(V, n_iter, pert_meth, pert_dist, rng)

Draws `n_iter` perturbed versions of the input matrix V adding random noise or sampling from the specified distribution.
"""
function _get_perturbed_matrix(V::Matrix{T}, pert_meth::Symbol, pert_dist::Union{Nothing, Distribution}, rng::AbstractRNG) where T <: Real
    if pert_meth == :multinomial
        V_pert = _get_perturbed_matrix_multi(V, rng)
    elseif pert_meth == :noise
        V_pert = _get_perturbed_matrix_noise(V, pert_dist, rng)
    end
    V_pert
end

