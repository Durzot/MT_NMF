"""
Created on Sat Apr 11 2020

@author: Yoann Pradat

Implementation of the Maximization-Mimization algorithm under various constraints and for the β divergence.
"""

"""
    _γ_β(β)

Compute the exponent used in update rules of Maximization-Minimization algorithm.
"""
function _γ_β(β::Real)
    if β < 1
	γ = 1.0 / (2.0-β)
    elseif  β > 2
	γ = 1.0 / (β-1.0)
    else
	γ = 1
    end
    γ
end

"""
    _update_nmf_MM!(V, V_a, W, H, params)

Update the H and W matrices in that order using the update rules of Maximization-Minimization algorithm.
"""
function _update_nmf_MM!(V::Matrix{T}, V_a::Matrix{T}, W::Matrix{T}, H::Matrix{T}, params::FIParams) where T <: Real
    #### compute regularization
    #### for J_H = ||H||₁, ∇J_H = J_{K,N} with J the matrix filled with ones
    #### for J_H = 0.5*||H||₂, ∇J_H = H
    regu_H = H -> params.α_H * (params.l₁ratio_H * ones(size(H)) + (1. - params.l₁ratio_H) * H) / length(H)
    regu_W = W -> params.α_W * (params.l₁ratio_W * ones(size(W)) + (1. - params.l₁ratio_W) * W) / length(W)
    
    #### exponent for multiplicative updates
    γ = _γ_β(params.β)

    if params.β > 2
        #### update rules for β > 2
        W .= W .* (((V .* V_a .^ (params.β-2)) * transpose(H) - regu_W(W)) ./ (V_a .^ (params.β-1) * transpose(H))) .^ γ
        V_a .= W * H

        H .= H .* ((transpose(W) * (V .* V_a .^ (params.β-2)) - regu_H(H)) ./ (transpose(W) * V_a .^ (params.β-1))) .^ γ
        V_a .= W * H

    elseif params.β == 2
        #### update rules for β = 2 can be written more efficiently
        W .= W .* (max.(V * transpose(H) - regu_W(W), 0) ./ (W * (H * transpose(H))))
        H .= H .* (max.((transpose(W) * V - regu_H(H)), 0) ./ ((transpose(W) * W) * H))
        V_a .= W * H

    elseif 1 < params.β < 2
        #### update rules for 1 < β < 2
        #### the update of h_kn is obtained by solving
        #### \sum_f w_{fk} [ (\tilde{v}_f \frac{h_k}{\tilde{h}_k})^{\beta-1} - v_f (\tilde{v}_f \frac{h_k}{\tilde{h}_k})^{\beta-2} ] + \alpha_h \nabla_{h_k} J_k (h_k) = 0 with J_k the component of J_h(h) = \sum_k J_{k}(h_k)
        
        if params.α_W == 0
            W .= W .* (((V .* V_a .^ (params.β-2)) * transpose(H)) ./ (V_a .^ (params.β-1) * transpose(H)))
        else
            #### solving a X^2 + b X - c = 0 with X = \frac{h_k}{\tilde{h}_k}^{\beta-1}
            a = V_a .^ (params.β-1) * transpose(H)
            b = regu_W(W)
            c = (V .* V_a .^ (params.β-2)) * transpose(H)
            Δ =  b .^ 2 + 4a .* c 

            W .= W .* ((- b + sqrt.(Δ)) ./ (2a)) .^ (1. / (params.β-1))
        end

        V_a .= W * H

        if params.α_H == 0
            H .= H .* ((transpose(W) * (V .* V_a .^ (params.β-2))) ./ (transpose(W) * V_a .^ (params.β-1)))
        else
            #### see above
            a = transpose(W) * V_a .^ (params.β-1) 
            b = regu_H(H)
            c = transpose(W) * (V .* V_a .^ (params.β-2))
            Δ =  b .^ 2 + 4a .* c 

            H .= H .* ((- b + sqrt.(Δ)) ./ (2 .* a)) .^ (1. / (params.β-1))
        end
        
        V_a .= W * H

    elseif params.β <= 1
        #### update rules for β <= 1
        W .= W .* (((V .* V_a .^ (params.β-2)) * transpose(H)) ./ (V_a .^ (params.β-1) * transpose(H) + regu_W(W))) .^ γ
        V_a .= W * H

        H .= H .* ((transpose(W) * (V .* V_a .^ (params.β-2))) ./ (transpose(W) * V_a .^ (params.β-1) + regu_H(H))) .^ γ
        V_a .= W * H
    end

    #### set to 0 values below eps(T)
    W[W .< eps(T)] .= 0
    H[H .< eps(T)] .= 0
    V_a .= W * H
    
    if params.scale_W
        cs_W = sum(W, dims=1)

        #### one or more columns of W may be null vectors
        indices = (1:size(W,2))[vec(cs_W .> 0)]

        W[:, indices] .= W[:, indices] .* repeat(cs_W[:, indices] .^ -1, size(W, 1), 1)
        H[indices, :] .= H[indices, :] .* repeat(transpose(cs_W[:, indices]), 1, size(H, 2))
    end
end

"""
    nmf_MM(V, W_init, H_init, params)

Run the Maximization-Minimization algorithm using the set of input parameters. For the parameters, see [`FIParams`](@ref)

# References
- C. Fevotte & J. Idier, "Algorithms for nonnegative matrix factorization
with the beta-divergence ", Neural Compuation, 2011.
"""
function nmf_MM(V::Matrix{T}, W_init::Matrix{T}, H_init::Matrix{T}, params::AbstractNMFParams, local_params::AbstractFIParams) where T <: Real
    W       = W_init[:,:]
    H       = H_init[:,:]
    V_a     = W * H

    #### metrics and stopping_crit
    metrics            = NMFMetrics(V, W_init, H_init, _β_cost, params, local_params)
    stopping_crit_met  = false
    iteration_is_sane  = true

    while !stopping_crit_met && iteration_is_sane 
        #### update V_a, W, H
        _update_nmf_MM!(V, V_a, W, H, local_params)

        #### update metrics 
        _update_metrics!(metrics, V, W, H, local_params)

        #### check sanity of iteration 
        iteration_is_sane = _check_sanity_iter(W, H)

        #### check the stopping criterion
        stopping_crit_met = _check_stopping_crit(metrics, params)

        #### print intermediate if verbose
        if params.verbose && (metrics.cur_iter % (params.max_iter ÷ 10) == 0)
            @printf("iter %d/%d | cost: %.5g\n", metrics.cur_iter, params.max_iter, metrics.absolute_cost[metrics.cur_iter])
        end
    end

    #### convergence information
    if !iteration_is_sane
        @warn "stopped nmf because of sanity check not passed at iteration" metrics.cur_iter
    else
        if metrics.cur_iter == params.max_iter && params.stopping_crit != :none
            @warn "maximum number of iterations reached"
        end
        if metrics.cur_iter < params.max_iter
            @info "stopping criterion met at" metrics.cur_iter
        end
    end

    NMFResults{T}(W, H, metrics, local_params, params)
end
