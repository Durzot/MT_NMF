"""
Created on Wed Apr 15 2020

@author: Yoann Pradat

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Implementation of the Maximum-Equalization algorithms and its variants under various constraints and for the β 
divergence as presented in C. Fevotte & J. Idier, "Algorithms for nonnegative matrix factorization
with the beta-divergence ", Neural Compuation, 2011.
"""

abstract type AbstractFIParams <: AbstractNMFParams end

"""
    FIParams([,β, scale_W_iter, scale_W_last, α_H, α_W, l₁ratio_H, l₁ratio_W, θ, alg, seed])

Local parameters passed to `nmf_FI`.

# Arguments
- `β::Real`               : parameter of the β-divergence
- `scale_W_iter::Bool`    : scale the components of W to have unit l₁ norm during iterations
- `scale_W_last::Bool`     : scale the components of W to have unit l₁ norm after last iteration
- `α_H::Real`             : constant that multiplies the regularization term on H
- `α_W::Real`             : constant that multiplies the regularization term on W
- `l₁ratio_H::Real`       : l₁/l₂ regularization balance for H (in [0; 1]). 1 is l₁ only, 0 is l₂ only
- `l₁ratio_W::Real`       : l₁/l₂ regularization balance for W (in [0; 1]). 1 is l₁ only, 0 is l₂ only
- `θ::Real`               : θ for mixed update between ME and H. Used only in `nmf_ME`.
- `alg::Symbol`           : symbol to choose the algorithm. Choose
    + :mm for Maximization-Minimization
    + :h  for Heuristic (Maximization-Minimization without exponent)
    + :me for Maximization-Equalization
"""
mutable struct FIParams <: AbstractFIParams
    β::Real
    scale_W_iter::Bool
    scale_W_last::Bool
    α_H::Real
    α_W::Real
    l₁ratio_H::Real
    l₁ratio_W::Real
    θ::Real
    alg::Symbol

    function FIParams(;
        β::Real               = 2,
        scale_W_iter::Bool    = true,
        scale_W_last::Bool    = true,
        α_H::Real             = 0,
        α_W::Real             = 0,
        l₁ratio_H::Real       = 0,
        l₁ratio_W::Real       = 0,
        θ::Real               = 0.95,
        alg::Symbol           = :mm)

        valid_α_H = α_H >= 0
        valid_α_W = α_W >= 0
        valid_l₁ratio_H = l₁ratio_H >= 0 && l₁ratio_H <= 1
        valid_l₁ratio_W = l₁ratio_W >= 0 && l₁ratio_W <= 1
        valid_θ = θ >= 0 && θ <= 1
        valid_alg = alg in [:mm :me :h]

        valid_α_H || throw(ArgumentError("α_H must be > 0"))
        valid_α_W || throw(ArgumentError("α_W must be > 0"))
        valid_l₁ratio_H || throw(ArgumentError("l₁ratio_H must be in [0,1]"))
        valid_l₁ratio_W || throw(ArgumentError("l₁ratio_W must be in [0,1]"))
        valid_θ  || throw(ArgumentError("θ must be between 0 and 1"))
        valid_alg || throw(ArgumentError("alg must be one of :mm, :h or :me"))

        new(β, scale_W_iter, scale_W_last, α_H, α_W, l₁ratio_H, l₁ratio_W, θ, alg)
    end
end

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
    _β_cost(V, W, H, params)

Compute the cost function that is minimized by the NMF alogrithm running with parameters params. It adds up
the β_divergence and regularization terms.

See also: [`β_divergence`](@ref), [`FIParams`](@ref)
"""
function _β_cost(V::Matrix{T}, W::Matrix{T}, H::Matrix{T}, params::FIParams) where T <: Real
    regu_H = params.α_H * (params.l₁ratio_H * norm(H, 1) + (1. - params.l₁ratio_H) * norm(H, 2)) / length(H)
    regu_W = params.α_W * (params.l₁ratio_W * norm(W, 1) + (1. - params.l₁ratio_W) * norm(W, 2)) / length(W)
    β_div  = β_divergence(V, W*H, params.β)

    β_div + regu_H + regu_W
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

    #### avoid underflow
    W .= max.(W, eps(T))
    H .= max.(H, eps(T))
    V_a .= W * H
    
    if params.scale_W_iter
        scale_col_W!(W, H, 1)
    end
end

"""
    _update_nmf_H!(V, V_a, W, H, params)

Update the H and W matrices in that order using the update rules of Heuristic algorithm.
"""
function _update_nmf_H!(V::Matrix{T}, V_a::Matrix{T}, W::Matrix{T}, H::Matrix{T}, params::AbstractFIParams) where T <: Real
    #### compute regularization
    #### for J_H = ||H||₁, ∇J_H = J_{K,N} with J the matrix filled with ones
    #### for J_H = 0.5*||H||₂, ∇J_H = H
    regu_H = H -> params.α_H * (params.l₁ratio_H * ones(size(H)) + (1. - params.l₁ratio_H) * H) / length(H)
    regu_W = W -> params.α_W * (params.l₁ratio_W * ones(size(W)) + (1. - params.l₁ratio_W) * W) / length(W)
    
    if params.β > 2
        #### update rules for β > 2
        W .= W .* (((V .* V_a .^ (params.β-2)) * transpose(H) - regu_W(W)) ./ (V_a .^ (params.β-1) * transpose(H)))
        V_a .= W * H

        H .= H .* ((transpose(W) * (V .* V_a .^ (params.β-2)) - regu_H(H)) ./ (transpose(W) * V_a .^ (params.β-1)))
        V_a .= W * H

    elseif params.β == 2
        #### update rules for β = 2 can be written more efficiently
        W .= W .* ((V * transpose(H) - regu_W(W)) ./ (W * (H * transpose(H))))
        H .= H .* ((transpose(W) * V - regu_H(H)) ./ ((transpose(W) * W) * H))
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

            W .= W .* ((- b + sqrt.(Δ)) ./ (2 .* a)) .^ (1. / (params.β-1))
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

            H .= H .* ((- b + sqrt.(Δ)) ./ (2a)) .^ (1. / (params.β-1))
        end
        
        V_a .= W * H

    elseif params.β <= 1
        #### update rules for β <= 1
        W .= W .* (((V .* V_a .^ (params.β-2)) * transpose(H)) ./ (V_a .^ (params.β-1) * transpose(H) + regu_W(W)))
        V_a .= W * H

        H .= H .* ((transpose(W) * (V .* V_a .^ (params.β-2))) ./ (transpose(W) * V_a .^ (params.β-1) + regu_H(H)))
        V_a .= W * H
    end

    #### avoid underflow
    W .= max.(W, eps(T))
    H .= max.(H, eps(T))
    V_a .= W * H
    
    if params.scale_W_iter
        scale_col_W!(W, H, 1)
    end
end


"""
    _update_nmf_ME!(V, V_a, W, H, params)

Update the H and W matrices in that order using the update rules of Maximization-Equalization algorithm.
"""
function _update_nmf_ME!(V::Matrix{T}, V_a::Matrix{T}, W::Matrix{T}, H::Matrix{T}, params::FIParams) where T <: Real
    #### compute regularization
    #### for J_H = ||H||₁, ∇J_H = J_{K,N} with J the matrix filled with ones
    #### for J_H = 0.5*||H||₂, ∇J_H = H
    regu_H = H -> params.α_H * (params.l₁ratio_H * ones(size(H)) + (1. - params.l₁ratio_H) * H) / length(H)
    regu_W = W -> params.α_W * (params.l₁ratio_W * ones(size(W)) + (1. - params.l₁ratio_W) * W) / length(W)

    if params.β == -1
        #### ME always defined
        W_h = W .* (((V .* V_a .^ (params.β-2)) * transpose(H)) ./ (V_a .^ (params.β-1) * transpose(H) + regu_W(W)))
        W .= ((W_h .^ 2 + 8 * W_h .* W) .^ .5 + W_h) / 4
        V_a .= W * H

        H_h = H .* ((transpose(W) * (V .* V_a .^ (params.β-2))) ./ (transpose(W) * V_a .^ (params.β-1) + regu_H(H)))
        H .= ((H_h.^2 + 8 * H_h .* H) .^ .5 + H_h) / 4
        V_a .= W * H

    elseif params.β == 0
        #### ME always defined, equal to H
        W .= W .* (((V .* V_a .^ (params.β-2)) * transpose(H)) ./ (V_a .^ (params.β-1) * transpose(H) + regu_W(W)))
        V_a .= W * H

        H .= H .* ((transpose(W) * (V .* V_a .^ (params.β-2))) ./ (transpose(W) * V_a .^ (params.β-1) + regu_H(H)))
        V_a .= W * H

    elseif params.β == 0.5
        #### ME always defined
        W_h = W .* (((V .* V_a .^ (params.β-2)) * transpose(H)) ./ (V_a .^ (params.β-1) * transpose(H) + regu_W(W)))
        W .= (sqrt.(W + 8 * W_h) - sqrt.(W)) .^ 2 / 4
        V_a .= W * H

        H_h = H .* ((transpose(W) * (V .* V_a .^ (params.β-2))) ./ (transpose(W) * V_a .^ (params.β-1) + regu_H(H)))
        H .= (sqrt.(H + 8 * H_h) - sqrt.(H)) .^2 / 4
        V_a .= W * H

    elseif params.β == 1.5
        #### mixed update prolonged ME and MM
	
        #### compute W update from Heuristic algorithm
        if params.α_W == 0
            W_h = W .* (((V .* V_a .^ (params.β-2)) * transpose(H)) ./ (V_a .^ (params.β-1) * transpose(H)))
        else
            a = V_a .^ (params.β-1) * transpose(H)
            b = regu_W(W)
            c = (V .* V_a .^ (params.β-2)) * transpose(H)
            Δ =  b .^ 2 + 4a .* c 

            W_h = W .* ((- b + sqrt.(Δ)) ./ (2a)) .^ (1. / (params.β-1))
        end

        #### compute mixed update
        DW = 12W_h - 3W
        IW = 3W_h .< W
        W .= real.((sqrt.(Complex.(DW)) - sqrt.(W)) .^2 / 4)
        W[IW] .= 0
        W .= W_h + params.θ * (W - W_h)
        V_a .= W * H              

        #### compute H update from Heuristic algorithm
        if params.α_H == 0
            H_h = H .* ((transpose(W) * (V .* V_a .^ (params.β-2))) ./ (transpose(W) * V_a .^ (params.β-1)))
        else
            a = transpose(W) * V_a .^ (params.β-1) 
            b = regu_H(H)
            c = transpose(W) * (V .* V_a .^ (params.β-2))
            Δ =  b .^ 2 + 4a .* c 

            H_h = H .* ((- b .+ sqrt.(Δ)) ./ (2a)) .^ (1. / (params.β-1))
        end

        #### compute mixed update
        DH = 12H_h - 3H
        IH = 3H_h .< H
        H .= real.((sqrt.(Complex.(DH)) - sqrt.(H)) .^2 / 4)
        H[IH] .= 0
        H .= H_h + params.θ * (H - H_h)
        V_a .= W * H
    
    elseif params.β == 2
        #### mixed update prolonged ME and MM
        W_h = W .* ((V * transpose(H) - regu_W(W)) ./ (W * (H * transpose(H))))
        W .= 2W_h - W
        W[W .<= 0] .= 0
        W .= W_h + params.θ * (W - W_h)
        V_a .= W * H              

        H_h = H .* ((transpose(W) * V - regu_H(H)) ./ ((transpose(W) * W) * H))
        H .= 2H_h - H
        H[H .<= 0] .= 0
        H .= H_h + params.θ * (H - H_h)
        V_a .= W * H              
    
    elseif params.β == 3
        #### mixed update prolonged ME and MM
        W_h = W .* (((V .* V_a .^ (params.β-2)) * transpose(H) - regu_W(W)) ./ (V_a .^ (params.β-1) * transpose(H)))
        DW = W .* (12W_h - 3W)
        IW = 3W_h .< W 
        W .= real.(sqrt.(Complex.(DW)) - W) / 2
        W[IW] .= 0
        W .= W_h + params.θ * (W - W_h)
        V_a .= W * H

        H_h = H .* ((transpose(W) * (V .* V_a .^ (params.β-2)) - regu_H(H)) ./ (transpose(W) * V_a .^ (params.β-1)))

        DH = H .* (12H_h - 3H)
        IH = 3H_h .< H
        H .= real.(sqrt.(Complex.(DH)) - H) / 2
        H[IH] .= 0
        H .= H_h + params.θ * (H - H_h)

        V_a .= W * H
    else
        throw(ArgumentError("ME algorithm not implemented for β=$(params.β)"))
    end

    #### avoid underflow
    W .= max.(W, eps(T))
    H .= max.(H, eps(T))
    V_a .= W * H
    
    if params.scale_W_iter
        scale_col_W!(W, H, 1)
    end
end

"""
    _nmf_FI(V, W_init, H_init, global_params, local_params)

Helper function for `nmf_FI`. Runs the FI algorithms with given intial matrices W and H.
"""
function _nmf_FI(V::Matrix{T}, W_init::Matrix{T}, H_init::Matrix{T}, global_params::AbstractNMFParams, local_params::AbstractFIParams) where T <: Real
    W       = W_init[:,:]
    H       = H_init[:,:]
    V_a     = W * H

    #### metrics and stopping_crit
    func_cost          = (V, W, H) -> _β_cost(V, W, H, local_params)
    func_divg          = (V, W, H) -> β_divergence(V, W * H, local_params.β)
    metrics            = NMFMetrics(V, W_init, H_init, func_cost, func_divg)
    stopping_crit_met  = false
    iteration_is_sane  = true

    #### scale W before first iter if specified
    if local_params.scale_W_iter
        scale_col_W!(W, H, 1)
    end

    while !stopping_crit_met && iteration_is_sane 
        #### update V_a, W, H
        if local_params.alg == :mm
            _update_nmf_MM!(V, V_a, W, H, local_params)
        elseif local_params.alg == :h
            _update_nmf_H!(V, V_a, W, H, local_params)
        elseif local_params.alg == :me
            _update_nmf_ME!(V, V_a, W, H, local_params)
        end

        #### update metrics 
        _update_metrics!(metrics, V, W, H, global_params.stopping_crit)

        #### check sanity of iteration looking for potential divisions by zeros 
        iteration_is_sane = _check_sanity_iter(W, H)

        #### check the stopping criterion
        stopping_crit_met = _check_stopping_crit(metrics, global_params)

        #### print intermediate if verbose
        if global_params.verbose && global_params.max_iter < 10
            @printf("iter %d/%d | cost: %.5g\n", metrics.cur_iter, global_params.max_iter, metrics.cur_cost)
        elseif global_params.verbose && (metrics.cur_iter % (global_params.max_iter ÷ 10) == 0)
            @printf("iter %d/%d | cost: %.5g\n", metrics.cur_iter, global_params.max_iter, metrics.cur_cost)
        end
    end

    if local_params.scale_W_last
        scale_col_W!(W, H, 1)
    end

    #### convergence information
    if !iteration_is_sane
        @warn "Stopped nmf because sanity check not passed at iteration" metrics.cur_iter
    else
        if metrics.cur_iter == global_params.max_iter && global_params.stopping_crit != :none
            @warn "Maximum number of iterations reached"
        end
        if global_params.verbose && metrics.cur_iter < global_params.max_iter
            @info "Stopping criterion $(global_params.stopping_crit) met at iter $(metrics.cur_iter)"
        end
    end

    NMFResults{T}(W, H, metrics, global_params, local_params)
end

"""
    nmf_FI(V, global_params, local_params)

Run NMF algorithm using the set of input parameters. For the parameters, see [`Params`](@ref) and [`FIParams`](@ref).

# References
- C. Fevotte & J. Idier, "Algorithms for nonnegative matrix factorization
with the beta-divergence ", Neural Compuation, 2011.
"""
function nmf_FI(V::Matrix{T}, global_params::AbstractNMFParams, local_params::AbstractFIParams) where T <: Real
    # initalization
    W_init, H_init = initialize_nmf(
        V, 
        rank = global_params.rank,
        init = global_params.init,
        dist = global_params.dist,
        seed = global_params.seed
    )

    # algorithm
    _nmf_FI(V, W_init, H_init, global_params, local_params)
end
