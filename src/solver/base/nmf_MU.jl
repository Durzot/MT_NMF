"""
Created on Thu Apr 23 2020

@author: Yoann Pradat

Multiplicative update algorithsm
"""

abstract type AbstractMUParams <: AbstractNMFParams end

"""
    MUParams([, α, β, ϵ, δ, σ, scale_W_iter, scale_W_last, alg])

Parameters passed to `nmf_MM`, `nmf_H` and `nmf_ME`.

# Arguments
- `α::Real`             : parameter of the α-divergence
- `β::Real`             : parameter of the β-divergence
- `ϵ::Real`             : threshold for replacing 0 values.
- `δ::Real`             : parameter in :mu_mod algorithm.
- `scale_W_iter::Bool`  : scale the components of W to have unit l₁ norm during iterations
- `scale_W_last::Bool`  : scale the components of W to have unit l₁ norm after last iteration
- `σ::Real`             : parameter in :mu_mod algorithm.
- `div::Symbol`         : symbol to choose the divergence. Either :α or :β
- `alg::Symbol`         : symbol to choose the algorithm. One of :mu, :mu_mod
    + mu: Multiplicative updates (MU)
        Reference for Amari alpha divergence:
            A.Cichocki, S.Amari, R.Zdunek, R.Kompass, G.Hori, and Z.He,
            "Extended SMART algorithms for non-negative matrix factorization,"
            Artificial Intelligence and Soft Computing, 2006.

            min D(V||W*H)

            where d(x|y) = | (x^α y^(1-α) - αx + (α-1)y) ÷ (α(α-1))  (α ∉ {0, 1})
                           |
                           | xlog(x÷y) - x + y                       (α = 1)
                           |
                           | ylog(y÷x) - y + x                       (α = 0)

            - Pearson's distance (α=2)
            - Hellinger's distance (α=0.5)
            - Neyman's chi-square distance (α=-1)
            - Kullback-Leibler (α = 1)
            - dual Kullback-Leibler (α = 0)

        Reference for beta divergence:
            A.Cichocki, S.Amari, R.Zdunek, R.Kompass, G.Hori, and Z.He,
            "Extended SMART algorithms for non-negative matrix factorization,"
            Artificial Intelligence and Soft Computing, 2006.

            C.Fevotte, J.Idier,
            "Algorithms for nonnegative matrix factorization with the beta-divergence",
            Neural Computation, 2011.

                min D(V||W*H)

                where d(x|y) = | (x^β + (β-1)y^β - βxy^(β-1)) ÷ (β(β-1)) (β ∉ {0 1})
                               |
                               | xlog(x÷y) - x + y                       (β = 1)
                               |
                               | x÷y - log(x÷y) - x + y                  (β = 0)

    + mu_mod: Modified multiplicative upates (MU)
        Reference:
            C.-J. Lin,
            "On the convergence of multiplicative update algorithms for nonnegative matrix factorization,"
            IEEE Trans. Neural Netw. vol.18, no.6, pp.1589?1596, 2007. 
    
    + mu_acc: Accelerated multiplicative updates (Accelerated MU)
        Reference:
            N. Gillis and F. Glineur, 
            "Accelerated Multiplicative Updates and hierarchical ALS Algorithms for Nonnegative 
            Matrix Factorization,", 
            Neural Computation 24 (4), pp. 1085-1105, 2012. 
            See http://sites.google.com/site/nicolasgillis/.
            The corresponding code is originally created by the authors, 
            Then, it is modifided by H.Kasai.
"""
mutable struct MUParams <: AbstractMUParams
    α::Real
    β::Real
    ϵ::Real
    δ::Real
    σ::Real
    scale_W_iter::Bool
    scale_W_last::Bool
    div::Symbol
    alg::Symbol

    function MUParams(;
        α::Real            = 1,
        β::Real            = 2,
        ϵ::Real            = 1e-16,
        δ::Real            = 1e-1,
        σ::Real            = 1e-10,
        scale_W_iter::Bool = true,
        scale_W_last::Bool = true,
        div::Symbol        = :β,
        alg::Symbol        = :mu)

        valid_div = [:α, :β]
        valid_alg = [:mu, :mu_mod]

        ϵ > 0 || throw(ArgumentError("ϵ must be > 0 "))
        div in valid_div || throw(ArgumentError(string("div must be one of ",  valid_div)))
        alg in valid_alg || throw(ArgumentError(string("alg must be one of ",  valid_alg)))

        new(α, β, ϵ, δ, σ, scale_W_iter, scale_W_last, div, alg)
    end
end

"""
    _mu_cost(V, W, H, local_params)

Cost function minimized by MU NMF algorithms
"""
function _mu_cost(V::Matrix{T}, W::Matrix{T}, H::Matrix{T}, local_params::AbstractMUParams) where T <: Real
    if local_params.div == :β
        β_divergence(V, W * H, local_params.β)
    elseif local_params.div == :α
        α_divergence(V, W * H, local_params.α)
    end
end

function _update_nmf_mu_mod!(V::Matrix{T}, V_a::Matrix{T}, W::Matrix{T}, H::Matrix{T}, params::AbstractMUParams) where T <: Real
    ##### this update is limited to euclidean cost function
    
    # update H
    WtW = transpose(W) * W
    WtV = transpose(W) * V
    gradH = WtW * H - WtV
    Hb = max.(H, (gradH .< 0) .* params.σ)
    H .= H - Hb ./ (WtW * Hb .+ params.δ) .* gradH
    
    # update W
    HHt = H * transpose(H)
    VHt = V * transpose(H)
    gradW = W * HHt - VHt
    Wb = max.(W, (gradW .< 0) .* params.σ)
    W .= W - Wb ./ (Wb * HHt .+ params.δ) .* gradW
end


function _update_nmf_mu_α!(V::Matrix{T}, V_a::Matrix{T}, W::Matrix{T}, H::Matrix{T}, params::AbstractMUParams) where T <: Real
    if params.α == 0
        #### dual Kullback-Leibler

        #### update H
        #### NOTE: if W has columns normalized to sum to 1, the normalization matrix
        #### (transpose(W) * ones(size(V))) .^ (-1) is just ones(K, N)
        H .= H .* exp.((transpose(W) * ones(size(V))) .^ (-1)  .* (transpose(W) * log.(V ./ (V_a .+ params.ϵ)))) 
        V_a .= W * H

        #### update W
        #### NOTE: if H has columns normalized to sum to 1, the normalization matrix
        #### (transpose(H) * ones(size(V))) .^ (-1) is just ones(K, N)
        W .= W .* exp.((ones(size(V)) * transpose(H)) .^ (-1) .* (log.(V ./ (V_a .+ params.ϵ)) * transpose(H)))
        V_a .= W * H

    else
        #### general α divergence updates for α != 0
        
        #### A.Cichocki, S.Amari, R.Zdunek, R.Kompass, G.Hori, and Z.He,
        #### "Extended SMART algorithms for non-negative matrix factorization,"
        #### Artificial Intelligence and Soft Computing, 2006.
        #### Equation (44)
        
        #### note: it is not clear in eqs (45) and (46) how the normalization
        #### made the denominators disappear in (44).
        
        # #### case of eqs (45), (46)
        # 
        # # update W
        # W .= W .* (((V .+ params.ϵ) ./ (V_a .+ params.ϵ)) .^ params.α * transpose(H)) .^ (1/params.α)

        # #### normalize to unit sums the columns of W if specified
        # #### note: order not as in (45-46)
        # cs_W = sum(W, dims=1)
        # W .= W .* repeat(cs_W .^ -1, size(W, 1), 1)

        # V_a .= W * H

        # # update H
        # H .= H .* ((transpose(W) * ((V .+ params.ϵ)./(V_a .+ params.ϵ)) .^ params.α)) .^ (1/params.α)
        # V_a .= W * H

        #### eqs (44)
        
        # update W
        W .= W .* ((((V .+ params.ϵ) ./ (V_a .+ params.ϵ)) .^ params.α * transpose(H)) ./ (ones(size(V)) * transpose(H)) .+ params.ϵ) .^ (1/params.α) 
        V_a .= W * H

        # update H
        H .= H .* (((transpose(W) * ((V .+ params.ϵ) ./ (V_a .+ params.ϵ)) .^ params.α)) ./ (transpose(W) * ones(size(V))) .+ params.ϵ) .^ (1/params.α)
        V_a .= W * H

    end
end




function _update_nmf_mu!(V::Matrix{T}, V_a::Matrix{T}, W::Matrix{T}, H::Matrix{T}, params::AbstractMUParams) where T <: Real

    if params.alg == :mu
        if params.div == :α
            if params.α == 0
                #### dual Kullback-Leibler

                #### update H
                #### NOTE: if W has columns normalized to sum to 1, the normalization matrix
                #### (transpose(W) * ones(size(V))) .^ (-1) is just ones(K, N)
                H .= H .* exp.((transpose(W) * ones(size(V))) .^ (-1)  .* (transpose(W) * log.(V ./ (V_a .+ params.ϵ)))) 
                V_a .= W * H

                #### update W
                #### NOTE: if H has columns normalized to sum to 1, the normalization matrix
                #### (transpose(H) * ones(size(V))) .^ (-1) is just ones(K, N)
                W .= W .* exp.((ones(size(V)) * transpose(H)) .^ (-1) .* (log.(V ./ (V_a .+ params.ϵ)) * transpose(H)))
                V_a .= W * H

            else
                #### general α divergence updates for α != 0
                
                #### A.Cichocki, S.Amari, R.Zdunek, R.Kompass, G.Hori, and Z.He,
                #### "Extended SMART algorithms for non-negative matrix factorization,"
                #### Artificial Intelligence and Soft Computing, 2006.
                #### Equation (44)
                
                #### note: it is not clear in eqs (45) and (46) how the normalization
                #### made the denominators disappear in (44).
                
                # #### case of eqs (45), (46)
                # 
                # # update W
                # W .= W .* (((V .+ params.ϵ) ./ (V_a .+ params.ϵ)) .^ params.α * transpose(H)) .^ (1/params.α)

                # #### normalize to unit sums the columns of W if specified
                # #### note: order not as in (45-46)
                # cs_W = sum(W, dims=1)
                # W .= W .* repeat(cs_W .^ -1, size(W, 1), 1)

                # V_a .= W * H

                # # update H
                # H .= H .* ((transpose(W) * ((V .+ params.ϵ)./(V_a .+ params.ϵ)) .^ params.α)) .^ (1/params.α)
                # V_a .= W * H

                #### eqs (44)
                
                # update W
                W .= W .* ((((V .+ params.ϵ) ./ (V_a .+ params.ϵ)) .^ params.α * transpose(H)) ./ (ones(size(V)) * transpose(H)) .+ params.ϵ) .^ (1/params.α) 
                V_a .= W * H

                # update H
                H .= H .* (((transpose(W) * ((V .+ params.ϵ) ./ (V_a .+ params.ϵ)) .^ params.α)) ./ (transpose(W) * ones(size(V))) .+ params.ϵ) .^ (1/params.α)
                V_a .= W * H

            end
        elseif params.div == :β
            if params.β == 2
                #### update rules for β = 2 can be written more efficiently

                # update H
                H .= H .* ((transpose(W) * V) ./ ((transpose(W) * W) * H))

                # update W
                W .= W .* ((V * transpose(H)) ./ (W * (H * transpose(H))))

                # update product
                V_a .= W * H
            else
                #### general β divergence updates for β != 2
                
                # update W
                W .= W .* (((V .* V_a .^ (params.β-2)) * transpose(H)) ./ max.(V_a .^ (params.β-1) * transpose(H), params.ϵ))
                V_a .= W * H

                # update H
                H .= H .* ((transpose(W) * (V .* V_a .^ (params.β-2))) ./ max.(transpose(W) * V_a .^ (params.β-1), params.ϵ))
                V_a .= W * H
            end
        end

    elseif params.alg == :mu_mod
        ##### this update is limited to euclidean cost function
        
        # update H
        WtW = transpose(W) * W
        WtV = transpose(W) * V
        gradH = WtW * H - WtV
        Hb = max.(H, (gradH .< 0) .* params.σ)
        H .= H - Hb ./ (WtW * Hb .+ params.δ) .* gradH
        
        # update W
        HHt = H * transpose(H)
        VHt = V * transpose(H)
        gradW = W * HHt - VHt
        Wb = max.(W, (gradW .< 0) .* params.σ)
        W .= W - Wb ./ (Wb * HHt .+ params.δ) .* gradW
    end

    #### replace values below given threshold
    W .= max.(W, 1e6 * params.ϵ)
    H .= max.(H, 1e6 * params.ϵ)

    if params.scale_W_iter
        scale_col_W!(W, H, 1)
    end
end

"""
    _nmf_MU(V, W_init, H_init, global_params, local_params)

Helper function for `nmf_MU`. Runs the Multiplicative Updates with given intial matrices W and H.
"""
function _nmf_MU(V::Matrix{T}, W_init::Matrix{T}, H_init::Matrix{T}, global_params::AbstractNMFParams, local_params::AbstractNMFParams) where T <: Real
    W    = W_init[:,:]
    H    = H_init[:,:]
    V_a  = W * H

    #### metrics and stopping_crit
    func_cost          = (V, W, H) -> _mu_cost(V, W, H, local_params)

    if local_params.div == :α
        func_divg          = (V, W, H) -> α_divergence(V, W * H, local_params.α)
    elseif local_params.div == :β
        func_divg          = (V, W, H) -> β_divergence(V, W * H, local_params.β)
    end
    
    metrics           = NMFMetrics(V, W_init, H_init, func_cost, func_divg)
    stopping_crit_met = false
    iteration_is_sane = true

    #### scale W before first iter if specified
    if local_params.scale_W_iter
        scale_col_W!(W, H, 1)
    end

    while !stopping_crit_met && iteration_is_sane
        #### update V_a, W, H
        _update_nmf_mu!(V, V_a, W, H, local_params)

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
    nmf_MU(V, global_params, local_params)

Multiplicative upates (MU) for non-negative matrix factorization (NMF).
The problem of interest is defined as
        min D(V || W * H),
        where 
        {V, W, H} >= 0.
Given a non-negative matrix V, factorized non-negative matrices W, H are calculated.

# Arguments
- `V::Matrix{T}`           : (F x N) non-negative matrix to factorize
- `global_params::Params`  : see [`Params`](@ref)
- `local_params::MUParams` : see [`MUParams`](@ref)

# References
Code largely borrowed from H. Kasai matlab NMFLibrary solver/base/nmf_mu.m at https://github.com/hiroyuki-kasai/NMFLibrary.
"""
function nmf_MU(V::Matrix{T}, global_params::AbstractNMFParams, local_params::AbstractMUParams) where T <: Real
    # initalization
    W_init, H_init = initialize_nmf(
        V, 
        rank = global_params.rank,
        init = global_params.init,
        dist = global_params.dist,
        seed = global_params.seed
    )

    # algorithm
    _nmf_MU(V, W_init, H_init, global_params, local_params)
end
