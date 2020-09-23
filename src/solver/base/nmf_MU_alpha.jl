"""
Created on Wed Sep 23 2020

@author: Yoann Pradat

Multiplicative update algorithms for α divergence.
"""

"""
    _update_nmf_mu_α!(V, WH, W, H, α, ϵ)
    
"""
function _update_nmf_mu_α!(V::Matrix{T}, WH::Matrix{T}, W::Matrix{T}, H::Matrix{T}, α::Real, ϵ::Real) where T <: Real
    if α == 0
        #### dual Kullback-Leibler

        #### update H
        #### NOTE: if W has columns normalized to sum to 1, the normalization matrix
        #### (transpose(W) * ones(size(V))) .^ (-1) is just ones(K, N)
        H .= H .* exp.((transpose(W) * ones(size(V))) .^ (-1)  .* (transpose(W) * log.(V ./ (WH .+ ϵ)))) 
        WH .= W * H

        #### update W
        #### NOTE: if H has columns normalized to sum to 1, the normalization matrix
        #### (transpose(H) * ones(size(V))) .^ (-1) is just ones(K, N)
        W .= W .* exp.((ones(size(V)) * transpose(H)) .^ (-1) .* (log.(V ./ (WH .+ ϵ)) * transpose(H)))
        WH .= W * H

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
        # W .= W .* (((V .+ ϵ) ./ (WH .+ ϵ)) .^ α * transpose(H)) .^ (1/α)

        # #### normalize to unit sums the columns of W if specified
        # #### note: order not as in (45-46)
        # cs_W = sum(W, dims=1)
        # W .= W .* repeat(cs_W .^ -1, size(W, 1), 1)

        # WH .= W * H

        # # update H
        # H .= H .* ((transpose(W) * ((V .+ ϵ)./(WH .+ ϵ)) .^ α)) .^ (1/α)
        # WH .= W * H

        #### eqs (44)
        
        # update W
        W .= W .* ((((V .+ ϵ) ./ (WH .+ ϵ)) .^ α * transpose(H)) ./ (ones(size(V)) * transpose(H)) .+ ϵ) .^ (1/α) 
        WH .= W * H

        # update H
        H .= H .* (((transpose(W) * ((V .+ ϵ) ./ (WH .+ ϵ)) .^ α)) ./ (transpose(W) * ones(size(V))) .+ ϵ) .^ (1/α)
        WH .= W * H

    end
end
