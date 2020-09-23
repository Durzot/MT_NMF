"""
Created on Thu Apr 09 2020

@author: Yoann Pradat

α and β divergences.
"""

"""
β_divergence(X, Y, β[, ϵ])

Compute the the β-divergence of X and Y, d_β(X||Y).

# Arguments
- `X::Matrix{<:Real}`: array 
- `Y::Matrix{<:Real}`: array
- `β::Real`: real, parameter of the β-divergence.
    If β == 2, this is half the Frobenius *squared* norm.
    If β == 1, this is the Kullback-Leibler divergence.
    If β == 0, this is the Itakura-Saito divergence.
    Else, this is the general β-divergence.
"""
function β_divergence(X::Matrix{T}, Y::Matrix{T}, β::Real; ϵ::Float64=eps(T)) where T <: Real
    if β == 2
        #### Frobenius norm
        res = norm(X - Y, 2) ^ 2 / 2.
    else
        # vec() flattens the array column-wise
        Y_vec = vec(Y)
        X_vec = vec(X)

        # 0 coordinates in X contribute 0 to the divergence.
        # drop these coordinates
        indices = X_vec .> ϵ 
        Y_vec = Y_vec[indices]
        X_vec = X_vec[indices]

        # avoid division by zero
        Y_vec[Y_vec .== 0] .= ϵ

        if β == 1
            #### Kullback-Leibler divergence
            sum_X_log_X_Y = sum(X_vec .* log.(X_vec ./ Y_vec))
            sum_X         = sum(X_vec)
            sum_Y         = sum(Y_vec)
            res           = abs(sum_X_log_X_Y - sum_X + sum_Y)

        elseif β == 0
            #### Itakura-Saito divergence
            X_Y         = X_vec ./ Y_vec
            sum_log_X_Y = sum(log.(X_Y))
            sum_X_Y_m1  = sum(X_Y .- 1)
            res         = abs(-sum_log_X_Y + sum_X_Y_m1)

        else
            #### General β-divergence for β not in (0, 1, 2)
            sum_X_β      = sum(X_vec .^ β)
            sum_Y_β      = sum(Y_vec .^ β)
            sum_X_Y_β_m1 = sum(X_vec .* Y_vec .^ (β-1))
            res          = abs((sum_X_β + (β-1) * sum_Y_β - β * sum_X_Y_β_m1)/(β*(β-1)))
        end
    end
    res
end

"""
α_divergence(X, Y, α[, ϵ])

Compute the the α-divergence of X and Y, d_α(X||Y).

# Arguments
- `X::Matrix{<:Real}`: array
- `Y::Matrix{<:Real}`: array
- `α::Real`: float, parameter of the α-divergence.
    If α == 2, this is Pearsoni's Chi-squared distance
    If α == 1, this is (at the limit) the Kullback-Leibler divergence
    If α == 0, this is (at the limit) the dual Kullback-Leibler divergence
    If α == 0.5, this is the squared Hellinger distance
    If α == -1, this is the Neyman's Chi-squared distance
    Else, this is the general α-divergence.
"""
function α_divergence(X::Matrix{T}, Y::Matrix{T}, α::Real; ϵ::Float64=eps(T)) where T <: Real
    # vec() flattens the array column-wise
    Y_vec = vec(Y)
    X_vec = vec(X)

    if α == 1
        # avoid division by zero
        Y_vec[Y_vec .== 0] .= ϵ

        #### Kullback-Leibler divergence
        sum_X_log_X_Y = sum(X_vec .* log.(X_vec ./ Y_vec))
        sum_X         = sum(X_vec)
        sum_Y         = sum(Y_vec)
        res           = abs(sum_X_log_X_Y - sum_X + sum_Y)
    elseif α == 0
        # avoid division by zero
        X_vec[X_vec .== 0] .= ϵ

        #### dual Kullback-Leibler divergence
        sum_Y_log_Y_X = sum(Y_vec .* log.(Y_vec ./ X_vec))
        sum_Y         = sum(Y_vec)
        sum_X         = sum(X_vec)
        res           = abs(sum_Y_log_Y_X - sum_Y + sum_X)
    else
        if α < 1
            # avoid division by zero
            Y_vec[Y_vec .== 0] .= ϵ
        end

        if α < 0
            # avoid division by zero
            X_vec[X_vec .== 0] .= ϵ
        end

        #### general α-divergence for α not in {0,1}
        sum_X_α_Y_α_m1 = sum(X_vec .^ α .* Y_vec .^ (1-α))
        sum_X          = sum(X_vec)
        sum_Y          = sum(Y_vec)
        res            = abs((sum_X_α_Y_α_m1 - α * sum_X + (α-1) * sum_Y)/(α*(α-1)))
    end
    res
end
