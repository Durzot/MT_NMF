"""
Created on Thu Apr 10 2020

@author: Yoann Pradat

Functions for initializing W and H matrices in the NMF V = WH.
"""

"""
    initialize_nmf(V[, rank, init, dist, ϵ, seed])

Compute initial guess for the non-negative rank K approximation of V: V = WH.
# Arguments
- `V::Matrix{<:Real}`   : matrix, shape (n_features, n_samples)
- `rank::Integer`       : integer, number of components desired
- `init::Symbol`        : symbol,  Method used to initialize the procedure. Valid options are:
    + :default: :nndsvd if rank <= min(n_samples, n_features), otherwise :random.
    + :random: Nonnegative random matrices, scaled with: sqrt(mean(V) / rank)
    + :nndsvd: Nonnegative Double Singular Value Decomposition (NNDSVD) initialization 
    (better for sparseness)
    + :nndsvda: NNDSVD with zeros filled with the average of V (better when sparsity is 
    not desired)
    + :nndsvdar: NNDSVD with zeros filled with small random values (generally faster, less 
    accurate alternative to NNDSVDa for when sparsity is not desired)
- `dist::Distribution` : the distribution from which random entries are drawn. Be aware 
    that entries will be multiplied by sqrt(mean(V) / rank). Used for intializations with randomness
- `seed:Integer`       : fix the seed of RNG
- `ϵ::<:Real`          : threshold below which values are assigned to 0

# References
- C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for nonnegative 
matrix factorization - Pattern Recognition, 2007
"""
function initialize_nmf(V::Matrix{T}; rank::Integer=2, init::Symbol=:default, dist::Distribution=truncated(Normal(0,1), 0, Inf), ϵ::Real=1e-6, seed::Integer=0) where T <: Real
    rng = Random.seed!(seed)

    if init == :default
        n_features, n_samples = size(V)
        if rank <= min(n_features, n_samples)
            init = :nndsvd
        else
            init = :random
        end
    end

    if init == :random
        W, H = _initialize_random(V, rank=rank, dist=dist, rng=rng)
    elseif init == :nndsvd
        W, H = _initialize_nndsvd(V, rank=rank)
        W[W .< ϵ] .= 0
        H[H .< ϵ] .= 0
    elseif init == :nndsvda
        W, H = _initialize_nndsvd(V, rank=rank)
        W[W .< ϵ] .= 0
        H[H .< ϵ] .= 0

        avg_V = mean(V)
        W[W .== 0] .= avg_V
        H[H .== 0] .= avg_V
    elseif init == :nndsvdar
        W, H = _initialize_nndsvd(V, rank=rank)
        W[W .< ϵ] .= 0
        H[H .< ϵ] .= 0

        avg_V = mean(V)
        W[W .== 0] .= avg_V * rand(rng, dist, size(W[W .== 0], 1)) ./ 100
        H[H .== 0] .= avg_V * rand(rng, dist, size(H[H .== 0], 1)) ./ 100

        if minimum(W) < 0 || minimum(H) < 0
            throw(ArgumentError("Please specify a distribution that does not produce negative values. Use `truncated()` if necessary"))
        end
    else
        throw(ArgumentError("""Invalid init parameter: choose one of :default, :random, :nndsvd, :nndsvda, :nndsvdar """))
    end
    W, H
end

"""
    _initialize_random(V[, rank, dist, rng])

Initialize W and H by random matrices of size (n_features, rank) and (rank, n_samples).
Entries are drawn from half-normal distribution i.e the absolute value of a normally distributed random
variable. The variance  is chosen to be the mean(V) / rank.
"""
function _initialize_random(V::Matrix{T}; rank, dist::Distribution, rng::AbstractRNG) where T <: Real
    n_features, n_samples = size(V)
    avg = sqrt(mean(V) / rank)

    W = avg * rand(rng, dist, n_features, rank)
    H = avg * rand(rng, dist, rank, n_samples)

    if minimum(W) < 0 || minimum(H) < 0
        throw(ArgumentError("Please specify a distribution that does not produce negative values. Use `truncated()` if necessary"))
    end

    W,H
end

"""
    _initialize_nndsvd(X; rank)

Initialize matrices of the decomposition X = WH using SVD.

# References
- C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
nonnegative matrix factorization - Pattern Recognition, 2007
"""
function _initialize_nndsvd(X::Matrix{T}; rank::Integer=2) where T <: Real
    n_features, n_samples = size(X)
    W = zeros(T, n_features, rank)
    H = zeros(T, rank, n_samples)
    
    #### Using Perron-Frobenius theorem, one can prove that the leading singular 
    #### vectors in SVD can be chosen to have non negative coordinates.
    #### These can therefore be used for initialization.
    U, S, Vt = svd(X, full=false)
    V = transpose(Vt)

    W[:, 1] = sqrt(S[1]) * abs.(U[:, 1])
    H[1, :] = sqrt(S[1]) * abs.(V[1, :])

    for j = 2:rank
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = max.(x, 0), max.(y, 0)
        x_n, y_n = abs.(min.(x, 0)), abs.(min.(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p, 2), norm(y_p, 2)
        x_n_nrm, y_n_nrm = norm(x_n, 2), norm(y_n, 2)

        # multiply positive and negative norms
        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            σ = m_p
        else
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            σ = m_n
        end

        λ = sqrt(S[j] * σ)
        W[:, j] = λ * u
        H[j, :] = λ * v
    end

    W, H
end
