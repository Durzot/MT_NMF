"""
Created on Fri May 22 2020

@author: Yoann Pradat

Functions for scaling matrices in NMF iterations.
"""

function scale_col_W!(W, H, p=1)
    #### col norm
    norms_W = mapslices(x -> norm(x, p), W, dims=1)

    #### one or more columns of W may be null vectors
    indices = (1:size(W,2))[vec(norms_W .> 0)]

    W[:, indices] .= W[:, indices] .* repeat(norms_W[:, indices] .^ -1, size(W, 1), 1)
    H[indices, :] .= H[indices, :] .* repeat(transpose(norms_W[:, indices]), 1, size(H, 2))
    
    nothing
end

function scale_row_H!(W, H, p=1)
    #### col norm
    norms_H = mapslices(x -> norm(x, p), H, dims=2)

    #### one or more columns of W may be null vectors
    indices = (1:size(H,1))[vec(norms_H .> 0)]

    W[:, indices] .= W[:, indices] .* repeat(transpose(norms_H[indices, :]), size(W, 1), 1)
    H[indices, :] .= H[indices, :] .* repeat(norms_H[indices, :] .^ -1, 1, size(H, 2))
    
    nothing
end
