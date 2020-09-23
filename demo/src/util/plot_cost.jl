"""
Created on Mon May 22 2020

@author: Yoann Pradat

Function for plotting cost over iterations.
"""

using LaTeXStrings
using Plots

function plot_cost(res::Array{NMFResults{T},1}, lab::Array{String,1}, ls::Array{Symbol,1}, col::Array{String,1}, fn::String; cst::Float64=1e-12, relative::Bool=false) where T <: Real
    n_res = length(res)
    n_lab = length(lab)
    n_col = length(col)
    n_ls  = length(ls)

    n_res == n_lab || throw(ArgumentError("array lab must have same size as array res"))
    n_res == n_col || throw(ArgumentError("array col must have same size as array res"))
    n_res == n_ls || throw(ArgumentError("array ls must have same size as array res"))

    p_obj = plot(dpi=100, size=(1000, 400))

    for n = 1:n_res
        #### plot cost for each iter
        if relative
            cost = res[n].metrics.absolute_cost ./ res[n].metrics.absolute_cost[1] .+ cst
        else
            cost = res[n].metrics.absolute_cost .+ cst
        end

        plot!(
            p_obj,
            cost,
            label = lab[n],
            ls    = ls[n],
            lw    = 1,
            color = col[n],
        )
    end

    #### plot settings
    plot!(
        p_obj,
        xlabel = "",
        xscale = :log,
        ylabel = "",
        yscale = :log,
    )

    #### save plot
    savefig(p_obj, fn)
    println("plot saved at $fn")
end
