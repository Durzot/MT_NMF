"""
Created on Wed 27 May 2020

@author: Yoann Pradat

Plot metrics as produced by the `rank_by_consensus` function.
"""

using  DataFrames
import PyPlot
const  plt = PyPlot

function plot_metric!(ax::PyCall.PyObject, df_metrics::DataFrame; axvline::Bool, variable::Symbol, color::String, label::String, legend::String, title::String)
    #### plot
    ax.plot(
        df_metrics.rank,
        df_metrics[:,variable],
        color  = color,
        marker = "o",
        label  = legend
    )

    ax.axvline(
        df_metrics[argmax(df_metrics[:,variable]),:rank],
        ymin  = 0,
        ymax  = 1,
        color = color,
        ls    = "--",
        lw    = 2
    )

    ax.text(
        x          = -0.1,
        y          = 1.1,
        s          = label,
        transform  = ax.transAxes,
        fontsize   = 15,
        fontweight = "bold"
    )

    #### axes
    ax.set_xlabel(
        xlabel     = "rank",
        fontsize   = 15,
        fontweight = "bold"
    )
    ax.set_ylabel(
        ylabel     = "$(variable)", 
        fontsize   = 15,
        fontweight = "bold"
    )
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)

    #### title
    ax.set_title(
        label=title,
        fontsize=18,
        fontweight="bold"
    )
end


"""
    plot_metrics_consensus(df_metrics, filepath)

Plot metrics as produced by the `rank_by_consensus` function.
"""
function plot_metrics_consensus(df_metrics::DataFrame, filepath::String)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,10))

    plot_metric!(
        ax[1,1],
        df_metrics,
        axvline     = true,
        variable    = :cophenetic,
        color       = "purple",
        label       = "A",
        legend      = "",
        title       = "Cophenetic correlation"
    )

    plot_metric!(
        ax[1,2],
        df_metrics,
        axvline     = true,
        variable    = :dispersion,
        color       = "purple",
        label       = "B",
        legend      = "",
        title       = "Dispersion"
    )

    plot_metric!(
        ax[2,1],
        df_metrics,
        axvline     = true,
        variable    = :silhouette,
        color       = "purple",
        label       = "C",
        legend      = "",
        title       = "Silhouette"
    )

    plot_metric!(
        ax[2,2],
        df_metrics,
        axvline     = true,
        variable    = :sparse_W,
        color       = "green",
        label       = "D",
        legend      = "basis",
        title       = "Sparseness"
    )

    plot_metric!(
        ax[2,2],
        df_metrics,
        axvline     = true,
        variable    = :sparse_H,
        color       = "red",
        label       = "D",
        legend      = "coefs",
        title       = "Sparseness"
    )

    ax[2,2].set_ylabel("sparsity index")
    ax[2,2].legend(loc="best", fontsize=15, frameon=false)

    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.savefig(filepath, bbox_inches="tight")
end

"""
    plot_metrics_stability(df_metrics, filepath)

Plot metrics as produced by the `rank_by_stability` function.
"""
function plot_metrics_stability(df_metrics::DataFrame, filepath::String)
    fig, ax1 = plt.subplots()

    variable1 = :stability
    color     = "red"

    p1 = ax1.plot(
        df_metrics.rank,
        df_metrics[:,variable1],
        color      = color,
        marker     = "o",
        markersize = 10,
        linestyle  = ":",
    )

    #### axes
    ax1.set_xlabel(
        xlabel     = "rank",
        fontsize   = 12,
        fontweight = "bold"
    )
    ax1.set_ylabel(
        ylabel     = "", 
        fontsize   = 15,
        fontweight = "bold"
    )

    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xticks(df_metrics.rank)

    #### new ax with twin x axis
    ax2 = ax1.twinx()

    variable2 = :fidelity
    color     = "dodgerblue"

    p2 = ax2.plot(
        df_metrics.rank,
        df_metrics[:,variable2],
        color      = color,
        marker     = "o",
        markersize = 10,
        linestyle  = ":",
    )

    #### axes
    ax2.set_xlabel(
        xlabel     = "rank",
        fontsize   = 12,
        fontweight = "bold"
    )
    ax2.set_ylabel(
        ylabel     = "", 
        fontsize   = 15,
        fontweight = "bold"
    )

    ax2.tick_params(axis="y", labelcolor=color)

    #### legend + title
    plt.legend([p1[1], p2[1]], ["$(variable1)", "$(variable2)"], loc="best", fontsize=12)

    plt.title(
        label="Rank by stability",
        fontsize=15,
        fontweight="bold"
    )

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(filepath, bbox_inches="tight")
end
