"""
Created on Tue May 06 2020

@author: Yoann Pradat

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Functions for plotting input and output of NMF factorizations and related heuristics.
"""

import matplotlib.cm as cm
from   matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as  np
import os
import pandas as pd
import seaborn as sns
from   sklearn.metrics.pairwise import cosine_similarity
import re

def plot_col_bar(ax, heights, ylabel="col sum", yticklabels_format=".2f", y_lim_quantile=1.0):
    ax.bar(
        range(len(heights)),
        heights,
        color  = "dodgerblue",
        align  = "edge",
        width  = 0.98
    )

    ax.set_xlim([-0.01*len(heights), len(heights)])
    ax.set_ylim([0, np.quantile(heights, y_lim_quantile)])

    #### esthetics
    ax.set_xlabel(
        xlabel     = "",
    )
    ax.set_ylabel(
        ylabel     = ylabel,
        fontsize   = 12,
        fontweight = "medium",
        ha         = "center",
        va         = "top"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(axis="x", which="both", length=0, labelsize=0, pad=0)
    ax.tick_params(axis="y", which="both", length=8)
    ax.set_xticks([])
    ax.set_yticks([0, np.quantile(heights, y_lim_quantile)])
    ax.set_yticklabels([0, ("{:%s}" % yticklabels_format).format(np.quantile(heights, y_lim_quantile))], va="bottom")


def plot_col_err(ax, means, errors, ylabel="", yticklabels_fmt=".3g", fmt='o', ms=20, mfc="red", mec="black", mew=1, ecolor="lightsalmon", elw=3, els='-', ymin=None, ymax=None):
    eb = ax.errorbar(
        x          = range(len(means)),
        y          = means,
        yerr       = errors,
        fmt        = fmt,
        mfc        = mfc,
        mec        = mec,
        ms         = ms,
        mew        = mew,
        ecolor     = ecolor,
        elinewidth = elw,
        capsize    = 0
    )

    eb[-1][0].set_linestyle(els)

    #### esthetics
    ax.set_xlabel(
        xlabel     = "",
    )
    ax.set_ylabel(
        ylabel     = ylabel,
        fontsize   = 12,
        fontweight = "medium",
        ha         = "center",
        va         = "bottom",
        labelpad   = 20
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(axis="x", which="both", length=0, labelsize=0, pad=0)
    ax.tick_params(axis="y", which="both", length=8, pad=-8)
    ax.set_xticks([])

    if ymin is None:
        ymin = min(np.array(means)-np.array(errors))
    if ymax is None:
        ymax = max(np.array(means)+np.array(errors))

    ax.set_yticks([ymin, ymax])
    ax.set_yticklabels([("{:%s}" % yticklabels_fmt).format(x) for x in [ymin, ymax]], ha="left", va="bottom")

def plot_col_color_bar(ax, colors):
    ax.bar(
        range(len(colors)),
        1,
        color  = colors,
        align  = "edge",
        width  = 1
    )

    ax.set_xlim([-0.01*len(colors), len(colors)])
    ax.set_ylim([0, 1])

    #### esthetics
    ax.set_xlabel(
        xlabel     = "",
    )
    ax.set_ylabel(
        ylabel     = "",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="x", which="both", length=0, labelsize=0, pad=0)
    ax.tick_params(axis="y", which="both", length=0, labelsize=0, pad=0)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_row_bar(ax, widths, xlabel, colors):
    ax.barh(
        range(len(widths)),
        widths[::-1],
        color  = colors[::-1],
        align  = "edge",
        height  = 1
    )

    #### esthetics
    ax.set_xlabel(
        xlabel     = xlabel,
    )
    ax.set_ylabel(
        ylabel     = "",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="x", which="both", length=6, labelsize=8, pad=2)
    ax.set_xticks([0, max(widths)])
    ax.set_xticklabels(labels=["", "%.1g" % max(widths)], ha="center", rotation=0)

    ax.tick_params(axis="y", which="both", length=0, labelsize=0, pad=0)
    ax.set_yticks([])

    ax.set_xlim([0, max(widths)])
    ax.set_ylim([0, len(widths)])


def plot_row_color_bar(ax, colors, row_labels, labelsize=5):
    if row_labels is None:
        row_labels = []

    ax.barh(
        range(len(colors)),
        1,
        color  = colors[::-1],
        align  = "edge",
        height  = 1
    )

    ax.set_xlim([-0.5, 1])
    ax.set_ylim([0, len(colors)])

    #### esthetics
    ax.set_xlabel(
        xlabel     = "",
    )
    ax.set_ylabel(
        ylabel     = "",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="x", which="both", length=0, labelsize=0, pad=0)
    ax.set_xticks([])
    ax.tick_params(axis="y", which="both", length=0, labelsize=labelsize, pad=0)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(labels=row_labels[::-1], va="bottom")

def plot_hmap(ax, cbar_ax, X, linewidths=0, linecolor="white", cmap=None, boundaries=None, cbar=True, cbar_kws=None, cbar_title="", xticklabels=False, yticklabels=False, labelsize=4):

    if boundaries is None:
        #### 0 and 6 evenly space quantiles
        boundaries = [0] + [np.round(np.quantile(X, i/6), decimals=3) for i in range(1, 7)]

    if cmap is None:
        #### color map from blue to blue/red
        cmap = sns.diverging_palette(250, 20, s=70, l=50, sep=1, as_cmap=True)

    if cbar_kws is None:
        cbar_kws    = {
            "spacing"     : "uniform",
            "ticks"       : boundaries[1  : -1],
            "format"      : "%g",
            "fraction"    : 0.03,
            "orientation" : "horizontal",
            "aspect"      : 50
        }

    sns.heatmap(
        X,
        vmin        = min(boundaries),
        vmax        = max(boundaries),
        norm        = cm.colors.BoundaryNorm(boundaries=boundaries, ncolors=256),
        linecolor   = linecolor,
        linewidths  = linewidths,
        cmap        = cmap,
        square      = False,
        xticklabels = xticklabels,
        yticklabels = yticklabels,
        ax          = ax,
        cbar_ax     = cbar_ax,
        cbar        = cbar,
        cbar_kws    = cbar_kws,
    )

    ax.set_xlim([-0.01*X.shape[1], X.shape[1]])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", which="both", pad=0, labelsize=labelsize, length=0)

    #### colorbar
    if cbar:
        if cbar_kws["orientation"] == "horizontal":
            cbar_ax.set_title(cbar_title, fontsize=12)
            cbar_ax.set_xlabel("")
            cbar_ax.set_ylabel("")
        else:
            cbar_ax.set_title("")
            cbar_ax.set_xlabel("")
            cbar_ax.set_ylabel(cbar_title, fontsize=12)

####
####
#### 1. PLOT INPUT MATRIX V
####
####

def plot_V(V, row_groups=None, row_groups_colors=None, col_groups=None, col_groups_colors=None, row_labels=[], hmap_boundaries=None, hmap_cmap=None, row_labelsize=5, figsize=(24,10)):
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(35, 60, wspace=0, hspace=0.2)

    #### upper plots
    if col_groups is None:
        ax1 = fig.add_subplot(gs[:5, 1:60])
    else:
        ax1 = fig.add_subplot(gs[:4, 1:60])
        ax2 = fig.add_subplot(gs[4:5,1:60])
        ax2_lgd = fig.add_subplot(gs[31:33, 45:60])

        #### plot columns color bar
        colors = [col_groups_colors[x] for x in col_groups]
        plot_col_color_bar(ax2, colors)

        #### plot columns color bar legend
        handles = [Line2D([0],[0], color=v, lw=4, label="%s" % k) for k,v in col_groups_colors.items()]
        handles.sort(key = lambda handle: handle.get_label())

        ax2_lgd.legend(
            handles        = handles,
            loc            = "center",
            frameon        = False,
            ncol           = 3,
            title          = "Col groups",
            title_fontsize = 12
        )

        ax2_lgd.axis("off")

    #### plot col sums
    plot_col_bar(ax1, V.sum(axis=0), y_lim_quantile=0.9)

    #### left plots
    if row_groups is not None:
        ax3 = fig.add_subplot(gs[5:30, 0])
        ax3_lgd = fig.add_subplot(gs[31:33, 1:20])

        #### plot rows color bar
        colors = [row_groups_colors[x] for x in row_groups]
        plot_row_color_bar(ax3, colors, row_labels, row_labelsize)

        #### plot columns color bar legend
        handles = [Line2D([0],[0], color=v, lw=4, label="%s" % k) for k,v in row_groups_colors.items()]
        handles.sort(key = lambda handle: handle.get_label())

        ax3_lgd.legend(
            handles        = handles,
            loc            = "center",
            frameon        = False,
            ncol           = 3,
            title          = "",
            title_fontsize = 12
        )

        ax3_lgd.axis("off")

    #### main heatmap
    ax4 = fig.add_subplot(gs[5:30, 1:60])
    ax4_cbar = fig.add_subplot(fig.add_subplot(gs[32:33, 20:40]))
    plot_hmap(ax4, ax4_cbar, V, cmap=hmap_cmap, boundaries=hmap_boundaries)

    return fig

####
####
#### 2. PLOT RESULTS FROM RANK SELECTION BY STABILITY
####
####

def plot_W_stab(W_avg, W_std, stab_avg, stab_std, row_groups=None, row_groups_colors=None, row_labels=[], hmap_boundaries=None, hmap_cmap=None, figsize=(24,10)):
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(35, 255, wspace=0, hspace=0.2)

    if row_groups is not None:
        colors = [row_groups_colors[x] for x in row_groups]
    else:
        colors = []

    ####
    #### W avg, hmap
    #### 

    #### upper plot
    ax1 = fig.add_subplot(gs[:5, 3:60])
    plot_col_bar(ax1, W_avg.sum(axis=0), ylabel="sum", yticklabels_format=".2f", y_lim_quantile=1)

    #### left plots
    if row_groups is not None:
        ax2 = fig.add_subplot(gs[5:30, :3])

        #### plot rows color bar
        plot_row_color_bar(ax2, colors, row_labels)

        #### plot columns color bar legend
        handles = [Line2D([0],[0], color=v, lw=4, label="%s" % k) for k,v in row_groups_colors.items()]
        handles.sort(key = lambda handle: handle.get_label())

    #### main heatmap
    ax3 = fig.add_subplot(gs[5:30, 3:60])
    ax3_cbar = fig.add_subplot(fig.add_subplot(gs[31:32, 30:90]))

    plot_hmap(ax3, ax3_cbar, W_avg, cmap=hmap_cmap, boundaries=hmap_boundaries)

    ####
    #### W std, hmap
    #### 

    #### upper plot
    ax1 = fig.add_subplot(gs[:5, 65:125])
    plot_col_bar(ax1, W_std.sum(axis=0), ylabel="", yticklabels_format=".2f", y_lim_quantile=1)

    #### main heatmap
    ax2 = fig.add_subplot(gs[5:30, 65:125])
    plot_hmap(ax2, None, W_std, cmap=hmap_cmap, boundaries=hmap_boundaries, cbar=False)

    ####
    #### W avg, stab_avg, stab_std, row bars
    #### 

    #### upper plot
    ax1 = fig.add_subplot(gs[:5, 138:255])
    plot_col_err(
        ax              = ax1,
        means           = stab_avg,
        errors          = stab_std,
        ylabel          = "silhouette",
        yticklabels_fmt = ".3g",
        fmt             = 's',
        ms              = 10,
        mfc             = 'red',
        mec             = "black",
        mew             = 1,
        ecolor          = 'lightsalmon',
        elw             = 5,
        els             = '-',
        ymax            = 1
    )

    if len(stab_avg) > 1:
        ax1.axhline(
            y     = np.mean(stab_avg),
            color = "red",
            ls    = "--"
        )

        ax1.annotate(
            s        = "%.4g" % np.mean(stab_avg),
            xy       = (len(stab_avg) - 1, np.mean(stab_avg) + 0.1 * (max(stab_avg) - min(stab_avg))),
            xycoords = "data",
            color    = "red"
        )

    #### left plots
    if row_groups is not None:
        ax2 = fig.add_subplot(gs[5:30, 135:138])
        ax2_lgd = fig.add_subplot(gs[31:33, 150:164])

        #### plot rows color bar
        plot_row_color_bar(ax2, colors, row_labels)

        #### plot columns color bar legend
        handles = [Line2D([0],[0], color=v, lw=4, label="%s" % k) for k,v in row_groups_colors.items()]
        handles.sort(key = lambda handle: handle.get_label())

        ax2_lgd.legend(
            handles        = handles,
            loc            = "center",
            frameon        = False,
            ncol           = 3,
            title          = "Row groups",
            title_fontsize = 12
        )

        ax2_lgd.axis("off")

    #### row bar plots
    subgs = gs[5:30, 139:255].subgridspec(1, W_avg.shape[1], wspace=0.05, hspace=0)

    for k in range(W_avg.shape[1]):
        ax3k = fig.add_subplot(subgs[k])
        plot_row_bar(ax=ax3k, widths=W_avg[:,k], xlabel="", colors=colors)

    return fig

def plot_metrics_stab(ranks, stab_avg, stab_std, fid_avg, fid_std, figsize=(14,6)):
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    xplot  = ranks
    means  = stab_avg
    errors = stab_std
    color  = "red"
    ecolor = "lightsalmon"

    eb = ax1.errorbar(
        x          = xplot-0.1,
        y          = means,
        yerr       = errors,
        fmt        = 's',
        mfc        = color,
        mec        = "black",
        ms         = 25,
        mew        = 1,
        ecolor     = ecolor,
        elinewidth = 5,
        capsize    = 0
    )

    eb[-1][0].set_linestyle('-')

    #### esthetics
    ax1.set_xlabel(
        xlabel     = "rank",
        fontsize   = 14,
        fontweight = "medium",
    )
    ax1.set_ylabel(
        ylabel     = "silhouette",
        fontsize   = 14,
        fontweight = "medium",
        ha         = "center",
        va         = "bottom",
        labelpad   = 8
    )

    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(True)

    ax1.tick_params(axis="x", which="both", length=8, pad=4)
    ax1.tick_params(axis="y", which="both", labelcolor=color, length=8, pad=4)
    ax1.set_xticks(xplot)
    ax1.set_xticklabels(["%d" % x for x in xplot])

    ymin = min(np.array(means)-np.array(errors))
    ymax = max(np.array(means)+np.array(errors))

    ax1.set_yticks([ymin, ymax])
    ax1.set_yticklabels([("{:.3g}").format(x) for x in [ymin, ymax]], ha="right", va="center")

    #### new ax with twin x axis
    ax2 = ax1.twinx()

    xplot  = ranks
    means  = fid_avg
    errors = fid_std
    color  = "dodgerblue"
    ecolor = "lightblue"

    eb = ax2.errorbar(
        x          = xplot+0.1,
        y          = means,
        yerr       = errors,
        fmt        = 's',
        mfc        = color,
        mec        = "black",
        ms         = 25,
        mew        = 1,
        ecolor     = ecolor,
        elinewidth = 5,
        capsize    = 0
    )

    eb[-1][0].set_linestyle('-')

    #### esthetics
    ax2.set_ylabel(
        ylabel     = "fidelity",
        fontsize   = 14,
        fontweight = "medium",
        ha         = "center",
        va         = "top",
        labelpad   = 8
    )

    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(True)

    ax2.tick_params(axis="y", which="both", labelcolor=color, length=8, pad=6)

    ymin = min(np.array(means)-np.array(errors))
    ymax = max(np.array(means)+np.array(errors))

    ax2.set_yticks([ymin, ymax])
    ax2.set_yticklabels([("{:.5g}").format(x) for x in [ymin, ymax]])

    return fig

####
####
#### 3. PLOT RESULTS FROM SIGNLE NMF (WITH OPTIONAL COMPARISON TO REF)
####
####

def plot_W_alone(W_fit, W_ref=None, row_groups=None, row_groups_colors=None, row_labels=[], hmap_boundaries=None, hmap_cmap=None, col_ref_labels=[], figsize=(16,10)):
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(35, 195, wspace=0, hspace=0.2)

    if row_groups is not None:
        colors = [row_groups_colors[x] for x in row_groups]
    else:
        colors = ["blue" for _ in range(W_fit.shape[1])]

    ####
    #### LEFT PART
    #### 

    #### upper plot
    ax1 = fig.add_subplot(gs[:5, 3:60])
    plot_col_bar(ax1, W_fit.sum(axis=0), ylabel="sum", yticklabels_format=".2f", y_lim_quantile=1)

    #### left plots
    if row_groups is not None:
        ax2 = fig.add_subplot(gs[5:30, :3])

        #### plot rows color bar
        plot_row_color_bar(ax2, colors, row_labels)

        #### plot columns color bar legend
        handles = [Line2D([0],[0], color=v, lw=4, label="%s" % k) for k,v in row_groups_colors.items()]
        handles.sort(key = lambda handle: handle.get_label())

    #### main heatmap
    ax3 = fig.add_subplot(gs[5:30, 3:60])
    ax3_cbar = fig.add_subplot(fig.add_subplot(gs[31:32, 3:60]))

    plot_hmap(
        ax         = ax3,
        cbar_ax    = ax3_cbar,
        X          = W_fit,
        cmap       = hmap_cmap,
        boundaries = hmap_boundaries
    )

    ####
    #### RIGHT PART
    #### 

    if W_ref is not None:
        ax1 = fig.add_subplot(gs[:5, 73:190])
        ax1_cbar = fig.add_subplot(gs[:5, 192:194])

        #### upper plot, cosine similarity hmap
        plot_hmap(
            ax         = ax1,
            cbar_ax    = ax1_cbar,
            X          = cosine_similarity(X  = W_ref.T, Y = W_fit.T),
            linewidths = 1,
            linecolor  = "white",
            cmap       = "YlGnBu",
            boundaries = np.linspace(0,1,11),
            cbar_kws   = {
                "spacing"     : "uniform",
                "ticks"       : np.linspace(0,1,11)[1:-1: 2],
                "format"      : "%g",
                "fraction"    : 0.05,
                "orientation" : "vertical",
                "aspect"      : 10
            },
            cbar_title  = "cosine sim",
            yticklabels = col_ref_labels,
            labelsize   = 8
        )

    #### left plots, row color bar
    if row_groups is not None:
        ax2 = fig.add_subplot(gs[5:30, 70:73])
        ax2_lgd = fig.add_subplot(gs[31:33, 85:99])

        #### plot rows color bar
        plot_row_color_bar(ax2, colors, row_labels)

        #### plot columns color bar legend
        handles = [Line2D([0],[0], color=v, lw=4, label="%s" % k) for k,v in row_groups_colors.items()]
        handles.sort(key = lambda handle: handle.get_label())

        ax2_lgd.legend(
            handles        = handles,
            loc            = "center",
            frameon        = False,
            ncol           = 3,
            title          = "",
            title_fontsize = 12
        )

        ax2_lgd.axis("off")

    #### mid plots, row bars
    subgs = gs[5:30, 74:190].subgridspec(1, W_fit.shape[1], wspace=0.05, hspace=0)

    for k in range(W_fit.shape[1]):
        ax3k = fig.add_subplot(subgs[k])
        plot_row_bar(ax=ax3k, widths=W_fit[:,k], xlabel="", colors=colors)

    return fig
