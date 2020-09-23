"""
Created on Tue May 05 2020

@author: Yoann Pradat

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Plots consensus matrices as produced by rank validation of NMF algorithms.
"""

from   matplotlib                          import cm
import matplotlib.patches                   as    mpatches
import matplotlib.pyplot                    as    plt
import numpy                                as    np
import os
import pandas                               as    pd
import seaborn                              as    sns

#### type alias
DataFrame = pd.core.frame.DataFrame
Figure    = plt.Figure


#### # HEATMAP
#### # #############################################################################################################

from dataclasses import dataclass, field
from typing import Dict, List, Union

def default_field(obj):
    return field(default_factory=lambda: obj)

@dataclass
class ConsensusConfig:
    colorbar: Dict[str, Union[int, float, str]] = default_field({
        'title_fontsize': 15,
        'fontsize': 12,
        'n_bins': 12,
    })


def plot_consensus(df_C: DataFrame, config: ConsensusConfig) -> Figure:
    fig, ax = plt.subplots(figsize=(14, 10))

    #### define color bins
    boundaries = [i/config.colorbar['n_bins'] for i in range(config.colorbar['n_bins']+1)]

    #### color map from blue to blue/red
    cmap = sns.diverging_palette(250, 20, s=70, l=50, sep=1, as_cmap=True)

    sns.heatmap(
        df_C,
        linecolor='white',
        linewidths=0,
        cmap=cmap,
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        norm=cm.colors.BoundaryNorm(boundaries=boundaries, ncolors=256),
        cbar_kws={
            'spacing': 'uniform',
            'ticks': [0.2, 0.4, 0.6, 0.8],
            'format': '%g',
            'fraction': 0.025,
            'pad': 0.08,
            'orientation': "horizontal",
            'aspect': 30
        }
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    #### colorobar
    ax.figure.axes[-1].set_xlabel(
        xlabel = 'Consensus index',
        size   = config.colorbar['title_fontsize'],
    )

    fig.tight_layout()
    return fig
