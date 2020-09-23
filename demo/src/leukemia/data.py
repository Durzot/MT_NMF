# -*- coding: utf-8 -*-
"""
Created on Fri 15 May 2020

@author: Yoann Pradat

Visualize data files from all_aml_38 dataset. Dataset used in
    1. Golub et al. Molecular classification of cancer: class discovery and class prediction by gene expression
    monitoring, 1999.
    2. Brunet et al. Metagenes and molecular class discovery using matrix factorization, 2004.
"""


import os
import numpy  as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from   matplotlib        import cm

from sklearn.linear_model import LinearRegression

#### # 1. LOAD DATA
#### # ############################################################################################################

folder   =  "../../../data/all_aml_38"
filename = "ALL_vs_AML_train_set_38_sorted.res.txt"
filepath = os.path.join(folder, filename)

df_train = pd.read_csv(
    filepath_or_buffer = filepath,
    sep = "\t"
)

#### drop null lines and columns
df_train = df_train.iloc[2:, :]
df_train = df_train.loc[:, df_train.isnull().mean(axis=0) < 1]

#### rename columns containin A, P, M entries for each sample
old_columns = df_train.columns
new_columns = []

for i, c in enumerate(old_columns):
    if c.startswith("Unnamed"):
        new_columns.append("type_%s" % old_columns[i-1])
    else:
        new_columns.append(c)

df_train.columns = new_columns


#### check regression slopes
print("Checking regression slopes genes expression ...")

df_baseline = df_train.iloc[:, 2:4]
for j in range(4, 78, 2):
    df_sample = df_train.iloc[:, j:j+2]
    mask_common_P = (df_baseline.iloc[:,1] == "P") & (df_sample.iloc[:,1] == "P")
    X = df_baseline.loc[mask_common_P].iloc[:,0].values
    Y = df_sample.loc[mask_common_P].iloc[:,0].values

    linreg = LinearRegression(fit_intercept=True, normalize=False)
    linreg = linreg.fit(X.reshape(-1,1), Y)

    print("coef linear reg between %s and %s: %.6g" % (df_baseline.columns[0], df_sample.columns[0], linreg.coef_))


#### # 2. HEATMAP 50 MOST CORRELATED GENES WITH AML-ALL DISTINCTION
#### # ############################################################################################################

X_train = df_train[["Accession"] + [x for x in df_train.columns if x.startswith("AML") or x.startswith("ALL")]]
X_train = X_train.set_index("Accession")

#### center and normalize gene expressions
X_train_norm = ((X_train.T - X_train.T.mean(axis=0))/X_train.T.std(axis=0)).T

#### select genes for the heatmap from the paper
all_genes = ["U22376", "X59417", "U05259", "M92287", "M31211", "X74262", "D26156", "S50223", "M31523", "L47738"]
aml_genes = ["M55150", "X95735", "U05136", "M16038", "U82759", "M23197", "M84526", "Y12670", "M27891", "X17042"]

acc_all_genes = [x for y in all_genes for x in X_train.index if x.startswith(y)]
acc_aml_genes = [x for y in aml_genes for x in X_train.index if x.startswith(y)]

X_heatmap = X_train_norm.loc[acc_all_genes + acc_aml_genes, :]

#### plot heatmap
folder = "demo_aml_all/out"
filename = "golub_20_genes_heatmap.pdf"
filepath = os.path.join(folder, filename)
fig, ax = plt.subplots(figsize=(14, 8))

boundaries = np.linspace(-3, 3, 13)
cmap = sns.diverging_palette(240, 10, n=13, s=100, l=50, sep=1, as_cmap=True)

sns.heatmap(
    X_heatmap,
    linecolor="royalblue",
    linewidths=0.75,
    cmap=cmap,
    vmin=-3,
    vmax=3,
    square=True,
    xticklabels=X_heatmap.columns.str[:3],
    yticklabels=X_heatmap.index,
    ax=ax,
    norm=cm.colors.BoundaryNorm(boundaries=boundaries, ncolors=256),
    cbar_kws={
        'spacing': 'uniform',
        'ticks': boundaries,
        'format': '%g',
        'fraction': 0.035,
        'pad': 0.08,
        'orientation': "horizontal",
        'aspect': 30
    }
)

ax.set_xlabel("")
ax.set_ylabel("")

#### colorobar
ax.figure.axes[-1].set_xlabel(
    xlabel = 'Normalized gene expression',
    size   = "large",
)
plt.savefig(filepath)


#### # 3. SAVE NORMALIZED MATRIX
#### # ############################################################################################################

#### TO FORCE POSITIVY, FLOOR VALUES TO 20: ARBITRARY
X_train[X_train < 20 ] = 20

folder = "demo_aml_all/out"
filename = "X_train_pos.txt"
filepath = os.path.join(folder, filename)

X_train.to_csv(
    path_or_buf = filepath,
    sep         = "\t",
    header      = True,
    index       = True
)
