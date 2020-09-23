"""
Created on Mon Apr 13 2020

@author: Yoann Pradat

Plot divergences
"""

include("./src/VNMF.jl")
using .VNMF
using LaTeXStrings
using Plots

#### # Plot β-divergence for different values of β
#### #################################################################################################################

βys = [-1, 0, 1, 2]
βzs = [-0.5, 0.5, 1.5, 2.5]
titles = [L"(a) \beta < 0", L"(b) 0 \leq \beta < 1", L"(c) 1 \leq \beta < 2", L"(d) 2 \leq \beta"]

h = 0.05
p = plot(layout=(2,2), dpi=300, size=(1000, 600))

for sp=1:4
    βy = βys[sp]
    βz = βzs[sp]

    if βz < 1
	X = h:h:8
    else
	X = h:h:4
    end

    Y = [β_divergence(hcat([1.]), hcat([x]), βy) for x in X]
    Z = [β_divergence(hcat([1.]), hcat([x]), βz) for x in X]

    plot!(
	X, 
	hcat(Y, Z), 
	title = titles[sp],
	labels = ["\$\\beta = $(βy)\$" "\$\\beta = $(βz)\$" ],
	ls    = [:solid :dash],
	lw    = 1,
	color = "black",
	subplot = sp
    )

    plot!(
	xlabel = "",
	xlims  = (0,X[end]),
	xticks = 0:X[end]/4:X[end],
	ylims  = (0,1.5),
	yticks = 0:0.5:1.5,
	subplot = sp
    )
end

folder   = "demo/plot/divergence"
mkpath(folder)
filename = "beta_divergences.pdf"
filepath = joinpath(folder, filename)

savefig(p, filepath)
println("plot saved at $(filepath)")

