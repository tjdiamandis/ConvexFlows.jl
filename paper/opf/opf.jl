using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Random, LinearAlgebra, SparseArrays
using BenchmarkTools
using Convex, MosekTools, SCS
using StatsBase, LogExpFunctions
using Graphs: Graph, connected_components
import GraphPlot
import Cairo
using Plots, LaTeXStrings

const GP = GraphPlot

Pkg.activate(joinpath(@__DIR__, "..", ".."))
using ConvexFlows
const CF = ConvexFlows

const FIGPATH = joinpath(@__DIR__, "..", "figures")

include("edges.jl")
include("objectives.jl")
include("utils.jl")


# ******************************************************************************
# ******************************************************************************
# I. Convergence of dual gap and objective value over iterations
# ******************************************************************************
# ******************************************************************************
# Plot example network
n = 100
Adj, xys = build_graph(n, d=0.11, α=0.8)
avg_degree = sum(sum(Adj, dims=1)) / n
network_fig_n100 = GP.gplot(
    Graph(Adj),
    [xy[1] for xy in xys],
    [xy[2] for xy in xys],
    NODESIZE=2.0/n,
    nodefillc="black",
    edgestrokec="dark gray",
)
GP.draw(GP.PDF(joinpath(FIGPATH, "network_n100.pdf"), 16GP.cm, 16GP.cm), network_fig_n100)

lines = TransmissionLine[]
for i in 1:n, j in i+1:n
    if Adj[i, j] > 0
        bi = rand((1., 2., 3.))
        push!(lines, TransmissionLine(bi, [i, j]))
        push!(lines, TransmissionLine(bi, [j, i]))
    end
end
m = length(lines)

# Define objective function
d = rand((0.5, 1., 2.), n)
Uy = QuadraticPowerCost(d)


# Solve with Mosek
prob_cvx, y_cvx, xs_cvx = build_convex_model(d, lines)
Convex.solve!(
    prob_cvx,
    Mosek.Optimizer(),
    silent_solver=true
)
prob_cvx.status != Convex.MOI.OPTIMAL && @warn "Problem not solved by Mosek!"
pstar = prob_cvx.optval


s = Solver(
    flow_objective=Uy,
    edges=lines,
    # edges=Edge[TransmissionLine(0., [1,2])],
    n=n,
)



dual_gaps = zeros(ITERS_NEEDED+1)
obj_diffs = zeros(ITERS_NEEDED+1)
dual_gaps[1] = NaN
obj_diffs[1] = (pstar - U(Uy, zeros(n))) / max(abs(pstar), abs(U(Uy, zeros(n))))

ITERS_NEEDED = 31
for max_iter in 1:ITERS_NEEDED
    CF.solve!(s, verbose=false, factr=1e1, memory=5, max_iter=max_iter)
    p_iter = U(Uy, s.y)
    d_iter = Ubar(Uy, s.ν) + sum(dot(s.xs[i], s.ν[lines[i].Ai]) for i in 1:m)
    # dual_gap = (d_iter - p_iter)
    dual_gap = (d_iter - p_iter) / max(abs(d_iter), abs(p_iter))
    
    dual_gaps[max_iter+1] = dual_gap
    obj_diffs[max_iter+1] = (pstar - p_iter)/max(abs(pstar), abs(p_iter))
end

iter_conv_plt = plot(
    0:ITERS_NEEDED,
    obj_diffs,
    yaxis=:log,
    label="Objective difference",
    xlabel="Iteration",
    # ylabel="Dual gap",
    # legend=:bottomright,
    # size=(400, 300),
    # ylims=(0, 1),
    yticks=10. .^ (-12:2:0),
    # xticks=0:5:ITERS_NEEDED,
    # grid=false,
    linewidth=3,
    color=:blue,
    linestyle=:dash,
    legendfontsize=12,
    tickfontsize=12,
    guidefontsize=12,
    legendtitlefontsize=12,
)

plot!(
    iter_conv_plt, 
    0:ITERS_NEEDED, 
    dual_gaps, 
    label="Duality gap",
    color=:black,
    linewidth=3,
)
savefig(iter_conv_plt, joinpath(FIGPATH, "opf-iter-conv.pdf"))
# ******************************************************************************
# ******************************************************************************

