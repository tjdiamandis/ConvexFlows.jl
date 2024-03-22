using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra, Random, SparseArrays
using StatsBase, LogExpFunctions
import Graphs: Graph, connected_components
using Plots, LaTeXStrings

Pkg.activate(joinpath(@__DIR__, "..", ".."))
using ConvexFlows

const FIGPATH = joinpath(@__DIR__, "..", "figures")

include(joinpath(@__DIR__, "..", "opf", "utils.jl"))
# ******************************************************************************

# Problem parameters
Random.seed!(1)
n = 100
T = 2
N = n*T

# Objective function
d = 4*rand(N) .+ 1.0
obj = NonpositiveQuadratic(d)

# Network
edges = Edge[]
Adj, xys = build_graph(n)
# Transmission Lines
h(w) = 3w - 16.0*(log1pexp(0.25 * w) - log(2))
wstar(ηrat, b) = ηrat ≥ 1.0 ? 0.0 : min(4.0 * log((3.0 - ηrat)/(1.0 + ηrat)), b)
for i in 1:n, j in i+1:n
    iszero(Adj[i, j]) && continue
    bi = rand((1., 2., 3.))

    for t in 1:T
        it = i + (t-1)*n
        jt = j + (t-1)*n
        push!(edges, Edge(
            (it, jt); h=h, ub=bi, wstar = @inline ηrat -> wstar(ηrat, bi))
        )
        push!(edges, Edge(
            (jt, it); h=h, ub=bi, wstar = @inline ηrat -> wstar(ηrat, bi)
        ))
    end
end

# Storage edges
ϵ = 1e-4
wstar_storage(ηrat, γ, b) = ηrat ≥ γ ? 0.0 : min(1/ϵ*(γ - ηrat), b)
for i in 1:n, t in 1:T-1
    it = i + (t-1)*n
    it_next = i + t*n
    γi = rand((0,1))*(0.5 + 0.5*rand())
    !iszero(γi) && push!(edges, Edge(
        (it, it_next); 
        h= w->γi*w - ϵ/2*w^2, 
        ub=Inf, 
        wstar = @inline ηrat -> wstar_storage(ηrat, γi, edges[i].ub)
    ))
end

prob = problem(obj=obj, edges=edges)
result_bfgs = solve!(prob; method=:bfgs)
solver_log = result_bfgs.log

# Plot stuff
# ******************************************************************************
pfeas = solver_log.g_norm
dual_val = solver_log.fx
primal_val = similar(dual_val)
xs_tmp = [zeros(2) for _ in 1:length(edges)]

for k in 1:length(dual_val)
    νt = solver_log.xk[k]
    ConvexFlows.find_arb!(xs_tmp, νt, prob.edges)
    yhat = ConvexFlows.netflows(xs_tmp, prob.edges, prob.n)
    primal_val[k] = U(obj, yhat)
end
gap = (dual_val .- primal_val)
# gap = (dual_val .- primal_val) ./ max.(abs.(dual_val), abs.(primal_val))


iters_total = length(dual_val) - 1
iter_conv_plt = plot(
    0:iters_total,
    # max.(gap, eps()),
    abs.(gap) .+ eps(),
    yaxis=:log,
    label="Duality gap",
    xlabel="Iteration",
    legend=:left,
    ylims=(1e-10, 1e3),
    yticks=10. .^ (-10:2:3),
    linewidth=3,
    color=:blue,
    legendfontsize=12,
    tickfontsize=12,
    guidefontsize=12,
    legendtitlefontsize=12,
    dpi=300,
)
 plot!(
    iter_conv_plt,
    0:iters_total,
    pfeas,
    yaxis=:log,
    label="Net flow residual",
    linewidth=3,
    color=:red,
)
plot!(
    iter_conv_plt, 
    0:iters_total, 
    sqrt(eps()).*ones(iters_total+1), 
    label=L"$\sqrt{\texttt{eps}}$",
    color=:black,
    linestyle=:dash,
    linewidth=1,
)
savefig(iter_conv_plt, joinpath(FIGPATH, "opf-two-node-conv.pdf"))