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

Pkg.activate(joinpath(@__DIR__, "..", ".."))
using ConvexFlows

const GP = GraphPlot
const CF = ConvexFlows

const FIGPATH = joinpath(@__DIR__, "..", "figures")

include("edges.jl")
include("objectives.jl")
include("utils.jl")
include("convex.jl")
# ******************************************************************************


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


# Solve with ConvexFlows.jl
s = Solver(
    flow_objective=Uy,
    edges=lines,
    # edges=Edge[TransmissionLine(0., [1,2])],
    n=n,
)

ITERS_NEEDED = 31
dual_gaps = zeros(ITERS_NEEDED+1)
obj_diffs = zeros(ITERS_NEEDED+1)
dual_gaps[1] = NaN
obj_diffs[1] = (pstar - U(Uy, zeros(n))) / max(abs(pstar), abs(U(Uy, zeros(n))))

# Hack to get iterations out of LBFGSB.jl
for max_iter in 1:ITERS_NEEDED
    CF.solve!(s, verbose=false, factr=1e1, memory=5, max_iter=max_iter)
    p_iter = U(Uy, s.y)
    d_iter = Ubar(Uy, s.ν) + sum(dot(s.xs[i], s.ν[lines[i].Ai]) for i in 1:m)
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
    legend=:topright,
    yticks=10. .^ (-12:2:0),
    linewidth=3,
    color=:blue,
    linestyle=:dash,
    legendfontsize=12,
    tickfontsize=12,
    guidefontsize=12,
    legendtitlefontsize=12,
    dpi=300,
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
# ******************************************************************************
# II. Solve time comparisons
# ******************************************************************************
# ******************************************************************************

function run_trial_jump(d, lines, optimizer=Mosek.Optimizer())
    prob_cvx, y_cvx, xs_cvx = build_convex_model(d, lines)
    GC.gc()
    model = Convex.solve!(
        prob_cvx,
        Mosek.Optimizer(),
        silent_solver=true
    )
    solve_time = Convex.MOI.get(model, Convex.MOI.SolveTimeSec())
    prob_cvx.status != Convex.MOI.OPTIMAL && @warn "Problem not solved by Mosek!"
    pstar = prob_cvx.optval
    return pstar, solve_time
end

function run_trial_convexflows(d, lines)
    n = length(d)
    m = length(lines)

    Uy = QuadraticPowerCost(d)
    s = Solver(
        flow_objective=Uy,
        edges=lines,
        n=n,
    )
    GC.gc()
    tt = @timed CF.solve!(s, verbose=false, factr=1e1, memory=5)
    solve_time = tt.time
    pstar = U(Uy, s.y)
    dstar = Ubar(Uy, s.ν) + sum(dot(s.xs[i], s.ν[lines[i].Ai]) for i in 1:m)
    dual_gap = (dstar - pstar) / max(abs(dstar), abs(pstar))
    
    return pstar, dual_gap, solve_time
end

function run_trial(n)
    Adj, _ = build_graph(n, d=0.11, α=0.8)
    @info "  Finished building graph..."

    lines = TransmissionLine[]
    for i in 1:n, j in i+1:n
        if Adj[i, j] > 0
            bi = rand((1., 2., 3.))
            push!(lines, TransmissionLine(bi, [i, j]))
            push!(lines, TransmissionLine(bi, [j, i]))
        end
    end

    # Define objective function
    d = rand((0.5, 1., 2.), n)

    p_mosek, t_mosek = run_trial_jump(d, lines)
    @info "  Finished running Mosek..."
    p_cf, gap_cf, t_cf = run_trial_convexflows(d, lines)
    @info "  Finished running ConvexFlows..."
    return p_mosek, p_cf, gap_cf, t_mosek, t_cf
end

function run_trials(ns)
    ts_mosek = zeros(length(ns))
    ts_cf = zeros(length(ns))
    for (i, n) in enumerate(ns)
        @info "Starting trial for n=$n..."
        p_mosek, p_cf, gap_cf, t_mosek, t_cf = run_trial(n)
        rel_obj_diff = (p_mosek - p_cf) / max(abs(p_mosek), abs(p_cf))
        @info "  reldiff:  $rel_obj_diff"
        @info "  dualgap:  $gap_cf"
        @info "-- Finished! --"
        ts_mosek[i] = t_mosek
        ts_cf[i] = t_cf
    end
    return ts_mosek, ts_cf
end

ns = 10. .^ (2:0.2:4) |> x -> round.(Int, x)
ts_mosek, ts_cf = run_trials(ns)

time_plt = plot(
    ns,
    ts_mosek,
    label="Mosek",
    xlabel="Number of nodes",
    ylabel="Solve time (s)",
    yscale=:log,
    legend=:bottomright,
    # size=(400, 300),
    # ylims=(1e-3, 1e3),
    # yticks=10. .^ (-3:3),
    # xticks=10. .^ (2:4),
    # grid=false,
    linewidth=3,
    color=:blue,
    linestyle=:dash,
    legendfontsize=12,
    tickfontsize=12,
    guidefontsize=12,
    legendtitlefontsize=12,
)
plot!(time_plt, ns, ts_cf, label="ConvexFlows", color=:black, linewidth=3)
savefig(time_plt, joinpath(FIGPATH, "opf-time.pdf"))
