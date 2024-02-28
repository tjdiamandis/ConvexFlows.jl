using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Random, LinearAlgebra, SparseArrays
using BenchmarkTools
using MosekTools, SCS, JuMP
using StatsBase, LogExpFunctions
using Graphs: Graph, connected_components
import GraphPlot
import Cairo
using Plots, LaTeXStrings
using JLD2

Pkg.activate(joinpath(@__DIR__, "..", ".."))
using ConvexFlows

const GP = GraphPlot
const CF = ConvexFlows

const FIGPATH = joinpath(@__DIR__, "..", "figures")
const SAVEPATH = joinpath(@__DIR__, "..", "data")
const SAVEFILE = joinpath(SAVEPATH, "opf.jld2")

include("edges.jl")
include("objectives.jl")
include("utils.jl")
include("jump.jl")
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
model, _ = build_jump_opf_model(lines, d, verbose=false, mosek_high_precision=true)
optimize!(model)
termination_status(model) ∉ (MOI.OPTIMAL, MOI.SLOW_PROGRESS) && @warn "Problem not solved by Mosek!"
pstar = objective_value(model)


# Solve with ConvexFlows.jl
s = Solver(
    flow_objective=Uy,
    edges=lines,
    # edges=Edge[TransmissionLine(0., [1,2])],
    n=n,
)

ITERS_NEEDED = 43
dual_gaps = zeros(ITERS_NEEDED+1)
obj_diffs = zeros(ITERS_NEEDED+1)
rp_norms = zeros(ITERS_NEEDED+1)
dual_gaps[1] = NaN
obj_diffs[1] = (pstar - U(Uy, zeros(n))) / max(abs(pstar), abs(U(Uy, zeros(n))))
rp_norms[1] = NaN
rp = zeros(n)
y_hat = zeros(n)

# Hack to get iterations out of LBFGSB.jl
for max_iter in 1:ITERS_NEEDED
    CF.solve!(s, verbose=false, factr=1, memory=10, max_iter=max_iter, final_netflows=false, pgtol=eps())
    y_hat .= 0
    for i in 1:m
        y_hat[lines[i].Ai] .+= s.xs[i]
    end

    p_hat = U(Uy, y_hat)
    d_iter = Ubar(Uy, s.ν) + sum(dot(s.xs[i], s.ν[lines[i].Ai]) for i in 1:m)
    dual_gap = (d_iter - p_hat) / max(abs(d_iter), abs(p_hat))
    rp .= s.y
    for i in 1:m
        rp[lines[i].Ai] .-= s.xs[i]
    end

    rp_norms[max_iter+1] = norm(rp) / max(norm(s.y), norm(rp .- s.y))
    dual_gaps[max_iter+1] = dual_gap
    obj_diffs[max_iter+1] = (pstar - p_hat)/max(abs(pstar), abs(p_hat))
end

iter_conv_plt = plot(
    0:ITERS_NEEDED,
    obj_diffs,
    yaxis=:log,
    label="Objective difference",
    xlabel="Iteration",
    legend=:bottomleft,
    yticks=10. .^ (-15:2:0),
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
    rp_norms,
    yaxis=:log,
    label="Primal Residual",
    linewidth=3,
    color=:red,
    linestyle=:dot,
)

plot!(
    iter_conv_plt, 
    0:ITERS_NEEDED, 
    dual_gaps .+ 10eps(), 
    label="Duality gap",
    color=:black,
    linewidth=3,
)
plot!(
    iter_conv_plt, 
    0:ITERS_NEEDED, 
    sqrt(eps()).*ones(ITERS_NEEDED+1), 
    label=L"$\sqrt{\texttt{eps}}$",
    color=:black,
    linestyle=:dash,
    linewidth=1,
)
savefig(iter_conv_plt, joinpath(FIGPATH, "opf-iter-conv.pdf"))
# ******************************************************************************


# ******************************************************************************
# ******************************************************************************
# II. Solve time comparisons
# ******************************************************************************
# ******************************************************************************

function run_trial_jump(d, lines, optimizer=Mosek.Optimizer())
    model, _ = build_jump_opf_model(lines, d, verbose=false)
    GC.gc()
    optimize!(model)
    time = solve_time(model)
    st = termination_status(model)
    st ∉ (MOI.OPTIMAL, MOI.SLOW_PROGRESS) && @info "    termination status: $st"
    pstar = objective_value(model)
    return pstar, time
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

function run_trial(n; rseed=1, verbose=false)
    Adj, _ = build_graph(n, d=0.11, α=0.8, rseed=rseed)
    verbose && @info "  Finished building graph..."

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
    verbose && @info "  Finished running Mosek..."
    p_cf, gap_cf, t_cf = run_trial_convexflows(d, lines)
    verbose && @info "  Finished running ConvexFlows..."
    return p_mosek, p_cf, gap_cf, t_mosek, t_cf
end

function run_trials(ns; trials=10, verbose=false)
    # compile
    run_trial(50)

    ts_mosek = zeros(trials, length(ns))
    ts_cf = zeros(trials, length(ns))
    for (i, n) in enumerate(ns)
        @info "Starting trial for n=$n..."
        for t in 1:trials
            p_mosek, p_cf, gap_cf, t_mosek, t_cf = run_trial(n; rseed=t, verbose=verbose)
            rel_obj_diff = (p_mosek - p_cf) / max(abs(p_mosek), abs(p_cf))
            verbose && @info "  reldiff:  $rel_obj_diff"
            rel_obj_diff > 1e-3 && @warn "  Possible incorrect solution!"
            verbose && @info "  dualgap:  $gap_cf"
            gap_cf > 1e-3 && @warn "  ConvexFlows didn't produce a solution!"
            ts_mosek[t, i] = t_mosek
            ts_cf[t, i] = t_cf
        end
        @info "-- Finished! --"
    end
    return ts_mosek, ts_cf
end

ns = 10. .^ (2:0.2:5) |> x -> round.(Int, x)
ts_mosek, ts_cf = run_trials(ns)
save(SAVEFILE,
    "ns", ns,
    "ts_mosek", ts_mosek,
    "ts_cf", ts_cf,
)


# ******************************************************************************
# Plot solve times
# ******************************************************************************
ns, ts_mosek, ts_cf = load(SAVEFILE, "ns", "ts_mosek", "ts_cf")

ts_med_mosek = median(ts_mosek, dims=1) |> vec
ts_med_cf = median(ts_cf, dims=1) |> vec
q75_mosek = [quantile(ts_mosek[:,i], 0.75) for i in 1:length(ns)] |> vec
q25_mosek = [quantile(ts_mosek[:,i], 0.25) for i in 1:length(ns)] |> vec
q75_cf = [quantile(ts_cf[:,i], 0.75) for i in 1:length(ns)] |> vec
q25_cf = [quantile(ts_cf[:,i], 0.25) for i in 1:length(ns)] |> vec

max_mosek = maximum(ts_mosek, dims=1) |> vec
max_cf = maximum(ts_cf, dims=1) |> vec
for (i, n) in enumerate(ns)
    if max_mosek[i] > 5*ts_med_mosek[i]
        println("Trial: n = $n")
        println("  max mosek: $(round(max_mosek[i], digits=3))")
        println("  med mosek: $(round(ts_med_mosek[i], digits=3))")
    end
    if max_cf[i] > 5*ts_med_cf[i]
        println("Trial: n = $n")
        println("  max cf: $(round(max_cf[i], digits=3))")
        println("  med cf: $(round(ts_med_cf[i], digits=3))")
    end
end

time_plt = plot(
    ns,
    ts_med_mosek,
    ribbon=(ts_med_mosek .- q25_mosek, q75_mosek .- ts_med_mosek),
    fillalpha=0.5,
    label="Mosek",
    xlabel="Number of nodes",
    ylabel="Solve time (s)",
    yscale=:log10,
    xscale=:log10,
    legend=:bottomright,
    minorgrid=true,
    yticks=10. .^ (-3:3),
    xticks=10. .^ (2:5),
    ylims=(1e-3, 5e2),
    linewidth=3,
    color=:blue,
    linestyle=:dash,
    legendfontsize=12,
    tickfontsize=12,
    guidefontsize=12,
    legendtitlefontsize=12,
    dpi=300,
)
# Plot maxes
plot!(
    time_plt,
    ns,
    max_mosek,
    label=nothing,
    color=:blue,
    seriestype=:scatter,
    markersize=3,
)

plot!(
    time_plt,
    ns,
    ts_med_cf,
    ribbon=(ts_med_cf .- q25_cf, q75_cf .- ts_med_cf),
    fillalpha=0.5,
    label="ConvexFlows.jl",
    color=:black,
    linewidth=3
)
# Plot maxes
plot!(
    time_plt,
    ns,
    max_cf,
    label=nothing,
    color=:black,
    seriestype=:scatter,
    markersize=3,
)
savefig(time_plt, joinpath(FIGPATH, "opf-time.pdf"))
