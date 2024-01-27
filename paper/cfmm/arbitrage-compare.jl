using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Random, LinearAlgebra
using BenchmarkTools
using JuMP, MosekTools, SCS
using StatsBase
using Plots, LaTeXStrings

Pkg.activate(joinpath(@__DIR__, "..", ".."))
using ConvexFlows
const CF = ConvexFlows

const FIGPATH = joinpath(@__DIR__, "..", "figures")

include("cfmms.jl")
include("objectives.jl")
include("jump.jl")
include("utils.jl")


function run_trial_flows(; Uy, Vis=nothing, cfmms, memory=5, verbose=false, factr=1e-1)
    m = length(cfmms)
    toks = union([Set(cfmm.Ai) for cfmm in cfmms]...)
    n = maximum(toks)
    @assert length(toks) == n

    if isnothing(Vis)
        s = Solver(
            flow_objective=Uy,
            edges=cfmms,
            n=n,
        )
    else
        s = Solver(
            flow_objective=Uy,
            edge_objectives=Vis,
            edges=cfmms,
            n=n
        )
    end

    GC.gc()
    time = solve!(s, verbose=verbose, factr=factr, memory=memory)
    
    pstar = U(Uy, s.y)
    dstar = Ubar(Uy, s.ν)
    if isnothing(Vis)
        dstar += sum(dot(s.ν[cfmms[i].Ai], s.xs[i]) for i in 1:m)
    else
        pstar += sum(U(Vis[i], s.xs[i]) for i in 1:m)
        dstar += sum(
            Ubar(Vis[i], s.ηts[i]) + 
            dot(s.ν[cfmms[i].Ai] + s.ηts[i], s.xs[i]) 
            for i in 1:m
        )
    end
    dual_gap = (dstar - pstar) / max(abs(dstar), abs(pstar))

    !valid_trades(cfmms, xs=s.xs) && @warn "Some invalid trades for router solution"
    avg_subopt = check_optimality(s.ν, cfmms, xs=s.xs)
    Uy_violation = norm(max.(-s.y, 0.0)) / norm(s.y)
    return time, pstar, dual_gap, Uy_violation
end


function run_trial(m; rseed=1)
    n = round(Int, 2*sqrt(m))
    cfmms = build_pools(m, n; swap_only=true, rseed=rseed)
    @info "  Finished building graph..."
    
    # Objective function
    min_price = 1e-2
    max_price = 1.0
    Random.seed!(rseed)
    c = rand(n) .* (max_price - min_price) .+ min_price
    Uy = ArbitragePenalty(c)
    
    # Mosek
    Vis_zero = true
    t_mosek, p_mosek = run_trial_jump(
        cfmms, 
        c; 
        Vis_zero=Vis_zero, 
        optimizer=() -> Mosek.Optimizer()
    )
    @info "  Finished running Mosek..."
    
    t_cf, p_cf, gap_cf, rp_cf = 
        run_trial_flows(
            Uy=Uy,
            cfmms=cfmms,
            verbose=false,
        )
    @info "  Finished running ConvexFlows..."
    
    return p_mosek, p_cf, gap_cf, rp_cf, t_mosek, t_cf
end

function run_trials(ms)
    ts_mosek = zeros(length(ms))
    ts_cf = zeros(length(ms))
    for (i, m) in enumerate(ms)
        @info "Starting trial for m=$m..."
        p_mosek, p_cf, gap_cf, rp_cf, t_mosek, t_cf = run_trial(m)
        rel_obj_diff = (p_mosek - p_cf) / max(abs(p_mosek), abs(p_cf))
        @info "  reldiff:  $rel_obj_diff"
        rel_obj_diff > 1e-3 && @warn "  Possible incorrect solution!"
        @info "  dualgap:  $gap_cf"
        @info "  rp norm:  $rp_cf"
        @info "-- Finished! --"
        ts_mosek[i] = t_mosek
        ts_cf[i] = t_cf
    end
    return ts_mosek, ts_cf
end


# ******************************************************************************
# ******************************************************************************
# II. Solve time comparisons
# ******************************************************************************
# ******************************************************************************
ms = 10 .^ range(2, 5, 10) .|> x -> round(Int, x)
ts_mosek, ts_cf = run_trials(ms)


time_plt = plot(
    ms,
    ts_mosek,
    label="Mosek",
    xlabel="Number of nodes",
    ylabel="Solve time (s)",
    yscale=:log10,
    xscale=:log10,
    legend=:bottomright,
    minorgrid=true,
    yticks=10. .^ (-3:3),
    xticks=10. .^ (2:5),
    ylims=(1e-3, 1e2),
    linewidth=3,
    color=:blue,
    linestyle=:dash,
    legendfontsize=12,
    tickfontsize=12,
    guidefontsize=12,
    legendtitlefontsize=12,
    dpi=300,
)
plot!(time_plt, ms, ts_cf, label="ConvexFlows.jl", color=:black, linewidth=3)
savefig(time_plt, joinpath(FIGPATH, "cfmm-time.pdf"))
