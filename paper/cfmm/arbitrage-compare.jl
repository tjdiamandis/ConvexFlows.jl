using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Random, LinearAlgebra
using BenchmarkTools
using JuMP, MosekTools, SCS
using StatsBase
using Plots, LaTeXStrings
using JLD2

Pkg.activate(joinpath(@__DIR__, "..", ".."))
using ConvexFlows
const CF = ConvexFlows

const FIGPATH = joinpath(@__DIR__, "..", "figures")
const SAVEPATH = joinpath(@__DIR__, "..", "data")
const SAVEFILE = joinpath(SAVEPATH, "arbitrage.jld2")

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


function run_trial(m; rseed=1, verbose=false)
    n = round(Int, 2*sqrt(m))
    cfmms = build_pools(m, n; swap_only=true, rseed=rseed)
    verbose && @info "  Finished building graph..."
    
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
    verbose && @info "  Finished running Mosek..."
    
    t_cf, p_cf, gap_cf, rp_cf = 
        run_trial_flows(
            Uy=Uy,
            cfmms=cfmms,
            verbose=false,
        )
    verbose && @info "  Finished running ConvexFlows..."
    
    return p_mosek, p_cf, gap_cf, rp_cf, t_mosek, t_cf
end

function run_trials(ms; trials=10, verbose=false)
    #compile
    run_trial(50)

    ts_mosek = zeros(trials, length(ms))
    ts_cf = zeros(trials, length(ms))
    for (i, m) in enumerate(ms)
        @info "Starting trial for m=$m..."
        for t in 1:trials
            p_mosek, p_cf, gap_cf, rp_cf, t_mosek, t_cf = run_trial(m; rseed=t, verbose=verbose)
            rel_obj_diff = (p_mosek - p_cf) / max(abs(p_mosek), abs(p_cf))
            verbose && @info "  reldiff:  $rel_obj_diff"
            rel_obj_diff > 1e-3 && @warn "  Possible incorrect solution!"
            verbose && @info "  dualgap:  $gap_cf"
            verbose && @info "  rp norm:  $rp_cf"
            gap_cf > 1e-3 && @warn "  ConvexFlows didn't produce a solution!"
            ts_mosek[t, i] = t_mosek
            ts_cf[t, i] = t_cf
        end
        @info "-- Finished! --"
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
save(SAVEFILE,
    "ms", ms,
    "ts_mosek", ts_mosek,
    "ts_cf", ts_cf,
)

# ******************************************************************************
# Plot solve times
# ******************************************************************************
ms, ts_mosek, ts_cf = load(SAVEFILE, "ms", "ts_mosek", "ts_cf")

ts_med_mosek = median(ts_mosek, dims=1) |> vec
ts_med_cf = median(ts_cf, dims=1) |> vec
q75_mosek = [quantile(ts_mosek[:,i], 0.75) for i in 1:length(ms)] |> vec
q25_mosek = [quantile(ts_mosek[:,i], 0.25) for i in 1:length(ms)] |> vec
q75_cf = [quantile(ts_cf[:,i], 0.75) for i in 1:length(ms)] |> vec
q25_cf = [quantile(ts_cf[:,i], 0.25) for i in 1:length(ms)] |> vec

max_mosek = maximum(ts_mosek, dims=1) |> vec
max_cf = maximum(ts_cf, dims=1) |> vec
for (i, m) in enumerate(ms)
    if max_mosek[i] > 5*ts_med_mosek[i]
        println("Trial: n = $m")
        println("  max mosek: $(round(max_mosek[i], digits=3))")
        println("  med mosek: $(round(ts_med_mosek[i], digits=3))")
    end
    if max_cf[i] > 5*ts_med_cf[i]
        println("  max cf: $(round(max_cf[i], digits=3))")
        println("  med cf: $(round(ts_med_cf[i], digits=3))")
    end
end

time_plt = plot(
    ms,
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
plot!(
    time_plt,
    ms,
    max_mosek,
    label=nothing,
    color=:blue,
    seriestype=:scatter,
    markersize=3,
)
plot!(
    time_plt,
    ms,
    ts_med_cf,
    ribbon=(ts_med_cf .- q25_cf, q75_cf .- ts_med_cf),
    fillalpha=0.5,
    label="ConvexFlows.jl",
    color=:black,
    linewidth=3
)
plot!(
    time_plt,
    ms,
    max_cf,
    label=nothing,
    color=:black,
    seriestype=:scatter,
    markersize=3,
)
savefig(time_plt, joinpath(FIGPATH, "cfmm-time.pdf"))
