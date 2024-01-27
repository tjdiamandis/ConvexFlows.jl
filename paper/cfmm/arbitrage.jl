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

# ******************************************************************************
# ******************************************************************************
# I. Convergence of dual gap and objective value over iterations
# ******************************************************************************
# ******************************************************************************
n = 100
m = round(Int, n^2 / 4)
cfmms = build_pools(m, n; swap_only=false)

# Define objective function
min_price = 1e-2
max_price = 1.0
Random.seed!(1)
c = rand(n) .* (max_price - min_price) .+ min_price
Uy = LinearNonnegative(c)

Vis_zero = true
time, pstar = run_trial_jump(
    cfmms, 
    c; 
    Vis_zero=Vis_zero, 
    optimizer=() -> Mosek.Optimizer()
)

# Solve with ConvexFlows.jl
s = Solver(
    flow_objective=Uy,
    edges=cfmms,
    n=n,
)

ITERS_NEEDED = 21
dual_gaps = zeros(ITERS_NEEDED+1)
obj_diffs = zeros(ITERS_NEEDED+1)
feas_violations = zeros(ITERS_NEEDED+1)
dual_gaps[1] = NaN
feas_violations[1] = NaN
obj_diffs[1] = (pstar - U(Uy, zeros(n))) / max(abs(pstar), abs(U(Uy, zeros(n))))

# Hack to get iterations out of LBFGSB.jl
for max_iter in 1:ITERS_NEEDED
    CF.solve!(s, verbose=false, factr=1e1, memory=5, max_iter=max_iter)
    # TODO: should we project here??
    p_iter = U(Uy, max.(s.y, 0.0))
    rp = norm(max.(-s.y, 0.0)) / norm(s.y)
    d_iter = Ubar(Uy, s.ν) + sum(dot(s.xs[i], s.ν[cfmms[i].Ai]) for i in 1:m)
    dual_gap = (d_iter - p_iter) / max(abs(d_iter), abs(p_iter))
    
    dual_gaps[max_iter+1] = dual_gap
    obj_diffs[max_iter+1] = (pstar - p_iter)/max(abs(pstar), abs(p_iter))
    feas_violations[max_iter+1] = rp
end

iter_conv_plt = plot(
    0:ITERS_NEEDED,
    abs.(obj_diffs),
    yaxis=:log10,
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
    abs.(dual_gaps), 
    label="Duality gap",
    color=:black,
    linewidth=3,
)
plot!(
    iter_conv_plt, 
    0:ITERS_NEEDED, 
    feas_violations .+eps(), 
    label="Feasibility violation",
    color=:red,
    linewidth=3,
)
plot!(
    iter_conv_plt, 
    0:ITERS_NEEDED, 
    sqrt(eps())*ones(ITERS_NEEDED+1),
    label=L"\sqrt{\texttt{eps}}",
    color=:black,
    linewidth=2,
    linestyle=:dash,
)
savefig(iter_conv_plt, joinpath(FIGPATH, "cfmm-iter-conv.pdf"))


# ******************************************************************************
# With Vᵢ's
# ******************************************************************************
Vis_zero = false
time, pstar = run_trial_jump(
    cfmms, 
    c; 
    Vis_zero=Vis_zero, 
    optimizer=() -> Mosek.Optimizer()
)

# Solve with ConvexFlows.jl
Vis = [NondecreasingQuadratic(length(cfmm)) for cfmm in cfmms]
s_vi = Solver(
    flow_objective=Uy,
    edge_objectives=Vis,
    edges=cfmms,
    n=n
)

ITERS_NEEDED_VI = 893
trials = 10:10:ITERS_NEEDED_VI
TRIAL_LENGTH = length(trials)
# ITERS_NEEDED_VI = 100
dual_gaps_vi = zeros(TRIAL_LENGTH+1)
obj_diffs_vi = zeros(TRIAL_LENGTH+1)
feas_violations_vi = zeros(TRIAL_LENGTH+1)
dual_gaps_vi[1] = NaN
feas_violations_vi[1] = NaN
obj_diffs_vi[1] = (pstar - U(Uy, zeros(n))) / max(abs(pstar), abs(U(Uy, zeros(n))))

# Hack to get iterations out of LBFGSB.jl
for (ind, max_iter) in enumerate(trials)
    GC.gc()
    CF.solve!(s_vi, verbose=false, factr=1e1, memory=5, max_iter=max_iter)
    p_iter = U(Uy, s_vi.y) + sum(U(Vis[i], s_vi.xs[i]) for i in 1:m)
    rp = norm(max.(-s_vi.y, 0.0)) / norm(s_vi.y)
    d_iter = Ubar(Uy, s_vi.ν) + 
        sum(
            Ubar(Vis[i], s_vi.ηts[i]) + 
            dot(s_vi.ν[cfmms[i].Ai] + s_vi.ηts[i], s_vi.xs[i]) 
            for i in 1:m
        )
    dual_gap = (d_iter - p_iter) / max(abs(d_iter), abs(p_iter))
    
    dual_gaps_vi[ind+1] = dual_gap
    obj_diffs_vi[ind+1] = (pstar_vi - p_iter)/max(abs(pstar_vi), abs(p_iter))
    feas_violations_vi[ind+1] = rp

    max_iter % 50 == 0 && println("Iteration $max_iter")
end

iter_conv_plt_vi = plot(
    vcat([0], trials),
    abs.(obj_diffs_vi),
    yaxis=:log10,
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
    iter_conv_plt_vi, 
    vcat([0], trials), 
    abs.(dual_gaps_vi), 
    label="Duality gap",
    color=:black,
    linewidth=3,
)
plot!(
    iter_conv_plt_vi, 
    collect(vcat([0], trials)), 
    feas_violations_vi, 
    label="Feasibility violation",
    color=:red,
    linewidth=3,
)
plot!(
    iter_conv_plt_vi, 
    vcat([0], trials), 
    sqrt(eps())*ones(TRIAL_LENGTH+1),
    label=L"\sqrt{\texttt{eps}}",
    color=:black,
    linewidth=2,
    linestyle=:dash,
)
savefig(iter_conv_plt_vi, joinpath(FIGPATH, "cfmm-vi-iter-conv.pdf"))
