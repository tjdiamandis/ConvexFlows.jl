using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Random, LinearAlgebra
using BenchmarkTools
using JuMP, MosekTools, SCS
using StatsBase

Pkg.activate(joinpath(@__DIR__, "..", ".."))
using ConvexFlows
const CF = ConvexFlows

include("cfmms.jl")
include("objectives.jl")
include("jump.jl")
include("utils.jl")

function build_pools(n_pools, n_tokens; rseed=1)
    cfmms = Vector{Edge}()
    Random.seed!(rseed)
    for i in 1:n_pools
        rn = rand()
        γ = 0.997
        if rn < 0.4
            Ri = 100 * rand(2) .+ 100
            Ai = sample(collect(1:n_tokens), 2, replace=false)
            push!(cfmms, Uniswap(Ri, γ, Ai))
        elseif rn < 0.8
            Ri = 100 * rand(2) .+ 100
            Ai = sample(collect(1:n_tokens), 2, replace=false)
            w = 0.8
            push!(cfmms, Balancer(Ri, γ, Ai, w))
        else
            Ri = 100 * rand(3) .+ 100
            Ai = sample(collect(1:n_tokens), 3, replace=false)
            push!(cfmms, BalancerThreePool(Ri, γ, Ai))
        end
            
    end
    return cfmms
end


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

    trial = @timed begin solve!(s, verbose=verbose, factr=factr, memory=memory) end
    time = trial.time
    
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

ms = round.(Int, 10 .^ range(2, 5, 20))
m = ms[10]
n = round(Int, 2*sqrt(m))
cfmms = build_pools(m, n)

min_price = 1e-2
max_price = 1.0
Random.seed!(1)
c = rand(n) .* (max_price - min_price) .+ min_price

Vis_zero = true
time, p_jump = run_trial_jump(
    cfmms, 
    c; 
    Vis_zero=Vis_zero, 
    optimizer=() -> Mosek.Optimizer()
)


Uy = LinearNonnegative(c)
time, p_flow, dual_gap, Uy_violation = run_trial_flows(Uy=Uy, cfmms=cfmms, verbose=true)
rel_diff = (p_jump - p_flow) / max(abs(p_flow), abs(p_jump))





m = 1_000
n = round(Int, sqrt(m))
cfmms = build_pools(m, n)

min_price = 1e-2
max_price = 1.0
Random.seed!(1)
c = rand(n) .* (max_price - min_price) .+ min_price

Vis_zero = false
time, p_jump = run_trial_jump(
    cfmms, 
    c; 
    Vis_zero=Vis_zero, 
    optimizer=() -> Mosek.Optimizer()
)


Uy = ArbitragePenalty(c)
Vis = [NondecreasingQuadratic(length(cfmm)) for cfmm in cfmms]
time, p_flow, dual_gap, Uy_violation = 
    run_trial_flows(Uy=Uy,
        Vis=Vis,
        cfmms=cfmms,
        verbose=true, 
        factr=1e-7,
        memory=10
    )
rel_diff = (p_flow - p_jump) / max(abs(p_flow), abs(p_jump))
