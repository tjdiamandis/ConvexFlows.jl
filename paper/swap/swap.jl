using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Random, LinearAlgebra
using BenchmarkTools
using JuMP, MosekTools

Pkg.activate(joinpath(@__DIR__, "..", ".."))
using ConvexFlows
const CF = ConvexFlows

Random.seed!(1)

include("cfmms.jl")
include("objectives.jl")


cfmms = [
    Uniswap([1e3, 1e4], 0.997, [1, 2]),
    Uniswap([1e3, 1e2], 0.997, [2, 3]),
    Uniswap([1e3, 2e4], 0.997, [1, 3])
]
n = maximum([maximum(cfmm.Ai) for cfmm in cfmms])
U = Swap(1, 2, 10.0, n)

s = Solver(
    flow_objective=U,
    edges=cfmms,
    n=n,
)
solve!(s, verbose=true, max_iter=1000, max_fun=1000)
netflows = s.y .|> x -> round(x, digits=2)
@show netflows
@show s.xs




Vis = [NondecreasingQuadratic(length(c)) for c in cfmms]
s2 = Solver(
    flow_objective=U,
    edge_objectives=Vis,
    edges=cfmms,
    n=n
)
solve!(s2, verbose=true, max_iter=1000, max_fun=1000)
netflows = s2.y .|> x -> round(x, digits=2)
@show netflows
@show s2.xs
netflows[2] - sum(norm(max.(-x, 0.0))^2 for x in s2.xs)


# Swap
function run_trial_mosek(cfmms)
    Rs = [cfmm.R for cfmm in cfmms]
    As = [cfmm.Ai for cfmm in cfmms]
    γs = [cfmm.γ for cfmm in cfmms]
    n_pools = length(cfmms)
    n_tokens = maximum([maximum(Ai) for Ai in As])
    ws = 0.5*ones(n_pools)

    # construct model
    routing = Model(Mosek.Optimizer)
    set_silent(routing)
    @variable(routing, Ψ[1:n_tokens])
    Δ = [@variable(routing, [1:2]) for _ in 1:n_pools]
    Λ = [@variable(routing, [1:2]) for _ in 1:n_pools]

    # Construct pools
    for i in 1:n_pools
        Ri = Rs[i]
        ϕRi = sqrt(Ri[1]*Ri[2])
        @constraint(routing, vcat(Ri + γs[i] * Δ[i] - Λ[i], ϕRi) in MOI.PowerCone(ws[i]))
        @constraint(routing, Δ[i] .≥ 0)
        @constraint(routing, Λ[i] .≥ 0)
    end

    # pool objectives
    @variable(routing, t[1:n_pools])
    x = [@variable(routing, [1:2]) for _ in 1:n_pools]
    for i in 1:n_pools
        @constraint(routing, [0.5; -2t[i]; x[i]] in MOI.RotatedSecondOrderCone(2+2))
        @constraint(routing, x[i] .≥ 0.0)
        @constraint(routing, x[i] .≥ -Λ[i] + Δ[i])
    end


    # net trade constraint
    net_trade = zeros(AffExpr, n_tokens)
    for i in 1:n_pools
        @. net_trade[As[i]] += Λ[i] - Δ[i]
    end
    @constraint(routing, Ψ .== net_trade)

    # Objective: arbitrage
    # TODO: change this to swap!
    # @constraint(routing, Ψ .>= 0)
    # c = rand(n_tokens)
    # @objective(routing, Max, sum(c .* Ψ))
    @constraint(routing, Ψ[1] == -10.0)
    @constraint(routing, Ψ[3] == 0.0)

    @objective(routing, Max, Ψ[2] + sum(t))

    GC.gc()
    optimize!(routing)
    time = solve_time(routing)
    status = termination_status(routing)
    status != MOI.OPTIMAL && @info "\t\tMosek termination status: $status"
    Ψv = round.(Int, value.(Ψ))
    @show value.(t)
    
    return time, Ψv, [value.(Λ[i]) - value.(Δ[i]) for i in 1:n_pools]
end

_, ym, xms = run_trial_mosek(cfmms)
@show ym
@show xms
ym[2] - sum(norm(max.(-x, 0.0))^2 for x in xms)