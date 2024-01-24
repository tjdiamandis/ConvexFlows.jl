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

n_pools = 100
n_tokens = round(Int, 2 * sqrt(n_pools))
cfmms = Edge[]
Random.seed!(1)
for i in 1:n_pools
    rn = rand()
    γ = 0.997
    if rn < 0.4
        Ri = 100 * rand(2) .+ 100
        Ai = sample(collect(1:n_tokens), 2, replace=false)
        push!(cfmms, Uniswap(Ri, γ, Ai))
    elseif rn < 0.8
        Ri = 1000 * rand(2) .+ 1000
        Ai = sample(collect(1:n_tokens), 2, replace=false)
        w = 0.8
        push!(cfmms, Balancer(Ri, γ, Ai, w))
    # else
    #     Ri = 1000 * rand(3) .+ 1000
    #     Ai = sample(collect(1:n_tokens), 3, replace=false)
    #     push!(cfmms, BalancerThreePool(Ri, γ, Ai))
    end
        
end



cfmms = [
    Uniswap([1e3, 1e4], 0.997, [1, 2]),
    Uniswap([1e3, 1e2], 0.997, [2, 3]),
    Uniswap([1e3, 2e4], 0.997, [1, 3])
]

n = maximum([maximum(cfmm.Ai) for cfmm in cfmms])
m = length(cfmms)


min_price = 1e-3
max_price = 1e1
c = rand(n) .* (max_price - min_price) .+ min_price

Vis_zero = false
optimizer, Δs, Λs = build_mosek_arbitrage_model(cfmms, c, Vis_zero)
GC.gc()
optimize!(optimizer)
time = solve_time(optimizer)
status = termination_status(optimizer)
status != MOI.OPTIMAL && @info "\t\tMosek termination status: $status"
ystar = value.(optimizer[:y])
pstar = dot(c, ystar) - 0.5*sum(abs2, max.(-ystar, 0.0)) - (Vis_zero ? 0.0 : 0.5*sum(value.(optimizer[:t])))
pstar = objective_value(optimizer)
@show pstar
Δs_v = [value.(Δ) for Δ in Δs]
Λs_v = [value.(Λ) for Λ in Λs]
net_flow = zeros(n)
for (i, cfmm) in enumerate(cfmms)
    @. net_flow[cfmm.Ai] += Λs_v[i] - Δs_v[i]
end
norm(net_flow - ystar) / max(norm(net_flow), norm(ystar))

# Uy = Swap(1, 2, 10.0, n)
Uy = ArbitragePenalty(c)
# U(Uy, ystar)
# Uy = LinearNonnegative(c)

# μ = n*rand(n)
# σ = rand(n)
# Uy = Markowitz(μ, σ)

s = Solver(
    flow_objective=Uy,
    edges=cfmms,
    n=n,
)
@time solve!(s, verbose=true, factr=1e1, memory=10)
# netflows = s.y .|> x -> round(x, digits=2)
netflows = s.y
# @show netflows
objval = U(Uy, s.y)
@show objval
dot(s.y, c) - sum(abs2, max.(-s.y, 0.0))
# g = zeros(n)
# grad_Ubar!(g, Uy, s.y)
# objval_g = U(Uy, g)

Vis = [NondecreasingQuadratic(length(cfmm)) for cfmm in cfmms]
s2 = Solver(
    flow_objective=Uy,
    edge_objectives=Vis,
    edges=cfmms,
    n=n
)
solve!(s2, verbose=true, max_iter=1000, max_fun=1000, memory=5)
netflows = s2.y .|> x -> round(x, digits=2)
@show netflows
# @show s2.xs
# netflows[2] - sum(norm(max.(-x, 0.0))^2 for x in s2.xs)
dot(s2.y, c) - sum(abs2, max.(-s2.y, 0.0)) - sum(norm(max.(-x, 0.0))^2 for x in s2.xs)

# Swap

time = solve_time(routing)
status = termination_status(routing)
status != MOI.OPTIMAL && @info "\t\tMosek termination status: $status"
Ψv = round.(Int, value.(Ψ))
@show value.(t)

return time, Ψv, [value.(Λ[i]) - value.(Δ[i]) for i in 1:n_pools]

_, ym, xms = run_trial_mosek(cfmms)
@show ym
@show xms
ym[2] - sum(norm(max.(-x, 0.0))^2 for x in xms)