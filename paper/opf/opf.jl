using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Random, LinearAlgebra, SparseArrays
using BenchmarkTools
using Convex, MosekTools, SCS
using StatsBase, LogExpFunctions

Pkg.activate(joinpath(@__DIR__, "..", ".."))
using ConvexFlows
const CF = ConvexFlows

include("edges.jl")
include("objectives.jl")

n = 100
Random.seed!(1)
lines = TransmissionLine[]
for i in 1:n, j in i+1:n
    if rand() < 0.1
        bi = rand((1., 2., 3.))
        push!(lines, TransmissionLine(bi, [i, j]))
        push!(lines, TransmissionLine(bi, [j, i]))
    end
end
m = length(lines)
Adj = zeros(Int, n, n)
for i in 1:m
    Adj[lines[i].Ai...] = 1
end
Lap = Diagonal(sum(Adj, dims=2)[:]) - Adj
λ2 = eigvals(Lap, sortby=x->-x)[2]
(λ2 ≤ sqrt(eps()) || any(iszero, diag(Lap))) && error("Graph is not connected")

# Define objective function
d = rand((0.5, 1., 2.), n)
Uy = QuadraticPowerCost(d)
s = Solver(
    flow_objective=Uy,
    edges=lines,
    # edges=Edge[TransmissionLine(0., [1,2])],
    n=n,
)
CF.solve!(s, verbose=true, factr=1e1, memory=5)
pstar = U(Uy, s.y)
dstar = Ubar(Uy, s.ν)
dual_gap = (dstar - pstar) / max(abs(dstar), abs(pstar))


for i in 1:m
    e = lines[i]
    x = s.xs[i]
    !check_optimality(x, e, s.ν[e.Ai]) && @warn "Line flows $i are not optimal"
end

g = zeros(n)
grad_Ubar!(g, Uy, s.ν)
grad_norm = norm(g + s.y)