using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Random, LinearAlgebra
using BenchmarkTools
using Convex, MosekTools, SCS
using StatsBase, LogExpFunctions

Pkg.activate(joinpath(@__DIR__, "..", ".."))
using ConvexFlows
const CF = ConvexFlows

include("edges.jl")

x = zeros(2)
η = rand(2)
b = 2.
e = TransmissionLine(b)

η = [10; 1.]
find_arb!(x, e, η)
check_optimality(x, e, η)

η = [1; 10.]
find_arb!(x, e, η)
check_optimality(x, e, η)

η = [1., 1.5]
find_arb!(x, e, η)
check_optimality(x, e, η)
