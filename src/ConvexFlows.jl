module ConvexFlows

using LinearAlgebra, SparseArrays, StaticArrays
using LBFGSB
using Printf

include("utils.jl")

include("bfgs/types.jl")
include("bfgs/bfgs.jl")

include("edges.jl")
include("objectives.jl")


include("solver.jl")
include("solver_bfgs.jl")

export Objective, Ubar, grad_Ubar!, lower_limit, upper_limit
export Edge, find_arb!
export Solver, solve!

export BFGSSolver, BFGSOptions

export problem

# Objectives
export NonpositiveQuadratic, U, ∇U, Ubar, ∇Ubar

# Edges
export Edge, EdgeGain, EdgeClosedForm

end
