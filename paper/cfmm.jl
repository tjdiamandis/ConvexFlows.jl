using Pkg
Pkg.activate(@__DIR__)
using Random, LinearAlgebra
using BenchmarkTools
using JuMP, MosekTools

Pkg.activate(joinpath(@__DIR__, ".."))
using ConvexFlows
const CF = ConvexFlows

Random.seed!(1)


# Define Uniswap edge
struct Uniswap{T} <: Edge{T}
    R::Vector{T}
    γ::T
    Ai::Vector{Int}

    function Uniswap(R::Vector{T}, γ::T, Ai::Vector{Int}) where T <: AbstractFloat
        length(R) != 2 && ArgumentError("R must be of length 2")
        length(Ai) != 2 && ArgumentError("Ai must be of length 2")
        return new{T}(R, γ, Ai)
    end
end

# Solves the maximum arbitrage problem for the two-coin constant product case.
# Assumes that v > 0 and γ > 0.
function CF.find_arb!(x::Vector{T}, e::Uniswap{T}, η::Vector{T}) where T
    # See App. A of "An Analysis of Uniswap Markets"
    @inline prod_arb_δ(m, r, k, γ) = max(sqrt(γ*m*k) - r, 0.0)/γ
    @inline prod_arb_λ(m, r, k, γ) = max(r - sqrt(k/(m*γ)), 0.0)

    R, γ = e.R, e.γ
    k = R[1]*R[2]
    x[1] = prod_arb_λ(η[1]/η[2], R[1], k, γ) - prod_arb_δ(η[2]/η[1], R[1], k, γ)
    x[2] = prod_arb_λ(η[2]/η[1], R[2], k, γ) - prod_arb_δ(η[1]/η[2], R[2], k, γ)
    return nothing
end


# Define objective function
struct Swap{T} <: Objective
    j_in::Int
    j_out::Int
    Δ::T
    n::Int
end

function CF.Ubar(obj::Swap{T}, ν) where T
    if ν[obj.j_out] >= 1.0
        return ν[obj.j_in] * obj.Δ
    else
        return convert(T, Inf)
    end
end

function CF.grad_Ubar!(g, obj::Swap{T}, ν) where T
    if ν[obj.j_out] >= 1.0
        g .= zero(T)
        g[obj.j_in] = obj.Δ
    else
        g .= convert(T, Inf)
    end
    return nothing
end

function CF.lower_limit(obj::Swap{T}) where {T}
    ret = Vector{T}(undef, obj.n)
    fill!(ret, eps())
    ret[obj.j_out] = one(T) + eps()
    return ret
end
CF.upper_limit(obj::Swap{T}) where {T} = convert(T, Inf) .+ zeros(T, obj.n)

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


struct NondecreasingQuadratic{T} <: Objective 
    n::Int
end
NondecreasingQuadratic(n::Int) = NondecreasingQuadratic{Float64}(n)

function CF.Ubar(::NondecreasingQuadratic{T}, η) where T
    any(η .< 0) && return convert(T, Inf)
    return 0.5*sum(abs2, η)
end

function CF.grad_Ubar!(g, ::NondecreasingQuadratic{T}, η) where T
    any(η .< 0) && return convert(T, Inf)
    g .= η
    return nothing
end

CF.lower_limit(obj::NondecreasingQuadratic{T}) where {T} = zeros(T, obj.n)
CF.upper_limit(obj::NondecreasingQuadratic{T}) where {T} = convert(T, Inf) .+ zeros(T, obj.n)

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