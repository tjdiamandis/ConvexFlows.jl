
# ----- Uniswap edge -----
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


# ----- Balancer swap pool -----
struct Balancer{T} <: Edge{T}
    R::Vector{T}
    γ::T
    Ai::Vector{Int}
    w::Vector{T}

    function Balancer(R::Vector{T}, γ::T, Ai::Vector{Int}, w::Vector{T}) where T <: AbstractFloat
        length(R) != 2 && ArgumentError("R must be of length 2")
        length(Ai) != 2 && ArgumentError("Ai must be of length 2")
        !isapprox(sum(w), atol=1e-8) && ArgumentError("weights must sum to 1")

        return new{T}(R, γ, Ai, w)
    end
end

# Solves the maximum arbitrage problem for the two-coin geometric mean case.
# Assumes that v > 0 and w > 0.
function CF.find_arb!(x::Vector{T}, e::Balancer{T}, η::Vector{T}) where T
    # See App. A of "An Analysis of Uniswap Markets"
    @inline geom_arb_δ(m, r1, r2, η, γ) = max((γ*m*η*r1*r2^η)^(1/(η+1)) - r2, 0)/γ
    @inline geom_arb_λ(m, r1, r2, η, γ) = max(r1 - ((r2*r1^(1/η))/(η*γ*m))^(η/(1+η)), 0)

    R, γ, w = e.R, e.γ, e.w
    ratio = w[1]/w[2]
    x[1] = geom_arb_λ(η[1]/η[2], R[1], R[2], ratio, γ) - geom_arb_δ(η[2]/η[1], R[2], R[1], ratio, γ)
    x[2] = geom_arb_λ(η[2]/η[1], R[2], R[1], 1/ratio, γ) - geom_arb_δ(η[1]/η[2], R[1], R[2], 1/ratio, γ)
    return nothing
end


# ----- Balancer three pool -----
# equal weights (e.g., used for stable swaps)
struct BalancerThreePool{T} <: Edge{T}
    R::Vector{T}
    γ::T
    Ai::Vector{Int}
    optimizer

    function BalancerThreePool(R::Vector{T}, γ::T, Ai::Vector{Int}) where T <: AbstractFloat
        length(R) != 3 && ArgumentError("R must be of length 2")
        length(Ai) != 3 && ArgumentError("Ai must be of length 2")

        optimizer = Model(Mosek.Optimizer)
        set_silent(optimizer)
        @variable(optimizer, Δ[1:3] .≥ 0)
        @variable(optimizer, Λ[1:3] .≥ 0)
        @variable(optimizer, x[1:3])

        k = (R[1]*R[2]*R[3])^1/3
        @constraint(optimizer, [k; x] ∈ MOI.GeometricMeanCone(3+1))
        @constraint(optimizer, x == R + γ*Δ - Λ)

        return new{T}(R, γ, Ai, optimizer)
    end
end

function CF.find_arb!(x::Vector{T}, e::BalancerThreePool{T}, η::Vector{T}) where T
    # See App. A of "An Analysis of Uniswap Markets"
    m = e.optimizer
    @objective(m, Max, dot(η, m[:Λ] - m[:Δ]))
    optimize!(m)
    x .= value.(m[:Λ] - m[:Δ])
    return nothing
end