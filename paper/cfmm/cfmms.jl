abstract type CFMM{T} <: Edge{T} end

# ----- Uniswap edge -----
struct Uniswap{T} <: CFMM{T}
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

function valid_trade(cfmm::Uniswap{T}, Δ::Vector{T}, Λ::Vector{T}) where T
    R, γ = cfmm.R, cfmm.γ
    ϕR = sqrt(R[1]*R[2])
    Rp = @. R + γ * Δ - Λ
    return all(Δ .≥ 0) && all(Λ .≥ 0) && all(Rp .≥ 0) && sqrt(Rp[1]*Rp[2]) ≥ ϕR - sqrt(eps(T))
end

function price(::Uniswap{T}, Rp::Vector{T}) where T
    return [Rp[2]/Rp[1], 1.0]
end

# checks how far η is from N(xstar)
function suboptimality(cfmm::CFMM{T}, Rp::Vector{T}, η::Vector{T}) where T
    ηp = price(cfmm, Rp)
    η_normalized = η ./ η[end]
    ηproj = clamp.(cfmm.γ .* ηp, η_normalized, ηp)
    return norm(ηproj - ηp)^2
end


# ----- Balancer swap pool -----
struct Balancer{T} <: CFMM{T}
    R::Vector{T}
    γ::T
    Ai::Vector{Int}
    w::T

    function Balancer(R::Vector{T}, γ::T, Ai::Vector{Int}, w::T) where T <: AbstractFloat
        length(R) != 2 && ArgumentError("R must be of length 2")
        length(Ai) != 2 && ArgumentError("Ai must be of length 2")
        !(w > 0 && w < 1) && ArgumentError("w must be in (0, 1)")

        return new{T}(R, γ, Ai, w)
    end
end

# Solves the maximum arbitrage problem for the two-coin geometric mean case.
# Assumes that v > 0 and w > 0.
function CF.find_arb!(x::Vector{T}, e::Balancer{T}, η::Vector{T}) where T
    # See App. A of "An Analysis of Uniswap Markets"
    @inline geom_arb_δ(m, r1, r2, η, γ) = max((γ*m*η*r1*r2^η)^(1/(η+1)) - r2, 0.0)/γ
    @inline geom_arb_λ(m, r1, r2, η, γ) = max(r1 - ((r2*r1^(1/η))/(η*γ*m))^(η/(1+η)), 0.0)

    R, γ, w = e.R, e.γ, e.w
    ratio = w/(1-w)
    x[1] = geom_arb_λ(η[1]/η[2], R[1], R[2], 1/ratio, γ) - geom_arb_δ(η[2]/η[1], R[2], R[1], ratio, γ)
    x[2] = geom_arb_λ(η[2]/η[1], R[2], R[1], ratio, γ) - geom_arb_δ(η[1]/η[2], R[1], R[2], 1/ratio, γ)
    return nothing
end

function valid_trade(cfmm::Balancer{T}, Δ::Vector{T}, Λ::Vector{T}) where T
    R, γ, w = cfmm.R, cfmm.γ, cfmm.w
    ϕR = R[1]^w*R[2]^(1-w)
    Rp = @. R + γ * Δ - Λ
    return all(Δ .≥ 0) && all(Λ .≥ 0) && all(Rp .≥ 0) && Rp[1]^w*Rp[2]^(1-w) ≥ ϕR - sqrt(eps(T))
end

function price(cfmm::Balancer{T}, Rp::Vector{T}) where T
    return [cfmm.w / (1 - cfmm.w) * Rp[2]/Rp[1], 1.0]
end


# ----- Balancer three pool -----
# equal weights (e.g., used for stable swaps)
struct BalancerThreePool{T} <: CFMM{T}
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

        k = geomean(R)
        @constraint(optimizer, [k; R + γ*Δ - Λ] ∈ MOI.GeometricMeanCone(3+1))

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

function valid_trade(cfmm::BalancerThreePool{T}, Δ::Vector{T}, Λ::Vector{T}) where T
    R, γ = cfmm.R, cfmm.γ
    ϕR = geomean(R)
    Rp = @. R + γ * Δ - Λ
    return all(Δ .≥ 0) && all(Λ .≥ 0) && all(Rp .≥ 0) && geomean(Rp) ≥ ϕR - sqrt(eps(T))
end

function price(::BalancerThreePool{T}, Rp::Vector{T}) where T
    return [Rp[3]/Rp[1], Rp[3]/Rp[2], 1.0]
end


function build_pools(n_pools, n_tokens; rseed=1, swap_only=true)
    cfmms = Vector{Edge}()
    Random.seed!(rseed)

    threshold = swap_only ? 0.5 : 0.4
    rns = rand(n_pools)
    for i in 1:n_pools
        rn = rns[i] 
        γ = 0.997
        if rn ≤ threshold
            Ri = 100 * rand(2) .+ 100
            Ai = sample(collect(1:n_tokens), 2, replace=false)
            push!(cfmms, Uniswap(Ri, γ, Ai))
        elseif rn ≤ 2*threshold
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