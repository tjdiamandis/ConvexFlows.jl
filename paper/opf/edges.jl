# ----- TransmissionLine edge ----
struct TransmissionLine{T} <: CF.Edge{T}
    b::T
    α::T
    β::T
    p_min::T
    p_max::T
    flow_max::T
    Ai::Vector{Int}
end
function TransmissionLine(b::T, Ai::Vector{Int}) where T <: AbstractFloat
    α, β = T(16), T(1/4)
    p_max = one(T)
    p_min = 3 - α * β * logistic(β * b)
    flow_max = 3b -  α * (log1pexp(β * b) - log(2)) 
    return TransmissionLine(b, α, β, p_min, p_max, flow_max, Ai)
end

function gain(w::T, e::TransmissionLine{T}) where T
    α, β, b = e.α, e.β, e.b
    w = w > b ? b : w
    return w - (α * (log1pexp(β * w) - log(2))  - 2w)
end

function d_gain(w::T, e::TransmissionLine{T}) where T
    α, β, b = e.α, e.β, e.b
    w = w > b ? b : w
    return 3 - α * β * logistic(β * w)
end

function CF.find_arb!(x::Vector{T}, e::TransmissionLine{T}, η::Vector{T}) where T
    η1, η2 = η[1], η[2]

    if iszero(η[2]) || e.p_max ≤ η1/η2
        x .= zero(T)
    elseif e.p_min ≥ η1/η2
        x[1] = -e.b
        x[2] = e.flow_max
    else 
        # assumes α*β = 4 (see paper)
        x[1] = -1/e.β * log((3η2 - η1)/(η2 + η1))
        x[2] = gain(-x[1], e)
    end

    # Alternative way that doesn't "short-circuit" by checking price
    # x[1] = clamp(1/e.β * log((3η2 - η1)/(η2 + η1)), zero(T), e.b)
    # x[2] = gain(x[1], e)
    return nothing
end

function check_optimality(x::Vector{T}, e::TransmissionLine{T}, η::Vector{T}; tol=sqrt(eps())) where T
    η1, η2 = η[1], η[2]
    p = η1/η2
    
    isapprox(p, d_gain(-x[1], e), atol=tol) && return true
    p ≥ d_gain(zero(T), e) && iszero(x[1]) && return true
    p ≤ d_gain(e.b, e) && isapprox(-x[1], e.b, atol=tol) && return true
    return false

end
