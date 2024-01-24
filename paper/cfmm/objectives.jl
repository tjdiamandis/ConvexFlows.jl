# U(y) = cᵀy - I(y ≥ 0)
struct LinearNonnegative{T} <: Objective 
    n::Int
    c::Vector{T}
end
function LinearNonnegative(c::Vector{T}) where T
    all(c .> 0) || throw(ArgumentError("all elements must be strictly positive"))
    n = length(c)
    return LinearNonnegative{Float64}(n, c)
end

function U(obj::LinearNonnegative{T}, y) where T
    return dot(obj.c, y)
end

function grad_U(obj::ArbitragePenalty{T}, y) where T
    return obj.c
end

# Assumes that ν - c ≥ 0
function CF.Ubar(obj::LinearNonnegative{T}, ν) where T
    return zero(T)
end

# Assumes that ν - c ≥ 0
function CF.grad_Ubar!(g, obj::LinearNonnegative{T}, ν) where T
    g .= zero(T)
    return nothing
end

# Add a small amount to the lower limit to avoid numerical issues
CF.lower_limit(obj::LinearNonnegative{T}) where {T} = obj.c .+ sqrt(eps(T))
CF.upper_limit(obj::LinearNonnegative{T}) where {T} = convert(T, Inf) .+ zeros(T, obj.n)



# U(y) = cᵀy - (¹/₂)∑ max(0, -yᵢ)²
struct ArbitragePenalty{T} <: Objective 
    n::Int
    c::Vector{T}
end
function ArbitragePenalty(c::Vector{T}) where T
    all(c .> 0) || throw(ArgumentError("all elements must be strictly positive"))
    n = length(c)
    return ArbitragePenalty{Float64}(n, c)
end

function U(obj::ArbitragePenalty{T}, y) where T
    return dot(obj.c, y) - 0.5 * sum(abs2, max.(-y, zero(T)))
end

function grad_U(obj::ArbitragePenalty{T}, y) where T
    return obj.c - max.(-y, zero(T))
end

# Assumes that ν - c ≥ 0
function CF.Ubar(obj::ArbitragePenalty{T}, ν) where T
    return 0.5 * sum(i -> abs2(ν[i] - obj.c[i]), 1:obj.n)
end

# Assumes that ν - c ≥ 0
function CF.grad_Ubar!(g, obj::ArbitragePenalty{T}, ν) where T
    @. g = ν - obj.c
end

# Add a small amount to the lower limit to avoid numerical issues
CF.lower_limit(obj::ArbitragePenalty{T}) where {T} = obj.c .+ sqrt(eps(T))
CF.upper_limit(obj::ArbitragePenalty{T}) where {T} = convert(T, Inf) .+ zeros(T, obj.n)



# Vᵢ(x) = -(¹/₂)∑ max(0, -xᵢ)²
struct NondecreasingQuadratic{T} <: Objective 
    n::Int
end
NondecreasingQuadratic(n::Int) = NondecreasingQuadratic{Float64}(n)

function U(::NondecreasingQuadratic{T}, x) where T
    return -0.5 * sum(w -> abs2(max(-w, 0)), x)
end

function grad_U(::NondecreasingQuadratic{T}, x) where T
    return -max(-x, 0)
end

# Could assume this is non-Inf because of the lower-limit
function CF.Ubar(::NondecreasingQuadratic{T}, η) where T
    any(η .< 0) && return convert(T, Inf)
    return 0.5*sum(abs2, η)
end

function CF.grad_Ubar!(g, ::NondecreasingQuadratic{T}, η) where T
    any(η .< 0) && return convert(T, Inf)
    g .= η
    return nothing
end

CF.lower_limit(obj::NondecreasingQuadratic{T}) where {T} = zeros(T, obj.n) .+ sqrt(eps(T))
CF.upper_limit(obj::NondecreasingQuadratic{T}) where {T} = convert(T, Inf) .+ zeros(T, obj.n)
