abstract type Objective end

@doc raw"""
    U(obj::Objective, y)

Evaluates the net flow utility function `objective` at `y`.
"""
function U end

@doc raw"""
    grad_U(obj::Objective, y)
Returns the gradient of the net flow utility function `objective` at `y`.
"""
function grad_U end

@doc raw"""
    Ubar(obj::Objective, ν)

Evaluates the 'conjugate' of the net flow utility function `objective` at `ν`.
Specifically,
```math
    \bar U(\nu) = \sup_y \left(U(y) - \nu^T y \right).
```
"""
function Ubar end

@doc raw"""
    grad!(g, obj::Objective, ν)

Computes the gradient of [`Ubar(obj, ν)`](@ref) at ν.
"""
function grad_Ubar! end

@doc raw"""
    lower_limit(obj)

Componentwise lower bound on argument `ν` for objective [`Ubar`](@ref).  
Returns a vector with length `length(ν)` (number of nodes).
"""
function lower_limit end

@doc raw"""
    upper_limit(obj)

Componentwise upper bound on argument `ν` for objective [`Ubar`](@ref).  
Returns a vector with length `length(ν)` (number of nodes).
"""
function upper_limit end


# quadratic cost: u(y) = -0.5*(-y + b)₊²
struct NonpositiveQuadratic{T} <: Objective
    n::Int
    b::Vector{T}
end
NonpositiveQuadratic(b::T) where T = NonpositiveQuadratic(length(b), b)

Base.length(obj::NonpositiveQuadratic) = obj.n

function U(obj::NonpositiveQuadratic{T}, y) where T
    return -0.5*sum(x->abs2(max(x, zero(T))), obj.d - y)
end

function ∇U(obj::NonpositiveQuadratic{T}, y) where T
    return obj.b - y
end

function Ubar(obj::NonpositiveQuadratic{T}, ν) where T
    return 0.5*sum(abs2, ν) - dot(obj.b, ν)
end

function ∇Ubar!(g, obj::NonpositiveQuadratic{T}, ν) where T
    @. g = ν - obj.b
    return nothing
end

# TODO: Linear, nonnegative quadratic