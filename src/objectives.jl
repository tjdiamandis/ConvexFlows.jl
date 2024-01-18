abstract type Objective end

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