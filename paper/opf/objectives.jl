struct QuadraticPowerCost{T} <: Objective
    n::Int
    d::Vector{T}
end
QuadraticPowerCost(d::Vector{T}) where T = QuadraticPowerCost(length(d), d)


function U(obj::QuadraticPowerCost{T}, y) where T
    return -0.5*sum(x->abs2(max(x, zero(T))), obj.d - y)
end

function grad_U(obj::QuadraticPowerCost{T}, y) where T
    return obj.d - y
end

function CF.Ubar(obj::QuadraticPowerCost{T}, ν) where T
    return 0.5*sum(abs2, ν) - dot(obj.d, ν)
end

function CF.grad_Ubar!(g, obj::QuadraticPowerCost{T}, ν) where T
    @. g = ν - obj.d
    return nothing
end

CF.lower_limit(obj::QuadraticPowerCost{T}) where {T} = zeros(T, obj.n)
CF.upper_limit(obj::QuadraticPowerCost{T}) where {T} = convert(T, Inf) .+ zeros(T, obj.n)