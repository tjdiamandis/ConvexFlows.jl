struct Swap{T} <: Objective
    j_in::Int
    j_out::Int
    Δ::T
    n::Int
end


function U(obj::Swap{T}, y)
    if !isapprox(y[obj.j_in], obj.Δ, atol=1e-6)
        return -Inf
    elseif any(y[i] < sqrt(eps()) for i in setdiff(1:obj.n, [obj.j_in, obj.j_out]))
        return -Inf
    else
        return y[obj.j_out]
    end
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


struct NondecreasingQuadratic{T} <: Objective 
    n::Int
end
NondecreasingQuadratic(n::Int) = NondecreasingQuadratic{Float64}(n)

function U(::NondecreasingQuadratic{T}, y) where T
    return sum(x -> abs2(max(-x, 0)), y)
end

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