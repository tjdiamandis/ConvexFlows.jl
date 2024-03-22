abstract type Edge{T} end

@def add_generic_fields begin
    Ai::Vetcor{Int}
end
Base.length(e::Edge) = length(e.Ai)

function find_arb! end

# Edge with gain function
struct EdgeGain{T} <: Edge{T}
    Ai::Tuple{Int, Int}
    h::Function
    ub::T
end
function Edge(
    inds::Tuple{Int, Int},
    h::Function,
    ub::T,
) where T
    return EdgeGain{T}(inds, h, ub)
end

# Edge with closed form solution
struct EdgeClosedForm{T} <: Edge{T}
    Ai::Tuple{Int, Int}
    h::Function
    ub::T
    wstar::Function
end
function Edge(
    inds::Tuple{Int, Int};
    h::Function,
    ub::T,
    wstar::Function,
) where T
    return EdgeClosedForm{T}(inds, h, ub, wstar)
end


function find_arb!(
    xs::Vector{V},
    ν::Vector{T}, 
    edges::Vector{<: Edge}
) where {T, V <: Vector{T}}

    Threads.@threads for i in 1:length(edges)
        ind1 = edges[i].Ai[1]
        ind2 = edges[i].Ai[2]
        find_arb!(xs[i], edges[i], ν[ind1] / ν[ind2])
    end

    return nothing
end


function find_arb!(x::Vector{T}, e::EdgeClosedForm{T}, ratio::T) where T
    x[1] = -e.wstar(ratio)
    x[2] = e.h(-x[1])
    return nothing
end

# let x = (x₁, x₂) be the solution to h'(x₁) - η₁/η₂ = 0
function find_arb!(x::Vector{T}, e::EdgeGain{T}, ratio::T) where T
    # TODO: Newton's method or bisection
end