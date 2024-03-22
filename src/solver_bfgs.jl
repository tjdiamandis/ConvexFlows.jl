struct ConvexFlowProblemTwoNode{
    T <: AbstractFloat,
    V <: Vector{T},
    OV <: Objective,
}
    U::OV
    edges::Vector{<: Edge}
    y::V
    xs::Vector{V}
    ν::V
    n::Int
    m::Int
end

# TODO: In all these places, should edges be typed?
function problem(;
    obj::Objective,
    edges::Vector{<: Edge}
)
    n = length(obj)
    m = length(edges)

    # TODO: allow for abstract floats?
    y = zeros(n)
    xs = [zeros(2) for e in edges]
    ν = zeros(n)
    return ConvexFlowProblemTwoNode(obj, edges, y, xs, ν, n, m)
end

function solve!(
    problem::ConvexFlowProblemTwoNode;
    options::Union{BFGSOptions,Nothing}=nothing,
    method=:bfgs
)
    n, m = problem.n, problem.m
    options = isnothing(options) ? BFGSOptions() : options
    
    problem.ν .= 1.0


    function f∇f!(
        g::Vector{T},
        ν::Vector{T},
        xs::Vector{V},
        obj::Objective,
        edges::Vector{ <: Edge}
    ) where {T, V <: Vector{T}}
        
        acc = Ubar(obj, ν)
        ∇Ubar!(g, obj, ν)
    
    
        find_arb!(xs, ν, edges)
        for i in 1:length(edges)
            i1 = edges[i].Ai[1]
            i2 = edges[i].Ai[2]

            # TODO: do these allocate? need views?
            acc += xs[i][1]*ν[i1] + xs[i][2]*ν[i2]
            g[i1] += xs[i][1]
            g[i2] += xs[i][2]
        end
    
        return acc
    end
    f∇f!(g, x, p) = f∇f!(g, x, problem.xs, problem.U, problem.edges)

    solver = BFGSSolver(n; method=method)
    result = solve!(solver, f∇f!, nothing; options=options, x0=problem.ν)

    problem.ν .= result.x
    ∇Ubar!(problem.y, problem.U, problem.ν)
    problem.y .= -problem.y
    yhat = netflows(problem)
    pres = norm(problem.y - yhat)

    options.verbose && @printf("\nDual problem solve status:\n")
    display(result)
    options.verbose && @printf("Primal feasibility ||y - ∑Aᵢxᵢ||: %.4e\n", pres)
    return nothing
end


function netflows(prob::ConvexFlowProblemTwoNode{T}) where T
    ret = zeros(prob.n)
    for (x, e) in zip(prob.xs, prob.edges)
        ret[e.Ai[1]] += x[1]
        ret[e.Ai[2]] += x[2]
    end
    return ret
end