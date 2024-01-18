struct Solver{
    T <: AbstractFloat,
    E <: Edge{T},
    V <: Vector{T}
}
    flow_objective::Objective
    edge_objectives::Union{Vector{Objective}, Nothing}
    edges::Vector{E}
    y::V
    xs::Vector{V}
    ν::V
    ηs::Union{Vector{V}, Nothing}
end

function Solver(;
    flow_objective::Objective,
    edge_objectives::Union{Vector{Objective}, Nothing}=nothing,
    edges::Vector{E},
    n::Int,
    m::Int,
) where {T, E <: Edge{T}}
    # dimension checks
    m != length(edges) && ArgumentError("m must be the number of edges")
    n != length(vcat([e.Ai for e in edges]...)) && ArgumentError("n must be the number of nodes")
    !isnothing(edge_objectives) && length(edge_objectives) != m && ArgumentError("edge_objectives must be of length m")

    # setup solver variables for primal and dual
    y = zeros(T, n)
    xs = [zeros(T, length(e.Ai)) for e in edges]
    ν = zeros(T, n)
    ηs = [zeros(T, length(e.Ai)) for e in edges]
    
    return Solver(
        flow_objective,
        edge_objectives,
        edges,
        y,
        xs,
        ν,
        ηs
    )
end

function find_arb!(s::Solver{T}) where T
    # if isnothing(s.ηs)
    #     for i in 1:length(s.xs)
    #         @views s.ηs[i] .= s.ν[s.edges[i].Ai]
    #     end
    # end
    Threads.@threads for i in 1:length(s.xs)
    # for i in 1:length(s.xs)
        if isnothing(s.edge_objectives)
            s.ηs[i] .= s.ν[s.edges[i].Ai]
        end
        find_arb!(s.xs[i], s.edges[i], s.ηs[i])
    end
end

function solve!(
    s::Solver{T}; 
    ν0=nothing, 
    η0=nothing,
    verbose=false,
    m=5,
    factr=1e1,
    pgtol=1e-5,
    max_fun=15_000,
    max_iter=10_000,
) where T

    # check arguments
    if !isnothing(η0) && length(η0) != length(s.edges)
        ArgumentError("η0 must be of length m")
    elseif !isnothing(η0) && isnothing(s.edge_objectives)
        ArgumentError("solver does not have edge objectives")
    end

    optimizer = L_BFGS_B(length(s.ν), 17)

    bounds = zeros(3, length(s.ν))
    bounds[1, :] .= 2
    bounds[2, :] .= lower_limit(s.flow_objective)
    bounds[3, :] .= upper_limit(s.flow_objective)

    if isnothing(ν0)
        s.ν .= one(T) / length(s.ν)
    else
        s.ν .= ν0
    end

    # Objective function
    function fn(ν::Vector{T}) where T
        # only update if find_arb! has not been called yet
        if any(ν .!= s.ν)
            s.ν .= ν
            find_arb!(s)
        end

        acc = zero(T)

        for (x, e) in zip(s.xs, s.edges)
            @views acc += dot(x, ν[e.Ai])
        end

        return Ubar(s.flow_objective, ν) + acc
    end

    function grad!(g, ν::Vector{T}) where T
        g .= zero(T)

        if any(ν .!= s.ν)
            s.ν .= ν
            find_arb!(s)
        end
        grad_Ubar!(g, s.flow_objective, ν)

        for (x, e) in zip(s.xs, s.edges)
            @views g[e.Ai] .+= x
        end

        return nothing
    end

    # ----- Solve with L-BFGS-B -----
    # initial find_arb call to set initial values
    find_arb!(s)

    # call LBFGSB solver
    _, ν = optimizer(
        fn,
        grad!,
        s.ν,
        bounds,
        m=m,
        factr=factr,
        pgtol=pgtol,
        iprint=verbose ? 1 : -1,
        maxfun=max_fun,
        maxiter=max_iter
    )

    # Do final find_arb to get final values of edge flows
    s.ν .= ν
    find_arb!(s)
    netflows!(s)
    
    # TODO: print objective, primal feasibility, etc
end

function netflows!(s::Solver{T}) where T
    s.y .= zero(T)
    for (x, e) in zip(s.xs, s.edges)
        @views s.y[e.Ai] .+= x
    end
    return nothing
end