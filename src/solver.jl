struct Solver{
    T <: AbstractFloat,
    V <: Vector{T},
    OV <: Objective,
}
    flow_objective::OV
    edge_objectives::Union{Vector{<: Objective}, Nothing}
    Vis_zero::Bool
    edges::Vector{<: Edge}
    y::V
    xs::Vector{V}
    ν::V
    ηts::Vector{V}                                  # used for cache if Vᵢ = 0
    arb_prices::Vector{V}
    μt::V
    n::Int
    m::Int
end

function Solver(;
    flow_objective::Objective,
    edge_objectives::Union{Vector{<:Objective}, Nothing}=nothing,
    edges::Vector{<: Edge},
    n::Int,
    # m::Int,
)
    # Need while using LBFGS.jl 
    T = Float64

    # dimension checks
    m = length(edges)
    n != length(vcat([e.Ai for e in edges]...)) && ArgumentError("n must be the number of nodes")
    !isnothing(edge_objectives) && length(edge_objectives) != m && ArgumentError("edge_objectives must be of length m")

    # setup solver variables for primal and dual
    y = zeros(T, n)
    xs = [zeros(T, length(e.Ai)) for e in edges]
    ν = zeros(T, n)
    ηs = [zeros(T, length(e.Ai)) for e in edges]
    arb_prices = [zeros(T, length(e.Ai)) for e in edges]
    Vis_zero = isnothing(edge_objectives) ? true : false
    μt = zeros(T, Vis_zero ? n : n + sum([length(e.Ai) for e in edges]))
    
    return Solver(
        flow_objective,
        edge_objectives,
        Vis_zero,
        edges,
        y,
        xs,
        ν,
        ηs,
        arb_prices,
        μt,
        n,
        m
    )
end

function find_arb!(s::Solver{T}) where T
    Threads.@threads for i in 1:length(s.xs)
        if s.Vis_zero
            s.arb_prices[i] .= s.ν[s.edges[i].Ai] 
        else
            s.arb_prices[i] .= s.ηts[i] .+ s.ν[s.edges[i].Ai]
        end
        find_arb!(s.xs[i], s.edges[i], s.arb_prices[i])
    end
end

function solve!(
    s::Solver{T}; 
    ν0=nothing, 
    η0=nothing,
    verbose=false,
    memory=5,
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

    nis = [length(e.Ai) for e in s.edges]

    # determine bounds
    # NOTE: zero is always LB since U, Vᵢ nondecreasing
    len_μ = s.Vis_zero ? s.n : s.n + sum(nis)
    bounds = zeros(3, len_μ)
    bounds[1, :] .= 2
    bounds[2, 1:s.n] .= max.(zero(T), lower_limit(s.flow_objective))
    bounds[3, 1:s.n] .= upper_limit(s.flow_objective)
    ind = s.n + 1
    if !s.Vis_zero
        for i in 1:s.m
            bounds[2, ind:ind+nis[i]-1] .= max.(zero(T), lower_limit(s.edge_objectives[i]))
            bounds[3, ind:ind+nis[i]-1] .= upper_limit(s.edge_objectives[i])
            ind += nis[i]
        end
    end

    # set initial values
    if isnothing(ν0)
        s.μt[1:s.n] .= bounds[2, 1:s.n] .+ ones(T, s.n)
    else
        s.μt[1:s.n] .= ν0
    end
    if !s.Vis_zero
        for i in 1:s.m
            s.ηts[i] .= s.μt[1:s.n][s.edges[i].Ai]
        end
    end

    # Objective function
    function fn(μ::Vector{T}) where T
        # only update if find_arb! has not been called yet
        # if any(ν .!= s.ν)
        #     s.ν .= ν
        #     find_arb!(s)
        # end

        # Load variables locally
        # TODO: make more efficient
        s.ν .= μ[1:s.n]
        if !s.Vis_zero
            ind = s.n + 1
            for i in 1:s.m
                s.ηts[i] .= μ[ind:ind+nis[i]-1]
                ind += nis[i]
            end
        end
        find_arb!(s)


        acc = zero(T)
        for i in 1:s.m
            if s.Vis_zero
                @views acc += dot(s.xs[i], s.ν[s.edges[i].Ai])
            else
                acc += Ubar(s.edge_objectives[i], s.ηts[i])
                @views acc += dot(s.xs[i], s.arb_prices[i])
            end
        end
        return Ubar(s.flow_objective, s.ν) + acc
    end

    function grad!(g, μ::Vector{T}) where T
        g .= zero(T)

        # if any(ν .!= s.ν)
        #     s.ν .= ν
        #     find_arb!(s)
        # end
        s.ν .= μ[1:s.n]
        if !s.Vis_zero
            ind = s.n + 1
            for i in 1:s.m
                s.ηts[i] .= μ[ind:ind+nis[i]-1]
                ind += nis[i]
            end
        end
        find_arb!(s)
        
        @views grad_Ubar!(g[1:s.n], s.flow_objective, s.ν)

        ind = s.n + 1
        for i in 1:s.m
            # add to ∇_ν
            @views g[s.edges[i].Ai] .+= s.xs[i]

            if !s.Vis_zero
                ni = length(s.edges[i])
                inds = ind:ind+ni-1
                
                # add to ∇_ηᵢ
                @views grad_Ubar!(g[inds], s.edge_objectives[i], s.ηts[i])
                @views g[inds] .+= s.xs[i]

                ind += ni
            end
        end

        return nothing
    end

    # ----- Solve with L-BFGS-B -----
    # initial find_arb call to set initial values
    find_arb!(s)

    # create and call LBFGSB solver
    optimizer = L_BFGS_B(len_μ, 17)
    _, μ = optimizer(
        fn,
        grad!,
        s.μt,
        bounds,
        m=memory,
        factr=factr,
        pgtol=pgtol,
        iprint=verbose ? 1 : -1,
        maxfun=max_fun,
        maxiter=max_iter
    )

    # Do final find_arb to get final values of edge flows
    s.ν .= μ[1:s.n]
    if !s.Vis_zero
        ind = s.n + 1
        for i in 1:s.m
            s.ηts[i] .= μ[ind:ind+nis[i]-1]
            ind += nis[i]
        end
    end
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