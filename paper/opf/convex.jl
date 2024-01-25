function Ai_mat(Ai::Vector{Int}, n::Int)
    m = length(Ai)
    A = spzeros(n, m)
    for i in 1:m
        A[Ai[i], i] = 1
    end
    return A
end


function build_convex_model(d, lines)
    n = length(d)
    m = length(lines)
    xs = Variable(2, m)
    y = Variable(n)

    cons = Constraint[
        y == sum(i -> Ai_mat(lines[i].Ai, n)*xs[:,i], 1:m),
    ]
    for i in 1:m
        α, β, b = lines[i].α, lines[i].β, lines[i].b
        x1, x2 = xs[1, i], xs[2, i]
        push!(cons, x1 ≤ 0)
        push!(cons, -b ≤ x1)
        push!(cons, 
            x2 ≤ -3x1 - α*( Convex.logsumexp([0.0; -β*x1]) - log(2) )
        )
    end

    obj = sum(i -> -0.5 * square(max(d[i] - y[i], 0.0)), 1:n)
    prob = maximize(obj, cons)
    return prob, y, xs
end

function Convex.solve!(
    problem::Problem{T},
    optimizer::Convex.MOI.ModelLike;
    check_vexity::Bool = true,
    verbose::Bool = true,
    warmstart::Bool = false,
    silent_solver::Bool = false,
) where {T}
    MOI = Convex.MOI
    MOIU = Convex.MOIU
    MOIB = Convex.MOIB

    if check_vexity
        vex = vexity(problem)
    end

    model = MOIB.full_bridge_optimizer(
        MOIU.CachingOptimizer(
            MOIU.UniversalFallback(MOIU.Model{T}()),
            optimizer,
        ),
        T,
    )

    id_to_variables,
    conic_constr_to_constr,
    conic_constraints,
    var_to_indices,
    constraint_indices = Convex.load_MOI_model!(model, problem)

    if warmstart
        warmstart_variables!(model, var_to_indices, id_to_variables, T, verbose)
    end

    if silent_solver
        MOI.set(model, MOI.Silent(), true)
    end

    MOI.optimize!(model)
    problem.model = model

    # populate the status, primal variables, and dual variables (when possible)
    Convex.moi_populate_solution!(
        model,
        problem,
        id_to_variables,
        conic_constr_to_constr,
        conic_constraints,
        var_to_indices,
        constraint_indices,
    )

    if problem.status != MOI.OPTIMAL && verbose
        @warn "Problem status $(problem.status); solution may be inaccurate."
    end

    return model
end
