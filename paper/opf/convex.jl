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
    
Convex.solve!(prob, Mosek.Optimizer())
prob.optval