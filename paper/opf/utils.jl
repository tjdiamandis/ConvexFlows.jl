# From S6.1 of https://web.stanford.edu/~boyd/papers/pdf/msg_pass_dyn.pdf
function build_graph(n; d=0.11, α=0.8, rseed=1)
    Random.seed!(rseed)
    xys = [sqrt(n)*rand(2) for _ in 1:n]
    I = Vector{Int}()
    sizehint!(I, n)
    J = Vector{Int}()
    sizehint!(J, n)
    V = Vector{Int}()
    sizehint!(V, n)
    for i in 1:n, j in i+1:n
        dist = norm(xys[i] - xys[j])
        γ = α * min(1.0, d^2 / dist^2)
        if rand() ≤ γ
            push!(I, i)
            push!(J, j)
            push!(V, 1)
        end
    end
    Adj = sparse(I, J, V, n, n)
    @. Adj = Adj + Adj'

    # connect isolated nets to nearest neighbor
    for i in 1:n
        sum(Adj[i, :]) > 0 && continue
        dists = [norm(xys[i] - xys[j]) for j in 1:n if j ≠ i]
        jmin = argmin(dists)
        Adj[i, jmin] = 1
        Adj[jmin, i] = 1
    end

    # connect connected components
    ccs = connected_components(Graph(Adj))
    nccs = length(ccs)
    remaining = Set(1:nccs)
    while nccs > 1
        cc_i, cc_j = sample(collect(remaining), 2, replace=false)
        i = rand(ccs[cc_i])
        j = rand(ccs[cc_j])
        Adj[i, j] = 1
        Adj[j, i] = 1
        ccs[cc_i] = union(ccs[cc_i], ccs[cc_j])
        remaining = setdiff(remaining,  Set([cc_j]))
        nccs -= 1
    end

    return Adj, xys
end