# Checks if a set of trades is valid for a set of CFMMs
function valid_trades(cfmms; xs=nothing, Δs=nothing, Λs=nothing)
    if !isnothing(xs) && isnothing(Δs) && isnothing(Λs)
        Δs = [max.(-x, 0.0) for x in xs]
        Λs = [max.(x, 0.0) for x in xs]
    elseif isnothing(Δs) || isnothing(Λs)
        error("Must provide either xs or Δs and Λs")
    end

    for (i, cfmm) in enumerate(cfmms)
        !valid_trade(cfmm, Δs[i], Λs[i]) && (println(i); true) && return false
    end
    return true
end


# returns relative netflow error
function net_flow_error(y, cfmms; xs=nothing, Δs=nothing, Λs=nothing)
    if isnothing(xs)
        xs = [Λs[i] - Δs[i] for i in 1:length(cfmms)]
    end

    net_flow = zeros(length(y))
    for (i, cfmm) in enumerate(cfmms)
        @. net_flow[cfmm.Ai] += xs[i]
    end
    return norm(net_flow - y) / max(norm(net_flow), norm(y))
end


function check_optimality(ν, cfmms; ηs=nothing, xs=nothing, Δs=nothing, Λs=nothing, tol=1e-3)
    if !isnothing(xs) && isnothing(Δs) && isnothing(Λs)
        Δs = [max.(-x, 0.0) for x in xs]
        Λs = [max.(x, 0.0) for x in xs]
    elseif isnothing(Δs) || isnothing(Λs)
        error("Must provide either xs or Δs and Λs")
    end

    if isnothing(ηs)
        ηs = [ν[cfmm.Ai] for cfmm in cfmms]
    end

    subopt = 0.0
    count = 0
    for (i, cfmm) in enumerate(cfmms)
        Rp = cfmm.R + cfmm.γ * Δs[i] - Λs[i]
        subopt_i = suboptimality(cfmm, Rp, ηs[i])
        subopt += subopt_i
        if subopt_i > tol
            count += 1
            @warn "CFMM $i is not optimal: $(subopt_i)"
        end
    end

    return subopt / length(cfmms), count
end