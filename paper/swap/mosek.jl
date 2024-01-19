function add_cfmm(optimizer, cfmm::Uniswap, Δ, Λ)
    R, γ = cfmm.R, cfmm.γ
    ϕR = sqrt(R[1]*R[2])
    @constraint(optimizer, vcat(R + γ * Δ - Λ, ϕR) in MOI.PowerCone(0.5))
    @constraint(optimizer, Δ .≥ 0)
    @constraint(optimizer, Λ .≥ 0)
    return nothing
end

function add_cfmm(optimizer, cfmm::Balancer, Δ, Λ)
    R, γ, w = cfmm.R, cfmm.γ, cfmm.w
    ϕR = R[1]^w*R[2]^(1-w)
    @constraint(optimizer, vcat(R + γ * Δ - Λ, ϕR) in MOI.PowerCone(w))
    @constraint(optimizer, Δ .≥ 0)
    @constraint(optimizer, Λ .≥ 0)
    return nothing
end

function add_cfmm(optimizer, cfmm::BalancerThreePool, Δ, Λ)
    R, γ = cfmm.R, cfmm.γ
    ϕR = (R[1]*R[2]*R[3])^(1/3)
    @constraint(optimizer, [ϕR; R + γ * Δ - Λ] in MOI.GeometricMeanCone(3+1))
    @constraint(optimizer, Δ .≥ 0)
    @constraint(optimizer, Λ .≥ 0)
    return nothing
end


function build_mosek_swap_model(cfmms, j_in, j_out, Δ, Vis_zero::Bool=true)
    m = length(cfmms)
    n = maximum([maximum(cfmm.Ai) for cfmm in cfmms])

    optimizer = Model(Mosek.Optimizer)
    set_silent(optimizer)
    @variable(optimizer, y[1:n])
    Δ = [@variable(optimizer, [1:length(c)]) for c in 1:cfmms]
    Λ = [@variable(optimizer, [1:length(c)]) for c in 1:cfmms]

    # edge constraints xᵢ = Λᵢ - Δᵢ ∈ Tᵢ
    for cfmm in cfmms
        add_cfmm(optimizer, cfmm, Δ[i], Λ[i])
    end

    # net flow constraint
    net_flow = zeros(AffExpr, n_tokens)
    for i in 1:m
        @. net_flow[As[i]] += Λ[i] - Δ[i]
    end
    @constraint(optimizer, y .== net_flow)
    
    # add node objective
    @constraint(optimizer, y[j_in] == -Δ)
    inds = collect(setdiff(1:n, [j_in, j_out]))
    @constraint(optimizer, y[inds] .== 0.0)
    
    # add edge objectives
    if !Vis_zero
        @variable(optimizer, t[1:n])
        x = [@variable(optimizer, [1:length(c)]) for c in cfmms]
        for i in 1:m
            @constraint(optimizer, [0.5; -2t[i]; x[i]] in MOI.RotatedSecondOrderCone(2+2))
            @constraint(optimizer, x[i] .≥ 0.0)
            @constraint(optimizer, x[i] .≥ -Λ[i] + Δ[i])
        end
    end
    
    # create objective
    if !Vis_zero
        @objective(optimizer, Max, y[j_out] + sum(t))
    else
        @objective(optimizer, Max, y[j_out])
    end

    return optimizer
end