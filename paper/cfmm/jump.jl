# Individual CFMM constraints
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


function build_jump_arbitrage_model(cfmms, c; Vis_zero::Bool=true, optimizer=Mosek.Optimizer())
    m = length(cfmms)
    n = maximum([maximum(cfmm.Ai) for cfmm in cfmms])

    model = Model(optimizer)
    set_silent(model)
    @variable(model, y[1:n])
    Δs = [@variable(model, [1:length(cfmm)]) for cfmm in cfmms]
    Λs = [@variable(model, [1:length(cfmm)]) for cfmm in cfmms]

    # edge constraints xᵢ = Λᵢ - Δᵢ ∈ Tᵢ
    for (i, cfmm) in enumerate(cfmms)
        add_cfmm(model, cfmm, Δs[i], Λs[i])
    end

    # net flow constraint
    net_flow = zeros(AffExpr, n)
    for (i, cfmm) in enumerate(cfmms)
        @. net_flow[cfmm.Ai] += Λs[i] - Δs[i]
    end
    @constraint(model, y .== net_flow)
    
    # add edge objectives
    if !Vis_zero
        @variable(model, t[1:m])
        x = [@variable(model, [1:length(cfmm)]) for cfmm in cfmms]
        for i in 1:m
            @constraint(model, [0.5; t[i]; x[i]] in MOI.RotatedSecondOrderCone(2+length(cfmms[i])))
            @constraint(model, x[i] .≥ 0.0)
            @constraint(model, x[i] .≥ -(Λs[i] - Δs[i]))
        end
    end
    
    # create objective
    # @constraint(model, y .≥ 0.0)
    @variable(model, p1[1:n])
    @variable(model, p2[1:n])
    for i in 1:n
        @constraint(model, [0.5; p1[i]; p2[i]] in MOI.RotatedSecondOrderCone(3))
        @constraint(model, p2[i] .≥ -y[i])
        @constraint(model, p2[i] .≥ 0.0)
    end
    if !Vis_zero
        @objective(model, Max, dot(c, y) - 0.5*sum(p1) - 0.5*sum(t))
    else
        @objective(model, Max, dot(c, y) - 0.5*sum(p1))
    end

    return model, Δs, Λs
end

function run_trial_jump(cfmms, c; Vis_zero::Bool=true, optimizer=Mosek.Optimizer())
    model, Δs, Λs = build_jump_arbitrage_model(cfmms, c, Vis_zero=Vis_zero, optimizer=optimizer)
    optimize!(model)
    time = solve_time(optimizer)
    status = termination_status(optimizer)
    status != MOI.OPTIMAL && @info "\t\tMosek termination status: $status"