function build_jump_opf_model(
    lines,
    d;
    optimizer=() -> Mosek.Optimizer(),
    verbose=false,
    mosek_high_precision=false,
)
    m = length(lines)
    n = length(d)

    model = Model(optimizer)
    if mosek_high_precision
        set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-16)
        set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_MU_RED", 1e-16)
        set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-16)
        set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-16)
        set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL", 1.0)
        set_optimizer_attribute(model, "MSK_IPAR_PRESOLVE_USE", 0)
    end
    !verbose && set_silent(model)
    @variable(model, y[1:n])

    # create objective
    # U(y) = ∑ -(¹/₂)(dᵢ - yᵢ)₊²
    @variable(model, t1[1:n])
    @variable(model, t2[1:n])
    for i in 1:n
        @constraint(model, [0.5; t1[i]; t2[i]] in MOI.RotatedSecondOrderCone(3))
        @constraint(model, t2[i] .≥ d[i] - y[i])
        @constraint(model, t2[i] .≥ 0.0)
    end
    @objective(model, Max, -0.5*sum(t1))

    # create constraints
    xs = [@variable(model, [1:2]) for _ in 1:m]
    uv = [@variable(model, [1:2]) for _ in 1:m]

    # net flow constraint
    net_flow = zeros(AffExpr, n)
    for (i, line) in enumerate(lines)
        @. net_flow[line.Ai] += xs[i]
    end
    @constraint(model, y .== net_flow)

    # line constraints
    for i in 1:m
        α, β, b = lines[i].α, lines[i].β, lines[i].b
        x1, x2 = xs[i][1], xs[i][2]
        u, v = uv[i][1], uv[i][2]

        @constraint(model, x1 ≤ 0)
        @constraint(model, -b ≤ x1)

        # log(1 + exp(-βx₁)) ≤ -3α⁻¹x₁ - α⁻¹x₂ + log(2)
        # NOTE: Mosek modeling cookbook flips 1st & last argument compared to MOI
        @constraint(model, u + v ≤ 1.0)
        @constraint(model, 
            [-β*x1 + (3x1 + x2)/α - log(2); 1.0; u] in MOI.ExponentialCone()
        )
        @constraint(model, 
            [(3x1 + x2)/α - log(2); 1.0;  v] in MOI.ExponentialCone()
        )
    end

    return model, xs
end

