function is_optimal(result::ConvexFlows.BFGSResult; eps_g_norm=1e-6)
    return result.status == :OPTIMAL && result.g_norm < eps_g_norm
end

# parameters
g_norm_tol = 1e-6
options = BFGSOptions(eps_g_norm=g_norm_tol, verbose=false)
Random.seed!(1)

@testset "quadratic" begin
    # test function quadratic: f(x) = ||x - v||² + cᵀx
    n = 10
    xstar = 10*rand(n)
    c = 10*rand(n)
    v = @. xstar + c/2
    p1 = (v=v, c=c)
    function f1∇f1!(g, x, p)
        v, c = p.v, p.c
        @. g = 2(x - v) + c
        return norm(x - v)^2 + dot(c, x)
    end
    x0 = zeros(n)
    solver1 = BFGSSolver(n)
    res1 = solve!(solver1, f1∇f1!, p1; options=options, x0=x0)
    @test isapprox(xstar, res1.x; atol=1e-6)
    @test is_optimal(res1; eps_g_norm=g_norm_tol)

    # LBFGS
    s1_lbfgs = BFGSSolver(n; method=:lbfgs)
    res1_lbfgs = solve!(s1_lbfgs, f1∇f1!, p1; options=options, x0=x0)
    @test is_optimal(res1_lbfgs; eps_g_norm=g_norm_tol)
end

@testset "rosenbrock" begin
    # test function: f(x) = 100(x₂ - x₁²)² + (1 - x₁)²
    # ∇f(x) = [200(x₂ - x₁²)*2x₁ + 2(1 - x₁), 200(x₂ - x₁²)]
    n = 2
    p2 = nothing
    function f2∇f2!(g, x, p)
        x₁, x₂ = x[1], x[2]
        g[1] = -200(x₂ - x₁^2)*2x₁ - 2(1 - x₁)
        g[2] = 200(x₂ - x₁^2)
        return 100(x₂ - x₁^2)^2 + (1 - x₁)^2
    end
    x0 = (0.1, 0.1)
    solver2 = BFGSSolver(n)
    res2 = solve!(solver2, f2∇f2!, p2; options=options, x0=x0)
    @test is_optimal(res2; eps_g_norm=g_norm_tol)
    
    # LBFGS
    s2_lbfgs = BFGSSolver(n; method=:lbfgs)
    res2_lbfgs = solve!(s2_lbfgs, f2∇f2!, p2; options=options, x0=x0)
    @test is_optimal(res2_lbfgs; eps_g_norm=g_norm_tol)

    
    # test iteration limit
    options_iter_limit = BFGSOptions(max_iters=2, verbose=false, eps_g_norm=g_norm_tol)
    res_limit = solve!(solver2, f2∇f2!, p2; options=options_iter_limit, x0=x0)
    @test res_limit.status == :ITERATION_LIMIT
end