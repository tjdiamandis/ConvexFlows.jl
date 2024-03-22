function print_header_bfgs(format, data)
    @printf(
        "\n─────────────────────────────────────────────────────────────────────────────────────────────────\n"
    )
    format_str = Printf.Format(join(format, " ") * "\n")
    Printf.format(
        stdout,
        format_str,
        data...
    )
    @printf(
        "─────────────────────────────────────────────────────────────────────────────────────────────────\n"
    )
end


function print_headers(::BFGSSolver, ::BFGSOptions)
    return ["Iteration", "f", "||∇f||", "Time"]
end

function header_format(::BFGSSolver, ::BFGSOptions)
    return ["%13s", "%14s", "%14s", "%14s"]
end

function print_iter(format, data)
    format_str = Printf.Format(join(format, " ") * "\n")
    Printf.format(
        stdout,
        format_str,
        data...
    )
end

function iter_format(::BFGSSolver, ::BFGSOptions)
    return ["%13s", "%14.3e", "%14.3e", "%13.3f"]
end


# ---- BFGS Stuff ----
# BFGS search direction (primal)
function compute_search_direction!(state::BFGSState{T}) where T
    Hk, g, pk = state.Hk, state.gk, state.pk
    mul!(pk, Hk, g)
    pk .= -pk
    return nothing
end

# BFGS update
function update_Hk!(state::BFGSState{T}) where T
    Hk, sk, yk = state.Hk, state.sk, state.yk
    ρk = 1 / dot(yk, sk)
    isnan(ρk) && return nothing

    Hk .= (I - ρk * sk * yk') * Hk * (I - ρk * yk * sk') + ρk * sk * sk'
    return nothing
end

# scaling from NW06 S6.1 p143
function scale_H0!(state::BFGSState{T}) where T
    Hk, sk, yk = state.Hk, state.sk, state.yk
    c = dot(yk, sk) / dot(yk, yk)
    isnan(c) && (c = one(T);)
    Hk .= c * Hk
    return nothing
end


# ---- LBFGS Stuff ----
# LBFGS search direction
function compute_search_direction!(state::LBFGSState{T}) where T
    pk, ind = state.pk, state.ind[1]
    yks, sks, γk = state.yks, state.sks, state.γk[1]
    ρs, αs, βs = state.ρs, state.αs, state.βs
    m = length(state.sks)
    
    # NW06 Algorithm 7.4 p178
    reverse_inds = mod.((ind-1:-1:ind-m) .- 1, m) .+ 1
    forward_inds = mod.((ind-m:ind-1) .- 1, m) .+ 1

    pk .= state.gk
    for i in reverse_inds
        (iszero(ρs[i]) || isinf(ρs[i])) && continue
        αs[i] = ρs[i] * dot(sks[i], pk)
        @. pk -= αs[i] * yks[i]
    end

    pk .= γk[1] * pk
    for i in forward_inds
        (iszero(ρs[i]) || isinf(ρs[i])) && continue
        βs[i] = ρs[i] * dot(yks[i], pk)
        @. pk += (αs[i] - βs[i]) * sks[i]
    end
    pk .= -pk

    return nothing
end

function update_Hk!(state::LBFGSState{T}) where T
    sks, yks, ind = state.sks, state.yks, state.ind[1]

    prev_ind = mod(ind-2, length(sks)) + 1
    state.γk[1] = dot(yks[prev_ind], sks[prev_ind]) / dot(yks[prev_ind], yks[prev_ind])
    state.γk[1] = isnan(state.γk[1]) ? one(T) : state.γk[1]

    sks[ind] .= state.sk
    yks[ind] .= state.yk
    state.ρs[ind] = 1 / dot(yks[ind], sks[ind])
    ind = mod(ind, length(sks)) + 1
    return nothing
end

function scale_H0!(::LBFGSState{T}) where T
    return nothing
end

# Bracketing line search 
#   ref: https://cs.nyu.edu/~overton/papers/pdffiles/bfgs_inexactLS.pdf
function line_search(solver::BFGSSolver{T}, f∇f!, p, fxk, gxk) where T
    xk, pk = solver.state.xk, solver.state.pk
    xnext, vn = solver.state.xnext, solver.state.vn
    c1, c2 = solver.c1, solver.c2
    lb0, ub0 = zero(T), typemax(T)

    # # ensure we don't step out of feasible region
    # # TODO: faster to broadcast to cache vector and then compute max?
    tmax_x = minimum(i -> pk[i] < 0 ? -xk[i] / pk[i] : typemax(T), 1:solver.n)
    # ub0 = min(tmax_λ, tmax_x) - sqrt(eps(T))
    ub0 = tmax_x - sqrt(eps(T))

    lb, ub = lb0, ub0
    @inline function hdh!(vn, α, fxk)
        @. xnext = xk + α * pk
        
        # ensures we don't step out of feasible region x ≥ 0
        any(xi -> xi < sqrt(eps()), xnext) && return typemax(T), typemax(T)

        fnext = f∇f!(vn, xnext, p)
        # fnext += sum(xi -> xi < sqrt(eps) ? -solver.g_norm*log(xi) : zero(T), xnext)
        h = fnext - fxk
        # vn .+= solver.g_norm ./ xnext
        dh = dot(vn, pk)
        return h, dh
    end
    dh_0 = dot(gxk, pk)

    # h(αk) = f(x + αk*p) - f(x)
    # looks for a weak Wolfe step
    αk = min(one(T), ub0)
    for _ in 1:20
        # TODO: should keep track of function evals
        h, dh = hdh!(vn, αk, fxk)
        # if h(αk) / αk < c₁h'(0) fails, ub = αk
        if h ≥ c1 * dh_0 * αk
            ub = αk

        # elseif h'(αk) > c₂h'(0) fails, lb = αk
        elseif dh ≤ c2 * dh_0
            lb = αk

        # else step satisifies weak Wolfe; STOP
        else
            break
        end

        αk = ub < typemax(T) ? (lb + ub) / 2 : 2lb
    end

    return αk
end


function update_x!(solver::BFGSSolver{T}, α::T) where T
    solver.state.xprev .= solver.state.xk
    @. solver.state.xk += α * solver.state.pk
    @. solver.state.sk = solver.state.xk - solver.state.xprev
    return nothing
end

function update_gradient!(solver::BFGSSolver{T}) where T
    state = solver.state
    @. state.yk = state.gnext - state.gk
    @. state.gk = state.gnext
    return nothing
end

function converged(solver::BFGSSolver{T}, options::BFGSOptions) where T
    return solver.g_norm < options.eps_g_norm
end

function solve!(
    solver::BFGSSolver{T},
    f∇f!::Function,
    p;
    options=BFGSOptions(),
    x0=nothing,
    H0=nothing,
    reset_solver=true,
) where {T}

    # --- steup ---
    options.verbose && @printf("Starting setup...")
    k = 0
    n = solver.n
    state = solver.state
    xk, gk, gnext = state.xk, state.gk, state.gnext

    # --- enable multithreaded BLAS ---
    BLAS.set_num_threads(options.num_threads)

    # --- Logging ---
    if options.logging
        tmp_log = create_temp_log(solver, options.max_iters + 1)
    end

    # --- Print Headers ---
    format = header_format(solver, options)
    headers = print_headers(solver, options)
    iter_fmt = iter_format(solver, options)
    options.verbose && print_header_bfgs(format, headers)

    # --- Initialize solver ---
    reset_solver && reset_solver!(solver)
    if !isnothing(x0) || !isnothing(H0)
        initialize!(solver, x0=x0, H0=H0)
    else
        solver.state.xk .= one(T)
    end


    # *********************
    # *  main algorithm   *
    # *********************
    solve_time_start = time_ns()

    # Compute values at x0
    solver.obj_val = f∇f!(gk, xk, p)
    solver.g_norm = norm(gk)
    time_sec = (time_ns() - solve_time_start) / 1e9
    options.logging && populate_log!(tmp_log, solver, options, k+1, time_sec, xk)
    options.verbose && print_iter(
        iter_fmt, (k, solver.obj_val, solver.g_norm, time_sec)
    )

    while k < options.max_iters &&
        (time_ns() - solve_time_start) / 1e9 < options.max_time_sec

        k += 1

        # --- Compute search direction and step ---
        compute_search_direction!(solver.state)
        αk = line_search(solver, f∇f!, p, solver.obj_val, gk)
        
        #  --- Update state ---
        update_x!(solver, αk)
        solver.obj_val = f∇f!(gnext, xk, p)
        update_gradient!(solver)
        solver.g_norm = norm(solver.state.gk)
        
        # TODO: add barrier?

        # --- Logging ---
        time_sec = (time_ns() - solve_time_start) / 1e9
        options.logging && populate_log!(tmp_log, solver, options, k+1, time_sec, xk)

        # --- Printing ---
        if options.verbose && (k == 1 || k % options.print_iter == 0)
            print_iter(
                iter_fmt,
                (k, solver.obj_val, solver.g_norm, time_sec)
            )
        end

        # --- Check convergence ---
        converged(solver, options) && break
        
        k == 1 && isnothing(H0) && scale_H0!(state)
        update_Hk!(state)
    end

    # --- Print Footer ---
    solve_time = (time_ns() - solve_time_start) / 1e9
    if !converged(solver, options)
        options.verbose && @printf("\nWARNING: did not converge after %d iterations, %6.3fs:", k, solve_time)
        if k >= options.max_iters
            options.verbose && @printf(" (max iterations reached)\n")
            status = :ITERATION_LIMIT
        elseif (time_ns() - solve_time_start) / 1e9 >= options.max_time_sec
            options.verbose && @printf(" (max time reached)\n")
            status = :TIME_LIMIT
        end
    else
        options.verbose && @printf("\nSOLVED in %6.3fs, %d iterations\n", solve_time, k)
        options.verbose && @printf("Total time: %6.3fs\n", solve_time)
        status = :OPTIMAL
    end 
    options.verbose && print_footer()

    # --- Construct Logs ---
    if options.logging
        log = BFGSLog(
            tmp_log.fx[1:k+1],
            tmp_log.g_norm[1:k+1],
            tmp_log.xk[1:k+1],
            tmp_log.iter_time[1:k+1],
            k,
            solve_time
        )
    else
        log = BFGSLog(k, solve_time)
    end

    # --- Construct Solution ---
    res = BFGSResult(
        status,
        solver.obj_val,
        solver.g_norm,
        solver.state.xk,
        log
    )

    return res
end
