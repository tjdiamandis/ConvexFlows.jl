abstract type SolverState{T} end

#  ---- BFGSState ----
struct BFGSState{
    T <: AbstractFloat,
    V <: AbstractVector{T}, 
    M <: AbstractMatrix{T},    
} <: SolverState{T}
    xk::V               # current iterate
    xprev::V            # previous iterate
    xnext::V            # next iterate
    pk::V               # search direction
    sk::V               # step
    gk::V               # gradient at current iterate
    gnext::V            # gradient at next iterate
    yk::V               # gradient difference
    Hk::M               # inverse Hessian approximation
    vn::V               # cache vector
end

function BFGSState(T::DataType, n::Int)
    return BFGSState(
        zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n),
        zeros(T, n), zeros(T, n), zeros(T, n), 
        Matrix{T}(I, n, n),
        zeros(T, n),
    )
end

function reset_state!(state::BFGSState{T}) where T
    state.xk .= zero(T)
    state.xprev .= zero(T)
    state.pk .= zero(T)
    state.sk .= zero(T)
    
    state.gk .= zero(T)
    state.gnext .= zero(T)
    state.yk .= zero(T)

    state.Hk .= zero(T)
    state.Hk[diagind(state.Hk)] .= one(T)
    
    return nothing
end

function initialize_state!(state::BFGSState{T}; x0=nothing, H0=nothing) where T
    !isnothing(x0) && (state.xk .= x0;)
    !isnothing(H0) && (state.Hk .= H0;)
    return nothing
end


#  ---- BFGSSolver ----
mutable struct BFGSSolver{
    T <: AbstractFloat,
}
    n::Int
    state::SolverState{T}
    obj_val::T
    g_norm::T
    cs_norm::T
    c1::T                   # Wolfe condition 1
    c2::T                   # Wolfe condition 2
end

function BFGSSolver(
    n::Int;
    method=:bfgs,
    T=Float64,
    K=nothing,
    c1=1e-4,
    c2=0.9,
    t=0,
)
    
    state = BFGSState(T, n)

    return BFGSSolver{T}(
        n,
        state,
        zero(T),
        zero(T),
        zero(T),
        convert(T, c1),
        convert(T, c2),
    )
end

function reset_solver!(solver::BFGSSolver{T}) where T
    reset_state!(solver.state)
    return nothing
end

function initialize!(solver::BFGSSolver{T}; x0=nothing, H0=nothing) where T
    initialize_state!(solver.state, x0=x0, H0=H0)
    return nothing
end


#  ---- Options ----
@kwdef mutable struct BFGSOptions
    max_iters::Int = 1000
    max_time_sec::Float64 = 60.0
    print_iter::Int = 1
    verbose::Bool = true
    logging::Bool = true
    eps_g_norm::Float64 = 1e-6
    eps_cs::Float64 = 1e-6
    num_threads::Int = Sys.CPU_THREADS
    μ::Float64 = 10
end


# --- Logging ---
struct BFGSLog{T, V <: AbstractVector{T}}
    fx::Union{Nothing, V}
    g_norm::Union{Nothing, V}
    iter_time::Union{Nothing, V}
    num_iters::Int
    solve_time::Float64
end

function BFGSLog(k::Int, solve_time::T) where {T <: AbstractFloat}
    return BFGSLog{T, Vector{T}}(
        nothing, nothing, nothing,
        k,
        solve_time
    )
end

function Base.show(io::IO, log::BFGSLog)
    print(io, "--- BFGSLog ---\n")
    print(io, "num iters  :  $(log.num_iters)\n")
    print(io, "solve time :  $(round(log.solve_time, digits=3))s\n")
end

function create_temp_log(solver::BFGSSolver, max_iters::Int)
    T = typeof(solver.obj_val)
    return BFGSLog(
        zeros(T, max_iters),
        zeros(T, max_iters),
        zeros(T, max_iters),
        zero(Int),
        zero(T)
    )
end

function populate_log!(solver_log::BFGSLog, solver::BFGSSolver, ::BFGSOptions, k, time_sec)
    solver_log.iter_time[k] = time_sec
    solver_log.fx[k] = solver.obj_val
    solver_log.g_norm[k] = solver.g_norm
    return nothing
end


# --- Results ---
struct BFGSResult{T}
    status::Symbol
    obj_val::T
    g_norm::T
    x::Vector{T}
    log::BFGSLog{T}
end

function Base.show(io::IO, result::BFGSResult)
    print(io, "--- Result ---\n")
    print(io, "Status:     ")
    color = result.status == :OPTIMAL ? :green : :red
    printstyled(io, " $(result.status)\n", color=color)
    print(io, "f(x)       :   $(@sprintf("%.4g", result.obj_val))\n")
    print(io, "∇f(x)      :   $(@sprintf("%.4g", result.obj_val))\n")
    print(io, "num iters  :   $(result.log.num_iters)\n")
    print(io, "solve time : $(round(result.log.solve_time, digits=3))s\n")
end