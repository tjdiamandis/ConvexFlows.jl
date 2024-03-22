using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra, Random, SparseArrays
using StatsBase, LogExpFunctions
import Graphs: Graph, connected_components
using Plots, LaTeXStrings

Pkg.activate(joinpath(@__DIR__, "..", ".."))
using ConvexFlows

const FIGPATH = joinpath(@__DIR__, "..", "figures")

include(joinpath(@__DIR__, "..", "opf", "utils.jl"))
# ******************************************************************************

# Problem parameters
n = 3
days = 5
T = 24*days
N = n*T

d_user = sin.((1:T) .* 2π ./ 24) .+ 1.5
c_user = 100.0
d_gen = 0.0*ones(T)
c_gen = 1.0

d = vec(vcat(d_user', d_user', d_gen'))
c = repeat([c_user, c_user, c_gen], T)
# obj = NonpositiveQuadratic(d, a=c)
obj = NonpositiveQuadratic(d; a=c)

# Network: two nodes, both connected to generator
function build_edges(n, T; bat_node)
    net_edges = [(i,n) for i in 1:n-1]
    edges = Edge[]

    # Transmission Lines (4 = 2 * 2)
    h(w) = 3w - 16.0*(log1pexp(0.25 * w) - log(2))
    function wstar(ηrat, b)
        ηrat ≥ 1.0 && return 0.0
        # ηrat ≤ 0.0 && return b
        return min(4.0 * log((3.0 - ηrat)/(1.0 + ηrat)), b)
    end
    for (i,j) in net_edges
        bi = 4.0
        for t in 1:T
            it = i + (t-1)*n
            jt = j + (t-1)*n
            push!(edges, Edge(
                (it, jt); h=h, ub=bi, wstar = @inline ηrat -> wstar(ηrat, bi))
            )
            push!(edges, Edge(
                (jt, it); h=h, ub=bi, wstar = @inline ηrat -> wstar(ηrat, bi)
            ))
        end
    end

    # Storage edges
    ϵ = 1e-2
    wstar_storage(ηrat, γ, b) = ηrat ≥ γ ? 0.0 : min(1/ϵ*(γ - ηrat), b)
    # only node 2 has storage
    for t in 1:T-1
        it = bat_node + (t-1)*n
        it_next = bat_node + t*n
        γi = 1.0
        storage_capacity = 10.0
        push!(edges, Edge(
            (it, it_next); 
            h= w->γi*w - ϵ/2*w^2, 
            ub=storage_capacity, 
            wstar = @inline ηrat -> wstar_storage(ηrat, γi, storage_capacity)
        ))
    end
    return edges
end

edges = build_edges(n, T, bat_node=2)
prob = problem(obj=obj, edges=edges)
result_bfgs = solve!(prob; method=:bfgs)

yt = reshape(prob.y, (n, T))

# Generator Plot
u3_gen = max.(d_gen - vec(yt[3,:]), 0.0)
@assert sum(abs2, [x[1] for x in prob.xs[1:2:4T]]) == 0.0
@assert sum(abs2, [x[1] for x in prob.xs[2T+1:2:4T]]) == 0.0
gen_u1 = [-x[1] for x in prob.xs[2:2:2T]]
gen_u2 = [-x[1] for x in prob.xs[2T+2:2:4T]]
generator_plt = plot(
    (1:T) ./ 24, 
    u3_gen,
    label="Generator",
    xlabel="Time (days)",
    ylabel="Power",
    lw=3,
    size=(800, 300),
    dpi=300,
    legend=:topright,
    legend_column=-1,
    color=:blue,
    legendfontsize=12,
    tickfontsize=12,
    guidefontsize=12,
    legendtitlefontsize=12,
    bottommargin=6Plots.mm,
    leftmargin=6Plots.mm,
    ylims=(0,8)
)
plot!(
    generator_plt,
    (1:T) ./ 24, 
    gen_u1,
    label="To user 1",
    lw=2,
    ls=:dash,
    color=:red,
)

plot!(
    generator_plt,
    (1:T) ./ 24, 
    gen_u2,
    label="To user 2",
    lw=2,
    ls=:dot,
    color=:green,
)



# Battery Plot
u2_sell = [-x[1] for x in prob.xs[2T+1:2:4T]]
u2_buy = [x[2] for x in prob.xs[2T+2:2:4T]]
u2_bat_in = vcat([0.0], [x[2] for x in prob.xs[4T+1:end]])
u2_bat_out = vcat([-x[1] for x in prob.xs[4T+1:end]], [0.0])
u2_bat_net = u2_bat_in .- u2_bat_out
u2_gen = max.(d_gen .- vec(yt[2,:]), 0.0)
@assert sum(abs2, u2_sell) + sum(abs2, u2_gen) == 0
battery_plot = plot(
    (1:T) ./ 24, 
    d_user,
    label="Demand",
    xlabel="Time (days)",
    ylabel="Power",
    lw=3,
    size=(800, 300),
    dpi=300,
    legend=:topright,
    legend_column=2,
    color=:black,
    legendfontsize=12,
    tickfontsize=12,
    guidefontsize=12,
    legendtitlefontsize=12,
    bottommargin=6Plots.mm,
    leftmargin=6Plots.mm,  
    ylims=(0,4)  
)
plot!(
    battery_plot,
    (1:T) ./ 24, 
    u2_buy,
    label="Purchased",
    lw=2,
    color=:blue,
)

plot!(
    battery_plot,
    (1:T) ./ 24, 
    max.(u2_bat_net, 0.0),
    label="Battery Discharge",
    lw=2,
    color=:red,
)

plot!(
    battery_plot,
    (1:T) ./ 24, 
    max.(-u2_bat_net, 0.0),
    label="Battery Charge",
    lw=2,
    ls=:dash,
    color=:green,
)


# No Battery Plot
u1_sell = [x[2] for x in prob.xs[1:2:2T]]
u1_buy = [x[2] for x in prob.xs[2:2:2T]]
u1_gen = max.(d_user .- vec(yt[1,:]), 0.0)
@assert sum(abs2, u1_sell) == 0

no_bat_plot = plot(
    (1:T) ./ 24, 
    d_user,
    label="Demand",
    xlabel="Time (days)",
    ylabel="Power",
    lw=3,
    size=(800, 300),
    dpi=300,
    legend=:topright,
    legend_column=-1,
    color=:black,
    legendfontsize=12,
    tickfontsize=12,
    guidefontsize=12,
    legendtitlefontsize=12,
    bottommargin=6Plots.mm,
    leftmargin=6Plots.mm,  
    ylims=(0,4)  
)
plot!(
    no_bat_plot,
    (1:T) ./ 24, 
    u1_buy,
    label="Purchased",
    lw=2,
    color=:blue,
)
plot!(
    no_bat_plot,
    (1:T) ./ 24, 
    u1_gen,
    label="Generated",
    lw=2,
    color=:red,
)

savefig(generator_plt, joinpath(FIGPATH, "opf-two-node-generator.pdf"))
savefig(battery_plot, joinpath(FIGPATH, "opf-two-node-battery.pdf"))
savefig(no_bat_plot, joinpath(FIGPATH, "opf-two-node-no_bat.pdf"))
