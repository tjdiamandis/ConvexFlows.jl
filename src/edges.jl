abstract type Edge{T} end

@def add_generic_fields begin
    Ai::Vetcor{Int}
end
Base.length(e::Edge) = length(e.Ai)

function find_arb! end