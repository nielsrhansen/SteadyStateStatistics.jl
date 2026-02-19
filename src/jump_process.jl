struct jump_process
    t::Array{Float64, 1} # Cumulative time
    Δ::Array{Float64, 2} # Jumps in d dimensions
end

struct compound_poisson
    d::Int
    λ::Array{Float64, 1}
    jump::Array{Sampleable, 1}
end

function rand(x::compound_poisson, r::Integer = 1)
    λ_total = sum(x.λ)
    Δ = zeros(x.d, r)
    t = rand(Distributions.Exponential(1 / λ_total), r) |> cumsum
    k = sample(1:x.d, StatsBase.Weights(x.λ / λ_total), r)
    for j in 1:r
        Δ[k[j], j] = rand(x.jump[k[j]])
    end
    return jump_process(t, Δ)
end

function compound_gamma(d; λ = 1, α = 1, θ = 1)
    if (length(λ) == 1)
        λ = fill(λ, d)
    end
    if (length(α) == 1)
        α = fill(α, d)
    end
    if (length(θ) == 1)
        θ = fill(θ, d)
    end
    return compound_poisson(d, λ, [Gamma(α[i], θ[i]) for i in 1:d])
end

function compound_signed_gamma(d; λ = 1, α = 1, θ = 1, p = 1)
    if (length(λ) == 1)
        λ = fill(λ, d)
    end
    if (length(α) == 1)
        α = fill(α, d)
    end
    if (length(θ) == 1)
        θ = fill(θ, d)
    end
    if (length(p) == 1)
        p = fill(p, d)
    end
    return compound_poisson(d, λ, [signed_gamma(α[i], θ[i], p[i]) for i in 1:d])
end

function compound_beta(d; λ = 1, α = 1, β = 1)
    if (length(λ) == 1)
        λ = fill(λ, d)
    end
    if (length(α) == 1)
        α = fill(α, d)
    end
    if (length(β) == 1)
        β = fill(β, d)
    end
    return compound_poisson(d, λ, [Beta(α[i], β[i]) for i in 1:d])
end

function plot(Z::jump_process; kwargs...)
    jumps = Z.Δ[1, :] .!= 0
    cum_jumps = cumsum(Z.Δ, dims = 2)
        Plots.plot(
            Z.t[jumps], 
            cum_jumps[1, jumps],
            xlabel = "Time",
            ylabel = "Cumulative jumps",
            title = "Jump process";
            kwargs...
        ) 
    for k in 2:size(Z.Δ, 1)
        jumps = Z.Δ[k, :] .!= 0
        Plots.plot!(
            Z.t[jumps], 
            cum_jumps[k, jumps];
            kwargs...
        )
    end
    Plots.ylims!(minimum(cum_jumps) - 1, maximum(cum_jumps) + 1) |> display
end