using LinearAlgebra, StatsBase, Distributions, ExponentialUtilities, Random
using DataFrames
using CairoMakie
using PairPlots
using Smoothers, SmoothingSplines, Plots #, PlotlyJS
using Cumulants
import Base: rand


λ = [1, 0.5]
λ_total = sum(λ)

Z = jump_process(
    [0.0, 0.0], 
    [[0.0, 0.0] [0.0, 0.0]]
)

"""
    sim_cp(d, λ, α, θ, r = 1)

    Simulation of a compound Poisson process with d-dimensional jumps
    generated independently from signed gamma distributions.

    # Arguments
    - `d::Int`: Dimension of the process
    - `r::Int`: Number of jumps
    - `λ::Array{Float64, 1}`: Intensities of the d Poisson processes
    - `α::Array{Float64, 1}`: Shape parameters of the gamma distributions
    - `θ::Array{Float64, 1}`: Scale parameters of the gamma distributions
    - `p::Array{Float64, 1}`: Probabilities of jumps have positive sign
"""

## Compound gamma simulation

cp = compound_gamma(2; λ = λ, α = [5, 5], θ = [1, 4])
Z = rand(cp, 100)

Plots.plot(Z.t, cumsum(Z.Δ[1, :]), seriestype = :scatter, xlabel = "Time", ylabel = "Jump Size", title = "Compound Poisson Process")
Plots.plot!(Z.t, cumsum(Z.Δ[2, :]), seriestype = :scatter)

cp = compound_signed_gamma(2; λ = λ, α = [5, 5], θ = [1, 4], p = [1, 0])
Z = rand(cp, 100)

Plots.plot(Z.t, cumsum(Z.Δ[1, :]), seriestype = :scatter, xlabel = "Time", ylabel = "Jump Size", title = "Compound Poisson Process")
Plots.plot!(Z.t, cumsum(Z.Δ[2, :]), seriestype = :scatter)

## Compound beta simulation


cp = compound_beta(2; λ = [100, 1], α = [1, 1], β = [100, 0.01])
Z = rand(cp, 1000)


jump_1 = Z.Δ[1, :] .> 0 
Plots.plot(Z.t[jump_1], cumsum(Z.Δ[1, jump_1]), seriestype = :scatter, xlabel = "Time", ylabel = "Jump Size", title = "Compound Poisson Process")

jump_2 = Z.Δ[2, :] .> 0 
Plots.plot!(Z.t[jump_2], cumsum(Z.Δ[2, jump_2]), seriestype = :scatter)


# The OSD distributions are given by the representation:
#
# X = int_0^infty exp(tM) dZ_t
#
# where M is a stable d x d matrix and Z is a d-dimensional Levy process.
# If Z is a compound Poisson process, 
#
# X = \sum_i^infty exp(T_i M) Δ Z_{T_i}
#
# where T_i is the i-th jump time and Δ Z_{T_i} = (Z_{T_i} - Z_{T_i-}) is 
# the i-th jump of Z. 

# Simple 2x2 example with signed gamma jumps for the underlying Z-Levy process.
# The process has positive feedforward and a negative feedback.

M = [-1 1; -0.3 -0.2]
cp = compound_signed_gamma(2; λ = λ, α = [1, 2], θ = [5, 1], p = [0, 1])
osd_cp = osd(M, cp)

Random.seed!(123)
X = rand(osd_cp, 1000)

Plots.plot(X[:, 1], X[:, 2], seriestype = :scatter, xlabel = "X_1", ylabel = "X_2", title = "OSD Distribution")

order = sortperm(X[:, 1])
spl = fit(SmoothingSpline, X[order, 1], X[order, 2], 200.0) 
Ypred = predict(spl)
Plots.plot!(X[order, 1], Ypred, color = :red, linewidth = 3, label = "Smoothing Spline")


# Simple 2x2 example with beta distributed jumps for the underlying Z-Levy process.
# The process has positive feedforward and a negative feedback.


Random.seed!(123)

M = [-1 1; -1 -1]
c = 0.5
# cp = compound_gamma(2; λ = λ, α = [1, 0.1], θ = [5, 10])
cp = compound_beta(2; λ = c * [100, 1], α = [1, 1], β = [100, 0.01])
osd_cp = osd(M , cp)

X = rand(osd_cp, 1000)

Plots.plot(X[:, 1], X[:, 2], seriestype = :scatter, xlabel = "X_1", ylabel = "X_2", title = "OSD Distribution")

order = sortperm(X[:, 1])
spl = fit(SmoothingSpline, X[order, 1], X[order, 2], 200.0) 
Ypred = predict(spl)
Plots.plot!(X[order, 1], Ypred, color = :red, linewidth = 3, label = "Smoothing Spline")


# Larger example

d = 6  # specify the size of the matrix
λ = 1
α = 1
β = 1
θ = 0.1
p = repeat([0, 1], 5)
C = diagm()

# cp = compound_signed_gamma(d; λ = λ, α = α, θ = θ, p = p)
cp = compound_beta(d; λ = λ, α = α, β = β)
Z = rand(cp, 100)

Plots.plot(Z.t, cumsum(Z.Δ[1, :]), seriestype = :scatter, xlabel = "Time", ylabel = "Jump Size", title = "Compound Poisson Process")
Plots.plot!(Z.t, cumsum(Z.Δ[2, :]), seriestype = :scatter)
Plots.plot!(Z.t, cumsum(Z.Δ[3, :]), seriestype = :scatter)
Plots.plot!(Z.t, cumsum(Z.Δ[4, :]), seriestype = :scatter)

ζ = 1
γ = -0.1

M = zeros(d, d)  # create an n x n matrix filled with zeros

for i in 1:d
    M[i, i] = -1.0  # set the diagonal elements to -1    
    if i > 1
        M[i, i-1] = ζ  # set the subdiagonal elements to ζ
        M[i-1, i] = γ  # set the superdiagonal elements to γ
    end
end

Random.seed!(123)

scale = 10
osd_cp = osd(scale * M, cp)
X = rand(osd_cp, 1000, 100)

Plots.plot(X[:, 1], X[:, 2], seriestype = :scatter, xlabel = "X_1", ylabel = "X_2", title = "OSD Distribution")
Plots.plot(X[:, 2], X[:, 3], seriestype = :scatter, xlabel = "X_1", ylabel = "X_2", title = "OSD Distribution")
Plots.plot(X[:, 5], X[:, 6], seriestype = :scatter, xlabel = "X_1", ylabel = "X_2", title = "OSD Distribution")

pairplot(
    DataFrame(X, :auto) => (
        PairPlots.Scatter(markersize = 4.0, alpha = 0.4),
        PairPlots.Contour(sigmas = 0.5:0.5:3, bandwidth = 3.0, color=:blue, alpha = 0.2)
        ), 
    fullgrid=true
)


Σ_hat = cov(X)
#heatmap(Σ_hat, title = "Empirical Covariance Matrix", xlabel = "i", ylabel = "j")
#moment(X, 2) |> Array
K_hat = cumulants(X, 3)[3] |> Array
#K6_hat = cumulants(X, 6)[6] |> Array
#K6_hat[CartesianIndex.(axes(K6_hat, 1), axes(K6_hat, 2), axes(K6_hat, 3), axes(K6_hat, 4), axes(K6_hat, 5), axes(K6_hat, 6))] # Diagonal elements


M * Σ_hat + Σ_hat * M'
diag((M * Σ_hat + Σ_hat * M')) 
λ * α * (α + 1) / ((α + β) * (α + β + 1))
# diag((M * Σ_hat + Σ_hat * M')) .+ λ * θ^2 * α * (α + 1)


kron_sum_M = kron(I(d), I(d), M) + kron(I(d), M, I(d)) + kron(M, I(d), I(d))

tmp = kron_sum_M * vec(K_hat) |> x -> reshape(x,d,d,d)
tmp[CartesianIndex.(axes(tmp, 1), axes(tmp, 2), axes(tmp, 3))] # Diagonal elements
λ * α * (α + 1) * (α + 2) / ((α + β) * (α + β + 1) * (α + β + 2))
# λ * θ^3 * α * (α + 1) * (α + 2) * [1,-1][p[1:3].+1]