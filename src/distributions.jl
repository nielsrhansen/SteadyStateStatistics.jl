struct signed_gamma <: Sampleable{Univariate, Continuous}
    α::Float64
    θ::Float64
    p::Float64
end

function rand(x::signed_gamma)
    sign = rand(Bernoulli(x.p)) ? 1 : -1
    return sign * rand(Gamma(x.α, x.θ))
end

function raw_moments(x::Sampleable, k::Int)
    if x isa Beta
        return prod((x.α + i) / (x.α + x.β + i) for i in 0:k-1)
    elseif x isa Gamma
        return prod((x.α + i) * x.θ for i in 0:k-1)
    else
        error("raw_moments not implemented for type $(typeof(x))")
    end
end