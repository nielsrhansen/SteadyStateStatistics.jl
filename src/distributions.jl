struct signed_gamma <: Sampleable{Univariate, Continuous}
    α::Float64
    θ::Float64
    p::Float64
end

function rand(x::signed_gamma)
    sign = rand(Bernoulli(x.p)) ? 1 : -1
    return sign * rand(Gamma(x.α, x.θ))
end
