struct osd
    M::Array{Float64, 2}
    cp::compound_poisson
end

function rand_naive(x::osd, n::Integer = 1; r::Integer = 100)
    d = size(x.M, 1)
    X = zeros(n, d)
    for i in 1:n
        Z = rand(x.cp, r)
        for j in 1:size(Z.Δ, 2)
            X[i, :] += expv(Z.t[j], x.M, Z.Δ[:, j])
        end 
    end 
    return X
end

function rand_eigen(x::osd, n::Integer = 1; ϵ::Float64 = 1e-8, r::Integer = 0)
    d = size(x.M, 1)
    M_eigen = eigen(x.M)
    Q = M_eigen.vectors
    Q_inv = inv(Q)
    Λ = M_eigen.values # Note, *vector* of eigenvalues
    X = zeros(ComplexF64, n, d)
    if (r == 0) 
        r = floor(Int, sum(x.cp.λ) * log(ϵ) / maximum(real(Λ)))
    end
    for i in 1:n
        Z = rand(x.cp, r)
        for j in 1:size(Z.Δ, 2)
            X[i, :] += Diagonal(exp.(Z.t[j] * Λ)) * (Q_inv * Z.Δ[:, j])
        end
        X[i, :] = Q * X[i, :]
    end 
    return real(X)
end

function rand(x::osd, n::Integer = 1; ϵ::Float64 = 1e-8, r::Integer = 0, method::Symbol = :eigen)
    if method === :eigen
        return rand_eigen(x, n; ϵ = ϵ, r = r)
    elseif method === :naive
        return rand_naive(x, n; r = r)
    else
        error("unknown method: $method (valid: :eigen, :naive)")
    end
end

"""
    filldiag!(A, v)

Fill the main diagonal of an N-dimensional array `A` with values from vector `v`,
i.e. sets `A[i,i,...,i] = v[i]` for `i = 1:min(size(A)..., length(v))`.
Returns `A`.
"""
function filldiag!(A::AbstractArray, v::AbstractVector)
    n = min(length(v), minimum(size(A)))
    @inbounds for i in 1:n
        A[ntuple(_ -> i, ndims(A))...] = v[i]
    end
    return A
end

function cumulants(x::osd, k::Int)
    d = size(x.M, 1)
    Ck = zeros(Float64, ntuple(_ -> d, k)...)
    filldiag!(Ck, cumulants(x.cp, k))
    if k == 1
        return - x.M \ Ck
    elseif k == 2
        kron_sum = kron(I(d), x.M) + kron(x.M, I(d))
        return - reshape(kron_sum \ vec(Ck), (d, d))
    elseif k == 3
        kron_sum = kron(I(d), I(d), x.M) + kron(I(d), x.M, I(d)) + kron(x.M, I(d), I(d))
        return - reshape(kron_sum \ vec(Ck), (d, d, d))
    elseif k == 4
        kron_sum = kron(I(d), I(d), I(d), x.M) + kron(I(d), I(d), x.M, I(d)) + kron(I(d), x.M, I(d), I(d)) + kron(x.M, I(d), I(d), I(d))
        return - reshape(kron_sum \ vec(Ck), (d, d, d, d))
    else
        error("cumulants of order > 4 currently not implemented for osd")
    end
end