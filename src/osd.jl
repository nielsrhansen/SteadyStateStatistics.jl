struct osd
    M::Array{Float64, 2}
    cp::compound_poisson
end

function rand_naive(x::osd, n::Integer = 1, r::Integer = 100)
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

function rand_eigen(x::osd, n::Integer = 1, r::Integer = 100)
    d = size(x.M, 1)
    M_eigen = eigen(x.M)
    Q = M_eigen.vectors
    Q_inv = inv(Q)
    Λ = M_eigen.values # Note, *vector* of eigenvalues
    X = zeros(ComplexF64, n, d)
    for i in 1:n
        Z = rand(x.cp, r)
        for j in 1:size(Z.Δ, 2)
            X[i, :] += Diagonal(exp.(Z.t[j] * Λ)) * (Q_inv * Z.Δ[:, j])
        end
        X[i, :] = Q * X[i, :]
    end 
    return real(X)
end

function rand(x::osd, n::Integer = 1, r::Integer = 100; method::Symbol = :eigen)
    if method === :eigen
        return rand_eigen(x, n, r)
    elseif method === :naive
        return rand_naive(x, n, r)
    else
        error("unknown method: $method (valid: :eigen, :naive)")
    end
end
