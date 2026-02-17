function cum_contraint(M::Array{Float64,2}, X::Array{Float64,2})
    d = size(M, 1)
    C3 = zeros((d, d, d))

    K_hat = cumulants(X, 3) 

    K2_hat = K_hat[2] |> Array
    C2 = M * K2_hat + K2_hat * M'
    mask = [!(i == j) for i in 1:d, j in 1:d]
    C2_vec = vec(C2)[vec(mask)]

    K3_hat = K_hat[3] |> Array
    @tullio C3[i,j,k] := M[i,l] * K3_hat[l,j,k] + M[j,l] * K3_hat[i,l,k] + M[k,l] * K3_hat[i,j,l]
    mask = [!(i == j == k) for i in 1:d, j in 1:d, k in 1:d]
    C3_vec = vec(C3)[vec(mask)]

    return (C2_vec, C3_vec)
end

function contractor_matrix_2(K2::Array{Float64,2}, mode::Int)
    d = size(K2, 1)
    A = reshape(zeros(d^2, d^2), d, d, d, d, )
    for i in 1:d
        for j in 1:d
                for α in 1:d
                    for β in 1:d
                        if mode == 1 && α == i
                            A[i, j, α, β] = K2[β, j]
                        elseif mode == 2 && α == j
                            A[i, j, α, β] = K2[i, β]
                    end
                end
            end
        end
    end
    return reshape(A, d^2, d^2)
end

function contractor_matrix_3(K3::Array{Float64,3}, mode::Int)
    d = size(K3, 1)
    A = reshape(zeros(d^3, d^2), d, d, d, d, d)
    for i in 1:d
        for j in 1:d
            for k in 1:d
                for α in 1:d
                    for β in 1:d
                        if mode == 1 && α == i
                            A[i, j, k, α, β] = K3[β, j, k]
                        elseif mode == 2 && α == j
                            A[i, j, k, α, β] = K3[i, β, k]
                        elseif mode == 3 && α == k
                            A[i, j, k, α, β] = K3[i, j, β]
                        end
                    end
                end
            end
        end
    end
    return reshape(A, d^3, d^2)
end

function contractor_matrix_4(K4::Array{Float64,4}, mode::Int)
    d = size(K4, 1)
    A = reshape(zeros(d^4, d^2), d, d, d, d, d, d)
    for i in 1:d
        for j in 1:d
            for k in 1:d
                for l in 1:d
                    for α in 1:d
                        for β in 1:d
                            if mode == 1 && α == i
                                A[i, j, k, l, α, β] = K4[β, j, k, l]
                            elseif mode == 2 && α == j
                                A[i, j, k, l, α, β] = K4[i, β, k, l]
                            elseif mode == 3 && α == k
                                A[i, j, k, l, α, β] = K4[i, j, β, l]
                            elseif mode == 4 && α == l
                                A[i, j, k, l, α, β] = K4[i, j, k, β]
                            end
                        end
                    end
                end
            end
        end
    end
    return reshape(A, d^4, d^2)
end

function contractor_half_A(K3::Array{Float64,3}, mode::Int)
    d = size(K3, 1)
    A = reshape(zeros(d^3, d^2), d, d, d, d, d)
    for i in 1:d
        for j in 1:d
            for k in 1:d
                for α in 1:d
                    if mode == 1 && α == i
                            A[i, j, k, α, α] = K3[α, j, k]
                        elseif mode == 2 && α == j
                            A[i, j, k, α, α] = K3[i, α, k]
                        elseif mode == 3 && α == k
                            A[i, j, k, α, α] = K3[i, j, α]
                    end
                    if α < d 
                        for β in (α + 1):d
                            if mode == 1 && α == i
                                A[i, j, k, α, β] = K3[β, j, k]
                            elseif mode == 2 && α == j
                                A[i, j, k, α, β] = K3[i, β, k]
                            elseif mode == 3 && α == k
                                A[i, j, k, α, β] = K3[i, j, β]
                            end 
                            if mode == 1 && β == i
                                A[i, j, k, α, β] -= K3[α, j, k]
                            elseif mode == 2 && β == j
                                A[i, j, k, α, β] -= K3[i, α, k]
                            elseif mode == 3 && β == k
                                A[i, j, k, α, β] -= K3[i, j, α]
                            end 
                        end
                    end
                end
            end
        end
    end
    mask = [α <= β for α in 1:d, β in 1:d] |> vec
    mask3 = [i <= j <= k && !(i == j == k) for i in 1:d, j in 1:d, k in 1:d]
    mask3_vec = mask3 |> vec
    return reshape(A, d^3, d^2)[mask3_vec, mask]
end


function construct_full_A(K2::Array{Float64,2}, K3::Array{Float64,3})
    d = size(K2, 1)

    mask_diag = [i == j for i in 1:d, j in 1:d]
    mask_diag_vec = mask_diag |> vec

    mask2 = [i < j for i in 1:d, j in 1:d]
    mask2_vec = mask2 |> vec
    A2 = contractor_matrix_2(K2, 1) + contractor_matrix_2(K2, 2)

    mask3 = [i <= j <= k && !(i == j == k) for i in 1:d, j in 1:d, k in 1:d]
    mask3_vec = mask3 |> vec

    A3 = contractor_matrix_3(K3, 1) + contractor_matrix_3(K3, 2) + contractor_matrix_3(K3, 3)

    return [mask_diag_vec'; A2[mask2_vec, :]; A3[mask3_vec, :]] 
end 


function construct_full_A3(K2::Array{Float64,2}, K3::Array{Float64,3})
    d = size(K2, 1)

    mask2 = [i < j for i in 1:d, j in 1:d]
    mask2_vec = mask2 |> vec
    A2 = contractor_matrix_2(K2, 1) + contractor_matrix_2(K2, 2)

    mask3 = [i <= j <= k && !(i == j == k) for i in 1:d, j in 1:d, k in 1:d]
    mask3_vec = mask3 |> vec

    A3 = contractor_matrix_3(K3, 1) + contractor_matrix_3(K3, 2) + contractor_matrix_3(K3, 3)

    return [A2[mask2_vec, :]; A3[mask3_vec, :]] 
end 


function construct_full_A4(K2::Array{Float64,2}, K3::Array{Float64,3}, K4::Array{Float64,4})
    d = size(K2, 1)

    A23 = construct_full_A3(K2, K3)

    mask4 = [i <= j <= k <= l && !(i == j == k == l) for i in 1:d, j in 1:d, k in 1:d, l in 1:d]
    mask4_vec = mask4 |> vec

    A4 = contractor_matrix_4(K4, 1) + contractor_matrix_4(K4, 2) + contractor_matrix_4(K4, 3) + contractor_matrix_4(K4, 4)

    return [A23; A4[mask4_vec, :]] 
end 

function linear_estimator(X::Array{Float64,2})
    d = size(X, 2)
    K_hat = cumulants(X, 3) 
    K2_hat = K_hat[2] |> Array
    K3_hat = K_hat[3] |> Array

    A = construct_full_A(K2_hat, K3_hat)
    b = zeros(size(A, 1))
    b[1] = - 1.0    

    return reshape(A \ b, d, d)
end

function quadratic_estimator(X::Array{Float64,2})
    d = size(X, 2)
    K_hat = cumulants(X, 3) 
    K2_hat = K_hat[2] |> Array
    K3_hat = K_hat[3] |> Array

    A = construct_full_A3(K2_hat, K3_hat)
    K = A' * A
    λs, vs = eigen(Symmetric(K))
    v_min = vs[:, argmin(λs)]
    M_hat = reshape(v_min, d, d)
    sig = - sign(tr(M_hat))

    return sig .* M_hat
end

function quadratic_estimator_4(X::Array{Float64,2})
    d = size(X, 2)
    K_hat = cumulants(X, 4) 
    K2_hat = K_hat[2] |> Array
    K3_hat = K_hat[3] |> Array
    K4_hat = K_hat[4] |> Array

    A = construct_full_A4(K2_hat, K3_hat, K4_hat)
    K = A' * A
    λs, vs = eigen(Symmetric(K))
    v_min = vs[:, argmin(λs)]
    M_hat = reshape(v_min, d, d)
    sig = - sign(tr(M_hat))

    return sig .* M_hat
end

function two_step_estimator(X::Array{Float64,2})
    d = size(X, 2)
    K_hat = cumulants(X, 3) 
    K2_hat_inv = K_hat[2] |> Array |> inv
    K3_hat = K_hat[3] |> Array

    K3_K1 = zeros((d, d, d))
    K3_K2 = zeros((d, d, d))
    K3_K3 = zeros((d, d, d))
    @tullio K3_K1[i, j, k] := K2_hat_inv[i, l] * K3_hat[l, j, k] 
    @tullio K3_K2[i, j, k] := K2_hat_inv[j, l] * K3_hat[i, l, k] 
    @tullio K3_K3[i, j, k] := K2_hat_inv[k, l] * K3_hat[i, j, l]

    A_half = contractor_half_A(K3_K1, 1) + contractor_half_A(K3_K2, 2) + contractor_half_A(K3_K3, 3)

    K = A_half' * A_half
    λs, vs = eigen(Symmetric(K))
    v_min = vs[:, argmin(λs)]
    S_hat = zeros((d, d))
    mask = [α <= β for α in 1:d, β in 1:d]
    S_hat[mask] = v_min
    sig = - sign(tr(S_hat))
    S_hat = sig .* S_hat 

    masku = [α < β for α in 1:d, β in 1:d]
    maskl = [α > β for α in 1:d, β in 1:d]
    S_hat[maskl] = -S_hat[masku]

    return S_hat * K2_hat_inv
end

function two_step_shrinkage_estimator(X::Array{Float64,2})
    d = size(X, 2)
    K_hat = cumulants(X, 3) 
    K2_hat_inv = K_hat[2] |> Array |> inv
    K3_hat = K_hat[3] |> Array

    K3_K1 = zeros((d, d, d))
    K3_K2 = zeros((d, d, d))
    K3_K3 = zeros((d, d, d))
    @tullio K3_K1[i, j, k] := K2_hat_inv[i, l] * K3_hat[l, j, k] 
    @tullio K3_K2[i, j, k] := K2_hat_inv[j, l] * K3_hat[i, l, k] 
    @tullio K3_K3[i, j, k] := K2_hat_inv[k, l] * K3_hat[i, j, l]

    A_half = contractor_half_A(K3_K1, 1) + contractor_half_A(K3_K2, 2) + contractor_half_A(K3_K3, 3)

    maskd = [α == β for α in 1:d, β in 1:d] |> vec
    mask = [α <= β for α in 1:d, β in 1:d] |> vec
    col_index = findall(maskd[mask])

    A_top = zeros((d, size(A_half, 2)))
    for i in 1:d
        A_top[i, col_index[i]] = 1 
    end

    A = [A_top; A_half]
    b = zeros(d + size(A_half, 1))
    b[1:d] .= - 1
    s_hat = A \ b

    S_hat = zeros((d, d))
    mask = [α <= β for α in 1:d, β in 1:d]
    S_hat[mask] = s_hat

    masku = [α < β for α in 1:d, β in 1:d]
    maskl = [α > β for α in 1:d, β in 1:d]
    S_hat[maskl] = -S_hat[masku]

    return S_hat * K2_hat_inv
end