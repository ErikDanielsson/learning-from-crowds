using LinearAlgebra

sigmoid(x) = 1 / (1 + exp(-x))
ξ(x, w, b) = sigmoid(w ⋅ x + b)
∂ξ(x, w, b) = ξ(x, w, b) * (1 - ξ(x, w, b))

function yit_estimate(yit, xi, wt, γt)
    return 1 - ξ(xi, wt, γt) ? yit == 0 : ξ(xi, wt, γt)
end

function p_tilde(xi, yi, w, γ, α, β)
    p = 1
    z_factor = ξ(xi, α, β)
    for t in eachindex(yi)
        wt = w[t, :]
        γt = γ[t, :]
        yit = y[t]
        p *= yit_estimate(yit, xi, wt, γt) * z_factor
    end
    return p
end

function Δp_tilde(xi, yi, w, γ, α, β)
    p = p_tilde(xi, yi, w, γ, α, β)
    return 2p - 1
end

# Since the gradient expressions for α and β are nearly identical, 
# we bundle them together by adding an extra column to s
function ∂f∂αβ(x, yit, wt, γt, α, β)
    s = zeros(α_length + 1)
    for i in axes(x, 1)
        xi = ones(α_length + 1)
        xi[1:α_length] = x[i, :]
        p = Δp(xi, yit, wt, γt, α, β)
        s += p * ∂ξ(xi, α, β) * xi
    return s
end

function ∂f∂ηt(xi, yit, wt, γt, α, β)
    return ((-1) ? yit == 1 : 1) * (1 - Δp(xi, yit, wt, γt, α, β))
end

# In the same manner as above, we bundle together the gradient expressions for w and γ
function ∂f∂wγt(x, yt, wt, γt, α, β)
    s = zeros(w_length + 1)
    for i in axes(x, 1)
        xi = ones(w_length + 1)
        xi[1:w_length] = x[i, :]
        yit = yt[i]
        s += ∂f∂η(xi, yit, wt, γt, α, β) * ∂ξ(xi, wt, γt) * xi
    end
    return s
end

function f(x, y, w, γ, α, β) 
    N, D = size(x)
    N, T = size(y)
    l = 0
    for i in 1:N
        xi = x[i, :]
        yi = y[i, :]
        classifier_eval = ξ(xi, w, b)
        zprob_eval = p_tilde(xi, yi, w, γ, α, β) 
        for t in 1:T
            wt = w[t, :]
            γt = γ[t, :]
            yit = yi[t]
            yit_eval = yit_estimate(yit, xi, wt, γt)
            # Contribution corresponding to z = 1
            l += (log(yit_eval) + log(classifier_eval)) * zprob_eval
            # Contributions corresponding to z = 0
            l += (log(1 - yit_eval) + log(1 - classifier_eval)) * (1 - zprob_eval)
        end
    end
    return l
end

α_length = 1
wt_length = 1

function gradient!(storage, x)
    # Compute indicies for the differenr parameters
    # α

end



function EM(X, Y, ϵ)
    N, D = size(X)
    _, T = size(Y)
    α = zeros(D)
    β = 0
    while true
        p = 1
    end
    return 1
end
