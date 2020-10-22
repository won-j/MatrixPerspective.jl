include("gaussianmle.jl")
using Random, LinearAlgebra, SparseArrays

Random.seed!(123)
n, p = 60, 100
# data matrix
pcov = 0.3 * ones(p, p) + 0.7 * I  # covariance matrix
pchol = cholesky(pcov)
X = randn(n, p) * pchol.U   # correlated predictors
S = X' * X / n # second monent
mu = reshape(sum(X, dims=1) / n , p )   # sample mean

# Variance constraints
# cvec should be scaled so that
#  cvec[i]' * (Omega \ cvec[i]) <= 1
cvec = Array{Vector{Float64}, 1}()
m = 5
for i=1:m
	e = zeros(p)
	e[i] = 1.0
	push!(cvec, e)
end

maxit = 5000
@time eta, Omega, Y, history = gaussianmle(X, cvec, maxiter=maxit, opttol=1e-5)

