using MatrixPerspective

using LinearAlgebra
import LinearAlgebra.BLAS.BlasInt

using Random
using DataFrames, Statistics

Random.seed!(1234);

reps = 10 #reps = 100
tol = 1e-8   # default

# KKT measures |g'(mu)|
df = DataFrame(p=Int[], Method=String[], Iters=Float64[], Secs=Float64[], KKT=Float64[], Obj=Float64[])
# MOSEK stalls if p > 30
dims = [5, 10, 30, 50, 100, 500, 1000, 2000]  
#dims = [10, 30] #dims = [100] #dims = [1000]
nummethods = 2
Means = zeros(nummethods, length(dims), size(df)[2]);
mosek_maxdim = 100
for i = 1:length(dims)
	p = dims[i]
	n = p + 2

	e = zeros(p + 1)  # last elementary unit vector
	e[end] = 1

	# workspaces
	Q = Matrix{Float64}(undef, n, n)
	evec = Vector{Float64}(undef, n)

	d = Vector{Float64}(undef, n)

	indxq = Vector{BlasInt}(undef, n)

	z = Vector{Float64}(undef, n)
	dlambda = Vector{Float64}(undef, n)
	w = Vector{Float64}(undef, n)
	Q2 = Vector{Float64}(undef, n * n)

	indx   = Vector{BlasInt}(undef, n + 1)
	indxc  = Vector{BlasInt}(undef, n)
	indxp  = Vector{BlasInt}(undef, n)
	coltyp = Vector{BlasInt}(undef, n)

	S = Vector{Float64}(undef, n * n )

	M = zeros(n - 1, n - 1) # for second derivative

	alph = Vector{Bool}(undef, n - 1)
	gam  = similar(alph)
	bet  = similar(alph)

println("p = ", p)
	for rep=1:reps
		X = Matrix(Symmetric(randn(p, p)))
		y = randn(p)
		while !(minimum(eigvals(X + 0.5 * y * y')) < 0)
			X = Diagonal(randn(p))
			y = randn(p)
		end
		G = [X  -y/sqrt(2); -y'/sqrt(2)  1]

		if p <= mosek_maxdim
			secs = @elapsed optval, primal, dual = mosek_ls_cov_adj(Matrix(G))
			mu = dual[2].dual

			lam2, P2 = eigen(G + mu * e * e')
			Lam2 = P2 * Diagonal(max.(lam2, 0)) * P2'
			mosek_deriv1 = 1 - Lam2[end, end]
		else
			(optval, mu) = (NaN, NaN) # if MOSEK stalls
			mosek_deriv1 = NaN
			secs = NaN
		end
		push!(df, [p, "MOSEK", NaN, secs, abs(mosek_deriv1), optval])

		secs = @elapsed nu, C, convhist = bisect!(Matrix(G), Q, evec, d,
										indxq, z, dlambda, w, Q2,
										indx, indxc, indxp, coltyp, S,
										M, alph, bet, gam; maxiter=50)

		# reconstruct the dual optimal value
		dualval = -0.5 * norm(C, 2)^2 + nu + 0.5 * norm(G, 2)^2

		deriv1 = 1 - C[end, end]

		push!(df, [p, "Bisection", convhist.iters, secs, abs(deriv1), dualval])

		secs = @elapsed nu, C, convhist = dual_ls_cov_adj!(Matrix(G), Q, evec, d,
										indxq, z, dlambda, w, Q2,
										indx, indxc, indxp, coltyp, S,
										M, alph, bet, gam; maxiter=50)

		# reconstruct the dual optimal value
		dualval = -0.5 * norm(C, 2)^2 + nu + 0.5 * norm(G, 2)^2

		deriv1 = 1 - C[end, end]

		push!(df, [p, "Newton", convhist.iters, secs, abs(deriv1), dualval])
	end
end
gdf = groupby(df, [:p, :Method])
mdf = combine(gdf, names(gdf)[3:end] .=> mean)
println(mdf)
sdf = combine(gdf, names(gdf)[3:end] .=> std)
println(sdf)

