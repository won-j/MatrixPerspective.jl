# MatrixPerspective.jl


```julia
versioninfo()
```

    Julia Version 1.5.1
    Commit 697e782ab8 (2020-08-25 20:08 UTC)
    Platform Info:
      OS: macOS (x86_64-apple-darwin19.5.0)
      CPU: Intel(R) Core(TM) i5-8279U CPU @ 2.40GHz
      WORD_SIZE: 64
      LIBM: libopenlibm
      LLVM: libLLVM-9.0.1 (ORCJIT, skylake)



```julia
import Pkg
Pkg.activate("..")
Pkg.status()
```

    [36m[1mProject [22m[39mMatrixPerspective v0.1.0
    [32m[1mStatus[22m[39m `~/Dropbox/Projects/CvxGMVQ/code-v1.5/MatrixPerspective/Project.toml`
     [90m [f65535da] [39m[37mConvex v0.13.7[39m
     [90m [a93c6f00] [39m[37mDataFrames v0.21.8[39m
     [90m [42fd0dbc] [39m[37mIterativeSolvers v0.8.4[39m
     [90m [6405355b] [39m[37mMosek v1.1.3[39m
     [90m [1ec41992] [39m[37mMosekTools v0.9.4[39m
     [90m [37e2e46d] [39m[37mLinearAlgebra[39m
     [90m [9a3f8284] [39m[37mRandom[39m
     [90m [10745b16] [39m[37mStatistics[39m


## Timing

The following code compares the three methods, interior point method (by using MOSEK, for $p < 100$), bisection, and semismooth Newton, for finding the unique root of the semismooth function
$$
    f(\mu) = 1 - \mathbf{e}^T(\bar{\mathbf{X}} - \mu \mathbf{e}\mathbf{e}^T)_{+}\mathbf{e}
$$
where $\mathbf{e}=(0, \dotsc, 0, 1)^T$ and
$$
    \bar{\mathbf{X}} = \begin{bmatrix} -\mathbf{X} & \frac{1}{\sqrt{2}}\mathbf{y} \\
                        \frac{1}{\sqrt{2}}\mathbf{y}^T & 1 \end{bmatrix}
$$
for given input $(\mathbf{X}, \mathbf{y})$. From this root the prox operator
$$
    \mathrm{prox}_{\phi}(\mathbf{X}, \mathbf{y}),
    \quad
    \phi(\boldsymbol{\Omega}, \boldsymbol{\eta}) = \begin{cases}
    \frac{1}{2}\boldsymbol{\eta}^T\boldsymbol{\Omega}^{\dagger}\boldsymbol{\eta}, &
    \boldsymbol{\Omega} \succeq \mathbf{0},~\boldsymbol{\eta} \in \mathcal{R}(\boldsymbol{\Omega}) \\
    \infty, & \text{otherwise}
    \end{cases}
$$
can be computed in a closed form.

The first table reports the mean of the performance measures, and the second table contains the standard deviation. The $p=5$ case (especially for MOSEK) should be ignored since there is an overhead of JIT compilation of the code.


```julia
;cat timing.jl
```

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
    



```julia
include("timing.jl")
```

    p = 5
    p = 10
    p = 30
    p = 50


    ┌ Warning: Problem status SLOW_PROGRESS; solution may be inaccurate.
    └ @ Convex /Users/jhwon/.julia/packages/Convex/aYxJA/src/solution.jl:252


    p = 100


    ┌ Warning: Problem status SLOW_PROGRESS; solution may be inaccurate.
    └ @ Convex /Users/jhwon/.julia/packages/Convex/aYxJA/src/solution.jl:252
    ┌ Warning: Problem status SLOW_PROGRESS; solution may be inaccurate.
    └ @ Convex /Users/jhwon/.julia/packages/Convex/aYxJA/src/solution.jl:252
    ┌ Warning: Problem status SLOW_PROGRESS; solution may be inaccurate.
    └ @ Convex /Users/jhwon/.julia/packages/Convex/aYxJA/src/solution.jl:252
    ┌ Warning: Problem status SLOW_PROGRESS; solution may be inaccurate.
    └ @ Convex /Users/jhwon/.julia/packages/Convex/aYxJA/src/solution.jl:252
    ┌ Warning: Problem status SLOW_PROGRESS; solution may be inaccurate.
    └ @ Convex /Users/jhwon/.julia/packages/Convex/aYxJA/src/solution.jl:252
    ┌ Warning: Problem status SLOW_PROGRESS; solution may be inaccurate.
    └ @ Convex /Users/jhwon/.julia/packages/Convex/aYxJA/src/solution.jl:252
    ┌ Warning: Problem status SLOW_PROGRESS; solution may be inaccurate.
    └ @ Convex /Users/jhwon/.julia/packages/Convex/aYxJA/src/solution.jl:252


    p = 500
    p = 1000
    p = 2000
    24×6 DataFrame
    │ Row │ p     │ Method    │ Iters_mean │ Secs_mean   │ KKT_mean    │ Obj_mean  │
    │     │ [90mInt64[39m │ [90mString[39m    │ [90mFloat64[39m    │ [90mFloat64[39m     │ [90mFloat64[39m     │ [90mFloat64[39m   │
    ├─────┼───────┼───────────┼────────────┼─────────────┼─────────────┼───────────┤
    │ 1   │ 5     │ MOSEK     │ NaN        │ 2.49323     │ 1.21816e-5  │ 6.53939   │
    │ 2   │ 5     │ Bisection │ 28.4       │ 0.157635    │ 5.97528e-9  │ 6.53939   │
    │ 3   │ 5     │ Newton    │ 4.2        │ 0.0943318   │ 5.58998e-12 │ 6.53939   │
    │ 4   │ 10    │ MOSEK     │ NaN        │ 0.00983312  │ 1.828e-5    │ 27.5118   │
    │ 5   │ 10    │ Bisection │ 29.4       │ 0.000295918 │ 4.56144e-9  │ 27.5118   │
    │ 6   │ 10    │ Newton    │ 4.9        │ 0.000165922 │ 2.34184e-10 │ 27.5118   │
    │ 7   │ 30    │ MOSEK     │ NaN        │ 0.102232    │ 4.92603e-6  │ 225.301   │
    │ 8   │ 30    │ Bisection │ 31.8       │ 0.001258    │ 5.44707e-9  │ 225.301   │
    │ 9   │ 30    │ Newton    │ 5.7        │ 0.000554437 │ 1.18062e-9  │ 225.301   │
    │ 10  │ 50    │ MOSEK     │ NaN        │ 0.657758    │ 7.47059e-6  │ 654.111   │
    │ 11  │ 50    │ Bisection │ 33.2       │ 0.00311099  │ 4.26987e-9  │ 654.111   │
    │ 12  │ 50    │ Newton    │ 5.8        │ 0.00112343  │ 5.79867e-10 │ 654.111   │
    │ 13  │ 100   │ MOSEK     │ NaN        │ 21.2737     │ 6.95149e-6  │ 2496.79   │
    │ 14  │ 100   │ Bisection │ 34.5       │ 0.0123491   │ 5.21449e-9  │ 2496.79   │
    │ 15  │ 100   │ Newton    │ 6.5        │ 0.00433008  │ 2.44491e-9  │ 2496.79   │
    │ 16  │ 500   │ MOSEK     │ NaN        │ NaN         │ NaN         │ NaN       │
    │ 17  │ 500   │ Bisection │ 36.9       │ 0.331822    │ 4.0035e-9   │ 62564.1   │
    │ 18  │ 500   │ Newton    │ 7.9        │ 0.109068    │ 9.81724e-10 │ 62564.1   │
    │ 19  │ 1000  │ MOSEK     │ NaN        │ NaN         │ NaN         │ NaN       │
    │ 20  │ 1000  │ Bisection │ 37.8       │ 2.19949     │ 4.43912e-9  │ 2.50248e5 │
    │ 21  │ 1000  │ Newton    │ 8.0        │ 0.740104    │ 3.05567e-12 │ 2.50248e5 │
    │ 22  │ 2000  │ MOSEK     │ NaN        │ NaN         │ NaN         │ NaN       │
    │ 23  │ 2000  │ Bisection │ 39.6       │ 14.6615     │ 5.84813e-9  │ 1.00103e6 │
    │ 24  │ 2000  │ Newton    │ 8.0        │ 4.32247     │ 2.88162e-9  │ 1.00103e6 │
    24×6 DataFrame
    │ Row │ p     │ Method    │ Iters_std │ Secs_std    │ KKT_std     │ Obj_std │
    │     │ [90mInt64[39m │ [90mString[39m    │ [90mFloat64[39m   │ [90mFloat64[39m     │ [90mFloat64[39m     │ [90mFloat64[39m │
    ├─────┼───────┼───────────┼───────────┼─────────────┼─────────────┼─────────┤
    │ 1   │ 5     │ MOSEK     │ NaN       │ 7.8583      │ 1.28047e-5  │ 3.22705 │
    │ 2   │ 5     │ Bisection │ 0.966092  │ 0.498075    │ 3.69496e-9  │ 3.22705 │
    │ 3   │ 5     │ Newton    │ 0.632456  │ 0.298139    │ 9.18807e-12 │ 3.22705 │
    │ 4   │ 10    │ MOSEK     │ NaN       │ 0.000569673 │ 1.72334e-5  │ 5.87331 │
    │ 5   │ 10    │ Bisection │ 1.95505   │ 3.38437e-5  │ 3.18761e-9  │ 5.87331 │
    │ 6   │ 10    │ Newton    │ 0.316228  │ 4.13268e-5  │ 4.37243e-10 │ 5.87331 │
    │ 7   │ 30    │ MOSEK     │ NaN       │ 0.012398    │ 7.51272e-6  │ 25.0831 │
    │ 8   │ 30    │ Bisection │ 1.0328    │ 4.06005e-5  │ 3.51572e-9  │ 25.0831 │
    │ 9   │ 30    │ Newton    │ 0.483046  │ 2.50481e-5  │ 3.14195e-9  │ 25.0831 │
    │ 10  │ 50    │ MOSEK     │ NaN       │ 0.110312    │ 9.48564e-6  │ 28.42   │
    │ 11  │ 50    │ Bisection │ 0.918937  │ 8.7696e-5   │ 2.88447e-9  │ 28.42   │
    │ 12  │ 50    │ Newton    │ 0.421637  │ 6.49869e-5  │ 1.49951e-9  │ 28.42   │
    │ 13  │ 100   │ MOSEK     │ NaN       │ 4.43358     │ 6.50369e-6  │ 60.6632 │
    │ 14  │ 100   │ Bisection │ 0.707107  │ 0.000243857 │ 3.36907e-9  │ 60.6632 │
    │ 15  │ 100   │ Newton    │ 0.527046  │ 0.000303794 │ 3.5232e-9   │ 60.6632 │
    │ 16  │ 500   │ MOSEK     │ NaN       │ NaN         │ NaN         │ NaN     │
    │ 17  │ 500   │ Bisection │ 0.875595  │ 0.00846495  │ 2.54059e-9  │ 436.492 │
    │ 18  │ 500   │ Newton    │ 0.316228  │ 0.00451969  │ 3.10448e-9  │ 436.492 │
    │ 19  │ 1000  │ MOSEK     │ NaN       │ NaN         │ NaN         │ NaN     │
    │ 20  │ 1000  │ Bisection │ 1.61933   │ 0.0865994   │ 3.93659e-9  │ 618.872 │
    │ 21  │ 1000  │ Newton    │ 0.0       │ 0.0137851   │ 2.3702e-12  │ 618.872 │
    │ 22  │ 2000  │ MOSEK     │ NaN       │ NaN         │ NaN         │ NaN     │
    │ 23  │ 2000  │ Bisection │ 0.966092  │ 0.378565    │ 3.164e-9    │ 1197.17 │
    │ 24  │ 2000  │ Newton    │ 0.0       │ 0.152881    │ 5.28822e-10 │ 1197.17 │


## PDHG
The following code illustrates how to use `prox_matrixperspective!()` for the PDHG algorithm for Gaussian joint likelihood estimation.


```julia
;cat gaussianmle.jl
```

    using MatrixPerspective
    
    using LinearAlgebra
    import LinearAlgebra.BLAS.BlasInt
    
    using IterativeSolvers
    
    # Joint MLE of multivariate Gaussian natural parameters
    # PDHG code based on http://proximity-operator.net/tutorial.html
    
    function gaussianmle(X::Matrix{T}, cvec::Array{Vector{T},1}; 
    					 ep::T = 10 / size(X)[2]^2, 
    					 init::Matrix{T} = Array{Float64}(undef, 0, 0), 
    					 maxiter::Integer = 1000, 
    					 opttol::T = 1e-4, 
    					 log::Bool = false
    					) where T <: Float64
    	n, p = size(X)  # sample size, dimension
    	m = length(cvec)
    
    	# workspaces
    	_n = p + 2
    	_Q = Matrix{Float64}(undef, _n, _n)
    	_evec = Vector{Float64}(undef, _n)
    
    	_d = Vector{Float64}(undef, _n)
    
    	_indxq = Vector{BlasInt}(undef, _n)
    
    	_z = Vector{Float64}(undef, _n)
    	_dlambda = Vector{Float64}(undef, _n)
    	_w = Vector{Float64}(undef, _n)
    	_Q2 = Vector{Float64}(undef, _n * _n)
    
    	_indx   = Vector{BlasInt}(undef, _n + 1)
    	_indxc  = Vector{BlasInt}(undef, _n)
    	_indxp  = Vector{BlasInt}(undef, _n)
    	_coltyp = Vector{BlasInt}(undef, _n)
    
    	_S = Vector{Float64}(undef, _n * _n )
    
    	_M = zeros(_n - 1, _n - 1) # for second derivative
    
    	_alph = Vector{Bool}(undef, _n - 1)
    	_gam  = similar(_alph)
    	_bet  = similar(_alph)
    
    	symmetrize!(A) = begin
        	for c in 1:size(A, 2)
        		for r in c:size(A, 1)
        			A[r, c] += A[c, r]
    				A[r, c] *= 0.5
        			A[c, r] =  A[r, c]
        		end
        	end
        	A
    	end
    
    	# select the step-sizes
    	L_f = 0.0 			    # Lipschitz modulus of $f$
    	L_K = m  				# |K'K|_2
    	tau = 2 / (L_f + 2)  
    	sigma = (1/tau - L_f/2) / L_K 
    
    	S = X' * X / n   # second moment
    	mu = reshape(sum(X, dims=1) / n, p) # sample mean
    
    	# initialize the primal solution
    	if isempty(init) 
    		#Omega = 1.0 * Matrix(I, p, p) # identity matrix
    		Omega = Matrix(Symmetric(inv(S - mu * mu' + 1e-2 * I)))
    	else 
    		Omega = init
    	end
    	Omega_old = similar(Omega)
    	#Omega_old = Omega
    	#eta     = zeros(p)
    	eta     = Omega * mu
    	#eta_old = similar(eta)
    	eta_old = similar(eta)
    	# initialize the dual solution
    	Y = [copy(Omega) for i=1:m+1]
    	zeta = copy(eta)
    
    	# execute the algorithm
    	history = ConvergenceHistory(partial = !log)
    	history[:tol] = opttol
    	IterativeSolvers.reserve!(T, history, :objval, maxiter)
    	IterativeSolvers.reserve!(Float64, history, :itertime, maxiter)
    	IterativeSolvers.nextiter!(history)
    	for it = 1:maxiter
    		tic = time()
    if it % 100 == 0 
    	println("it = ", it)
    end
        	# primal forward-backward step
    		copyto!(Omega_old, Omega)
    		copyto!(eta_old, eta)
    		## prox opertor of -logdet
    		M = (Omega - tau * (sum(Y) + S)) ./ ( 1 + ep * tau)
    		OmD, OmP = eigen!(Symmetric(M))
    		evals = 0.5 * (OmD + sqrt.(OmD.^2 .+ 4 * tau / (1 + ep * tau)))
    		fill!(Omega, zero(T))
    		@inbounds for k=1:length(evals)
           		@views pvec = OmP[:, k]
           		BLAS.gemm!('N', 'T', evals[k], pvec, pvec, 1.0, Omega)
        	end
    		## eta update
    		eta -= tau * (zeta - 2 * mu)
    
    		# primal adjustment
    		Omega_tilde = 2 * Omega - Omega_old
    		eta_tilde   = 2 * eta   - eta_old
        
        	# dual forward-backward step
    		for i=1:m
    			#Y_tilde = cvec[i] * cvec[i]'  - Y[i] ./ sigma - Omega_tilde
    			#getPSDpart!(Y_tilde)
    			#Y[i] = -sigma * Y_tilde
    			## less allocation using low-level matrix functions
    			#Y[i] .+= sigma * Omega_tilde
    			BLAS.axpy!(sigma, Omega_tilde, Y[i]) 
    			# Y[i] .= sigma * cvec[i] * cvex[i]' - Y[i]
    			BLAS.gemm!('N', 'T', sigma, cvec[i], cvec[i], -1.0, Y[i])
    			# ([Y[i])_+
    			getPSDpart!(Y[i])
    			# Y[i] .*= -1.0
    			rmul!(Y[i], -1.0)
    		end
    		# prox operator of $\phi^*$
    		M = Y[m + 1] + sigma * Omega_tilde
    		symmetrize!(M)
    		z = zeta + sigma * eta_tilde
    		# the following puts *primal* variables to Y[m + 1] and zeta
    		prox_matrixperspective!( Y[m + 1], zeta,
    									M, z, 1.0,
    						 			_Q, _evec, _d, 
    									_indxq, _z, _dlambda, _w, _Q2,
    									_indx, _indxc, _indxp, _coltyp, _S,
    									_M, _alph, _bet, _gam
    								 )
    		## restore dual variables
    		# Y[m + 1] .= M - Y[m + 1]
    		BLAS.axpy!(-1.0, M, Y[m + 1]) # Y[m + 1] .= -M + Y[m + 1]
    		rmul!(Y[m + 1], -1.0)         # Y[m + 1] .*= -1.0
    		# zeta .= z - zeta
    		BLAS.axpy!(-1.0, z, zeta)     # zeta .= -z + zeta
    		rmul!(zeta, -1.0)             # zeta .*= -1.0
    
        	# objective value
    		D, P = eigen(Symmetric(Omega))
    		objval = -sum(Base.log.(max.(D, 1e-15))) + tr(Omega * S) - 2 * dot(mu, eta) + (eta' * P) * (pinv(Diagonal(D)) * P' * eta) * 0.5 + 0.5 * ep * sum(abs2.(D))
    if it % 100 == 1
    	println("objval = ", objval)
    end
    		toc = time()
    		push!(history, :objval, objval)
    		push!(history, :itertime, toc - tic)
        	# stopping rule
        	if norm(vec(Omega - Omega_old)) < opttol * norm(vec(Omega_old)) && 
    		   norm(eta - eta_old) < opttol * norm(eta_old) && it > 10
    			IterativeSolvers.setconv(history, true)
            	break
        	end
    		IterativeSolvers.nextiter!(history)
    	end # end for
    	log && IterativeSolvers.shrink!(history)
    	eta, Omega, Y, history
    end # end of function
    


Function `gaussianmle()` is called as follows.


```julia
;cat testgaussianmle.jl
```

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
    



```julia
include("testgaussianmle.jl")
```

    objval = 824.1079588444118
    it = 100
    objval = -68.55345451618902
    it = 200
    objval = -90.7432608163429
    it = 300
    objval = -106.47412453177637
    it = 400
    objval = -119.48174383805825
    it = 500
    objval = -130.905231884441
    it = 600
    objval = -141.24124058414466
    it = 700
    objval = -150.75060287270733
    it = 800
    objval = -159.58819651238332
    it = 900
    objval = -167.8548291740341
    it = 1000
    objval = -175.62112227294827
    it = 1100
    objval = -182.93963829174476
    it = 1200
    objval = -189.85149972890522
    it = 1300
    objval = -196.39020970969847
    it = 1400
    objval = -202.58396017158287
    it = 1500
    objval = -208.45708168793593
    it = 1600
    objval = -214.03098642732363
    it = 1700
    objval = -219.32480171804784
    it = 1800
    objval = -224.3558092685453
    it = 1900
    objval = -229.13975914037536
    it = 2000
    objval = -233.6911010552277
    it = 2100
    objval = -238.02315987101295
    it = 2200
    objval = -242.1482724796969
    it = 2300
    objval = -246.07789742449864
    it = 2400
    objval = -249.82270476506264
    it = 2500
    objval = -253.39265129471005
    it = 2600
    objval = -256.7970446309472
    it = 2700
    objval = -260.0445986525942
    it = 2800
    objval = -263.14348205382316
    it = 2900
    objval = -266.1013613073703
    it = 3000
    objval = -268.92543899954114
    it = 3100
    objval = -271.62248826913253
    it = 3200
    objval = -274.1988839186122
    it = 3300
    objval = -276.660630647678
    it = 3400
    objval = -279.0133887725162
    it = 3500
    objval = -281.2624977290658
    it = 3600
    objval = -283.41299760919617
    it = 3700
    objval = -285.46964894025103
    it = 3800
    objval = -287.4369508881293
    it = 3900
    objval = -289.3191580396211
    it = 4000
    objval = -291.12029589988504
    it = 4100
    objval = -292.8441752243772
    it = 4200
    objval = -294.4944052907238
    it = 4300
    objval = -296.0744062042605
    it = 4400
    objval = -297.58742032087287
    it = 4500
    objval = -299.0365228620533
    it = 4600
    objval = -300.42463178951584
    it = 4700
    objval = -301.75451700000485
    it = 4800
    objval = -303.0288088951596
    it = 4900
    objval = -304.25000637601295
    it = 5000
    118.756086 seconds (13.64 M allocations: 17.783 GiB, 2.01% gc time)





    ([-116.11434098621383, 48.95213796995712, 90.65112808590104, 74.88200654160379, -109.65229977959252, -98.84490753993417, -30.09665095952194, 136.9517017393632, -104.99144451079381, -18.399580485564563  …  -49.91108946571656, 117.59080136676128, -68.00851618779222, -94.5335199879541, 74.52083217154109, 90.99154106488278, 108.42836157637981, 133.7807753285948, -48.671229175687735, -102.01022091106664], [21.33766714582228 -2.6238047353834855 … 0.6781649941396397 4.81120439765941; -2.6238047353834855 12.248940051450315 … -2.527585193176585 -3.467548444667807; … ; 0.6781649941396397 -2.527585193176585 … 17.906439248985176 3.2836072748514855; 4.81120439765941 -3.467548444667807 … 3.2836072748514855 21.367264541224216], [[-0.14575876261675827 -0.03563277907757659 … -0.03404245270484068 -0.06075941502196815; -0.03563277907757659 -0.008710933888275197 … -0.008322156244423087 -0.014853493357741087; … ; -0.03404245270484068 -0.008322156244423087 … -0.007950730133517705 -0.014190567175007886; -0.06075941502196815 -0.014853493357741087 … -0.014190567175007886 -0.025327509972888063], [-0.008689324794789004 -0.03553955835073861 … -0.006459451495274717 -0.01291964301956979; -0.03553955835073861 -0.14535769321489891 … -0.026419320114234028 -0.05284166696589258; … ; -0.006459451495274717 -0.026419320114234028 … -0.00480181309885309 -0.009604176318880721; -0.01291964301956979 -0.05284166696589258 … -0.009604176318880721 -0.01920945294313533], [-0.0007944588520018045 -0.0017130282012015343 … -0.0008562326215157404 -0.0015634040994131277; -0.0017130282012015343 -0.003693665959813736 … -0.001846226049026201 -0.00337104345356667; … ; -0.0008562326215157404 -0.001846226049026201 … -0.0009228096588016267 -0.0016849678081576932; -0.0015634040994131277 -0.00337104345356667 … -0.0016849678081576932 -0.0030766003448800656], [-0.0 -0.0 … -0.0 -0.0; -0.0 -0.0 … -0.0 -0.0; … ; -0.0 -0.0 … -0.0 -0.0; -0.0 -0.0 … -0.0 -0.0], [-0.03234378615574314 -0.020965129193548145 … -0.011242839572479554 -0.026884814875212406; -0.020965129193548145 -0.013589523501846373 … -0.007287569334164546 -0.017426643080354694; … ; -0.011242839572479554 -0.007287569334164546 … -0.0039080595278443535 -0.00934527760981235; -0.026884814875212406 -0.017426643080354694 … -0.00934527760981235 -0.02234720658224793], [-0.010322316049599323 0.01708816716554684 … 0.00295235387583076 -0.008294524720988461; 0.01708816716554684 -0.028288753771398945 … -0.004887499696737629 0.013731242514756659; … ; 0.00295235387583076 -0.004887499696737629 … -0.0008444222562311587 0.002372371867952583; -0.008294524720988461 0.013731242514756659 … 0.002372371867952583 -0.006665087565281169]], Not converged after 5001 iterations.)




```julia

```
