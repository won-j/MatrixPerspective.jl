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

