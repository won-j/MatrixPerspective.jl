using LinearAlgebra
using IterativeSolvers
using Convex, Mosek, MosekTools

"""
	rank1update!(M::Matrix{T}, A::Matrix{T}, v::Vector{T}, alph::T) 

Rank-1 update of symmetric matrix `A`:
	`M := A + alph * u * u'`
Only the upper triangular part of `A` is used.
"""
function rank1update!( M::Matrix{T}, A::Matrix{T}, 
					   v::Vector{T}, alph::T 
					 ) where T <: AbstractFloat
	M .= A  # overwrite
	BLAS.syr!('U', alph, v, M) # M += alph * v * v'
end

"""
	getPSDpart!(M::Matrix{T})

Get the positive semidefinite part of matrix `M` and overwrite it.
Returns the eigenvalues and the last elements of the eigenvectors of `M` 
(before overwritten) 
"""
function getPSDpart!(M::Matrix{T}) where T <: AbstractFloat
   	lam, P = eigen!(Symmetric(M))  # M = P * diag(lam) * P'
	#M .= P * Diagonal(max.(lam, 0)) * P'
	fill!(M, zero(T))
	@inbounds for k=1:length(lam)
       @views pvec = P[:, k]
       BLAS.gemm!('N', 'T', max(lam[k], zero(T)), pvec, pvec, 1.0, M)
    end
	Pte = P[end, :]  # P' * e
	lam, Pte
end

"""
	getRank1UpdatePSDPart!(valG::AbstractVector{T}, 
							vecG::AbstractMatrix{T}, 
							y::T, 
							v::AbstractVector{T}, 
							d::AbstractVector{T}, 
							Q::AbstractMatrix{T}, 
							M::AbstractMatrix{T},
							indxq::AbstractVector{BlasInt},
							z::AbstractVector{T},
							dlambda::AbstractVector{T},
							w::AbstractVector{T},
							Q2::AbstractVector{T},
							indx::AbstractVector{BlasInt},
							indxc::AbstractVector{BlasInt},
							indxp::AbstractVector{BlasInt},
							coltyp::AbstractVector{BlasInt},
							S::AbstractVector{T},
						) where T <: AbstractFloat

First compute the rank-1 update of symmetric matrix `G`:
```
	M := G + y * w * w'
```
when the eigenvalue decomposition `(valG, vecG)` of `G` is given,
and then compute the positive semidefinite part of matrix `M` and overwrite. Finally return the eigenvalues and the last elements of the eigenvectors of `M` 

# Input
- `valG::AbstractVector{T}`: eigenvalues of `G`. Length `n - 1`.
- `vecG::AbstractMatrix{T}`: eigenvectors of `G`. Order `n - 1`.
- `y::T`: scalar for rank-one update
- `v::AbstractVector{T}`: updating vector. It is assumed that the first element of `v` is zero.
- `d::AbstractVector{T}`, `Q::AbstractMatrix{T}`, `M::AbstractMatrix{T}`, `indxq::AbstractVector{BlasInt}`, `z::AbstractVector{T}`, `dlambda::AbstractVector{T}`, `w::AbstractVector{T}`, `Q2::AbstractVector{T}`, `indx::AbstractVector{BlasInt}`, `indxc::AbstractVector{BlasInt}`, `indxp::AbstractVector{BlasInt}`, `coltyp::AbstractVector{BlasInt}`, `S::AbstractVector{T}`: workspaces. In particular, `Q` is a matrix of order `n` and `M` is a matrix of order `n - 1`.

# Output
- On return, `M` contains the computed rank-one update of `G`.
- `lam::view(::Vector{T}, 2:n)`: eigenvalues of `M`. A view of length-`n` vector `d`.
- `Pte::view(::Matrix{T}, n, 2:n)`: if `P` is the eigenvector matrix of `M`, `Pte` is the vector consisting of the last row of `P`, i.e., `P' * e` where `e = [0; 0; ...; 1]`.
"""
function getRank1UpdatePSDPart!(valG::AbstractVector{T}, 
								vecG::AbstractMatrix{T}, 
								y::T, 
								v::AbstractVector{T}, 
								d::AbstractVector{T}, 
								Q::AbstractMatrix{T}, 
								M::AbstractMatrix{T},
								indxq::AbstractVector{BlasInt},
								z::AbstractVector{T},
								dlambda::AbstractVector{T},
								w::AbstractVector{T},
								Q2::AbstractVector{T},
								indx::AbstractVector{BlasInt},
								indxc::AbstractVector{BlasInt},
								indxp::AbstractVector{BlasInt},
								coltyp::AbstractVector{BlasInt},
								S::AbstractVector{T},
								) where T <: AbstractFloat
	n = size(Q)[1]
	Q[1, 1] = 1.0
	#Q[2:end, 2:end] .= vecG
	for i=2:n
		for j=2:n
			@inbounds Q[i, j] = vecG[i - 1, j - 1]
		end
	end
	Q[1, 2:end] .= 0.0
	Q[2:end, 1] .= 0.0
	#d .= [1.0; valG]
	d[1] = 1.0
	for i=2:n
		d[i] = valG[i - 1]
	end

	k = rank1eig!(d, Q, y, v, indxq, z, dlambda, w, Q2,
				 indx, indxc, indxp, coltyp, S)
	#= 	k eigenvalues are non-deflated. 
		The initial, artificial eigenvalue must have been deflated
		The corresponding artificial eigenvector is [1 0 ... 0]
		and unchanged.
	=#
	# find the permuted location of the artificial eigenvalue in INDX
	indxartf = -1
	for i=(k + 1):n
		@inbounds if indx[i] == 1
			indxartf = i	# caution: scope rule
			break
		end
	end
	# Move the artificial eigenvector to the first column
	#Q[:, indxartf] .= Q[:, 1]
	for i=1:n
		@inbounds Q[i, indxartf] = Q[i, 1]
	end
	Q[1, 1] = 1.0
	Q[2:end, 1] .= 0.0
	d1 = d[1]
	d[1] = d[indxartf]
	d[indxartf] = d1
	P = view(Q, 2:n, 2:n)
	lam = view(d, 2:n)
	Pte = view(Q, n, 2:n)  #Pte = P[end, :]  # P' * e

	##M .= P * Diagonal(max.(lam, 0)) * P'
	#P2 .= Diagonal(max.(lam, 0.0)) * P'
	# reuse Q2
	P2 = reshape(view(Q2, 1:((n - 1) * (n - 1))), n - 1, n - 1)
	for i=1:(n - 1)
		for j=1:(n - 1)
			@inbounds P2[j, i] = max(lam[j], 0.0) * P[i, j]
		end
	end
	#BLAS.gemm!('N', 'N', 1.0, P, P2, 0.0, M) # M .= P * P2
	mul!(M, P, P2) #M .= P * P2 

	lam, Pte
end

"""
	dual_ls_cov_adj!(G::Matrix{T},
					 Q::Matrix{T}, evec::Vector{T},
					 d::Vector{T}, indxq::Vector{BlasInt},
					 z::Vector{T}, dlambda::Vector{T},
					 w::Vector{T}, Q2::Vector{T},
					 indx::Vector{BlasInt},
					 indxc::Vector{BlasInt},
					 indxp::Vector{BlasInt},
					 coltyp::Vector{BlasInt},
					 S::Vector{T}, M::Matrix{T}, 
					 alph::Vector{Bool}, 
					 bet::Vector{Bool},
					 gam::Vector{Bool}; 
					 tol::T=convert(T, 1e-8),
					 maxiter::Integer=100,
					 log::Bool = false
					) where T <: AbstractFloat

Solve the optimization problem
```
        min_X (1/2)|| G - X ||_F^2
        s.t.  e'Xe = 1, X >= 0
```
via dual nonsmooth Newton method.

# Input
- `G::Matrix{T}`: input symmetric matrix
- `Q::Matrix{T}`, `evec::Vector{T}`, `d::Matrix{T}`, `indxq::Vector{BlasInt}`, `z::Vector{T}`, `dlambda::Vector{T}`, `w::Vector{T}`, `Q2::Vector{T}`, `indx::Vector{BlasInt}`, `indxc::Vector{BlasInt}`, `indxp::Vector{BlasInt}`, `coltyp::Vector{BlasInt}`, `S::Vector{T}`, `M::Matrix{T}`, `alph::Vector{Bool}`, `bet::Vector{Bool}`, `gam::Vector{Bool}`: workspaces. 

# Keyword arguments
- `tol::T`: tolerance for convergence. Default is 1e-8.
- `maxiter::Integer`: maximum number of Newtwon iteration. Default is 100.
- `log::Bool`: whether to record iterate history or not. Defaut is `false`.

# Output
- `y::T`: dual variable. `-mu` in the paper. 
- `C::Matrix{T}`: dual variable. `C(mu)` in the paper. 
- `history::ConvergenceHistory`: convergence history. Valid only if `log == false`.
"""
function dual_ls_cov_adj!(G::Matrix{T},
						 Q::Matrix{T}, evec::Vector{T},
						 d::Vector{T}, indxq::Vector{BlasInt},
						 z::Vector{T}, dlambda::Vector{T},
						 w::Vector{T}, Q2::Vector{T},
						 indx::Vector{BlasInt},
						 indxc::Vector{BlasInt},
						 indxp::Vector{BlasInt},
						 coltyp::Vector{BlasInt},
						 S::Vector{T}, M::Matrix{T}, 
						 alph::Vector{Bool}, 
						 bet::Vector{Bool},
						 gam::Vector{Bool}; 
						 tol::T=convert(T, 1e-8),
						 maxiter::Integer=100,
						 log::Bool = false
						) where T <: AbstractFloat
	# lower and upper bounds of the dual variable mu
	# note y == -mu
	#lb = zero(T) # lb = G[end, end] - one(T)  
	lb = 0.0
	ub = 0.5 * norm(G, 2)^2   # Frobenius norm squared

	history = ConvergenceHistory(partial = !log)
	history[:tol] = tol
	IterativeSolvers.reserve!(T, history, :objval, maxiter)
	IterativeSolvers.reserve!(Float64, history, :itertime, maxiter)
	IterativeSolvers.nextiter!(history)

	C = copy(G)
	e = zeros(size(G)[1])
	e[end] = one(T)   # elementary unit vector

	# if G - lb * e * e' >= 0 we are done (lb = 0.0)
	BLAS.syr!('U', -lb, e, C) # C -= lb * e * e'
    tau = minimum(eigvals(Symmetric(C)))
	if (tau > -tol) 
		push!(history, :objval, zero(T))
		push!(history, :itertime, 0.0)
		IterativeSolvers.setconv(history, true)
		log && IterativeSolvers.shrink!(history)
		return lb, C, 0
	end

	#
	# Otherwise we do Newton
	#
	valG, vecG = eigen(G)  # eigendecomposition for reuse
	perm = sortperm(valG)
	valG = valG[perm]
	vecG = vecG[:, perm]
	n = size(G)[1] + 1

	#yinit = - 0.5 * (lb + ub)
	yinit = - lb
	y = yinit
	(l, u) = (-ub, -lb)
	# for reuse
	P = reshape(view(Q2, 1:((n - 1) * (n - 1))), n - 1, n - 1)
	for iter = 1:maxiter # Newton iterations
		tic = time()
		## first derivative
		evec[1] = 0.0
		evec[2:end] .= e
		lam, Pte = getRank1UpdatePSDPart!(valG, vecG, y, evec, 
										  d, Q, C,
										  indxq, z, dlambda, w, Q2,
				 						  indx, indxc, indxp, coltyp,
										  S )
		deriv1 = C[end, end] - one(T)
		if ( abs(deriv1) < tol )  # converged
			toc = time()
			obj = 0.5 * norm(C, 2)^2 - y - 0.5 * norm(G, 2)^2
			push!(history, :objval, obj)
			push!(history, :itertime, toc - tic)
			IterativeSolvers.setconv(history, true)
			break
		end
		if deriv1 > zero(T)
			u = y
		else
			l = y
		end
		# second derivative
		alph .= (lam .> 0.0)
		gam  .= (lam .< 0.0)
		bet  .= .~alph .& .~gam
		if all(bet .== false)   # no multiple zero eigenvalues
			fill!(M, 0.0)
			M[alph, alph] .= 1.0
			M[alph, bet]  .= 1.0
			M[bet, alph]  .= 1.0
			lam_alph = view(lam, alph)
			lam_gam  = view(lam, gam)
			MV = view(M, alph, gam)
			#MV .= [lam_alph[i] / (lam_alph[i] - lam_gam[j])  for i in 1:size(MV)[1], j in 1:size(MV)[2] ]
			MW = view(M, gam, alph)
			#MW .= MV'
			for i=1:size(MV)[1]
				for j=1:size(MV)[2]
					@inbounds MV[i, j] = lam_alph[i] / (lam_alph[i] - lam_gam[j])  
					@inbounds MW[j, i] = MV[i, j]
				end
			end

			#deriv2 = dot(Pte, (M .* (Pte * Pte')) * Pte )
			#M .*= Pte * Pte' # memory consuming but faster
			# reuse Q2
			for i=1:(n - 1)
				for j=1:(n - 1)
					@inbounds P[i, j] = Pte[i] * Pte[j]
				end
			end
			M .*= P
			deriv2 = dot(Pte, M * Pte )
			if ( deriv2 > zero(T) )
				newton_dir = - deriv1 / deriv2
			else
				#newton_dir = - deriv1  # gradient descent
				newton_dir = - deriv1 / (deriv2 + abs(deriv1))
			end
		else  # not subdifferentiable at y
			newton_dir = - deriv1  # gradient descent
		end
		# Boyd and Xiao's guarded Newton
		# Need to guard the range to be in [lb, ub]
		#t = 0.99 * one(T)
		t = one(T)
        ycand = y +  newton_dir  # pure Newton
		# scaled interval
		lnew = 0.5 * one(T) * (u + l) - 0.5 * t * ( u - l)
		unew = 0.5 * one(T) * (u + l) + 0.5 * t * ( u - l)
		ynew = max(lnew, min(unew, ycand)) # projection 
		y = ynew
		toc = time()
		obj = 0.5 * norm(C, 2)^2 - y - 0.5 * norm(G, 2)^2
		push!(history, :objval, obj)
		push!(history, :itertime, toc - tic)
		IterativeSolvers.nextiter!(history)
	end
	log && IterativeSolvers.shrink!(history)
	y, C, history
end

"""
	dual_ls_cov_adj(G::Matrix{T}; 
					 tol::T=convert(T, 1e-8),
					 maxiter::Integer=100,
					 log::Bool = false
					) where T <: AbstractFloat

Solve the optimization problem
```
        min_X (1/2)|| G - X ||_F^2
        s.t.  e'Xe = 1, X >= 0
```
via dual nonsmooth Newton method.

# Input
- `G::Matrix{T}`: input symmetric matrix

# Keyword arguments
- `tol::T`: tolerance for convergence. Default is 1e-8.
- `maxiter::Integer`: maximum number of Newtwon iteration. Default is 100.
- `log::Bool`: whether to record iterate history or not. Defaut is `false`.

# Output
- `y::T`: dual variable. `-mu` in the paper. 
- `C::Matrix{T}`: dual variable. `C(mu)` in the paper. 
- `history::ConvergenceHistory`: convergence history. Valid only if `log == false`.
"""
function dual_ls_cov_adj(G::Matrix{T}; 
						 tol::T=convert(T, 1e-8),
						 maxiter::Integer=100,
						 log::Bool = false
						) where T <: AbstractFloat
	n = size(G)[1] + 1

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

	y, C, convhist = dual_ls_cov_adj!(G, Q, evec, d, 
									  indxq, z, dlambda, w, Q2,
									  indx, indxc, indxp, coltyp, S,
									  M, alph, bet, gam;
									  tol=tol, maxiter=maxiter,
									  log
									 )
end

"""
	bisect!(G::Matrix{T},
			 Q::Matrix{T}, evec::Vector{T},
			 d::Vector{T}, indxq::Vector{BlasInt},
			 z::Vector{T}, dlambda::Vector{T},
			 w::Vector{T}, Q2::Vector{T},
			 indx::Vector{BlasInt},
			 indxc::Vector{BlasInt},
			 indxp::Vector{BlasInt},
			 coltyp::Vector{BlasInt},
			 S::Vector{T}, M::Matrix{T}, 
			 alph::Vector{Bool}, 
			 bet::Vector{Bool},
			 gam::Vector{Bool}; 
			 tol::T=convert(T, 1e-8),
			 maxiter::Integer=100,
			 log::Bool = false
			) where T <: AbstractFloat

Solve the optimization problem
```
        min_X (1/2)|| G - X ||_F^2
        s.t.  e'Xe = 1, X >= 0
```
via dual bisection method.

# Input
- `G::Matrix{T}`: input symmetric matrix
- `Q::Matrix{T}`, `evec::Vector{T}`, `d::Matrix{T}`, `indxq::Vector{BlasInt}`, `z::Vector{T}`, `dlambda::Vector{T}`, `w::Vector{T}`, `Q2::Vector{T}`, `indx::Vector{BlasInt}`, `indxc::Vector{BlasInt}`, `indxp::Vector{BlasInt}`, `coltyp::Vector{BlasInt}`, `S::Vector{T}`, `M::Matrix{T}`, `alph::Vector{Bool}`, `bet::Vector{Bool}`, `gam::Vector{Bool}`: workspaces. 

# Keyword arguments
- `tol::T`: tolerance for convergence. Default is 1e-8.
- `maxiter::Integer`: maximum number of Newtwon iteration. Default is 100.
- `log::Bool`: whether to record iterate history or not. Defaut is `false`.

# Output
- `y::T`: dual variable. `-mu` in the paper. 
- `C::Matrix{T}`: dual variable. `C(mu)` in the paper. 
- `history::ConvergenceHistory`: convergence history. Valid only if `log == false`.
"""
function bisect!(G::Matrix{T},
						 Q::Matrix{T}, evec::Vector{T},
						 d::Vector{T}, indxq::Vector{BlasInt},
						 z::Vector{T}, dlambda::Vector{T},
						 w::Vector{T}, Q2::Vector{T},
						 indx::Vector{BlasInt},
						 indxc::Vector{BlasInt},
						 indxp::Vector{BlasInt},
						 coltyp::Vector{BlasInt},
						 S::Vector{T}, M::Matrix{T}, 
						 alph::Vector{Bool}, 
						 bet::Vector{Bool},
						 gam::Vector{Bool}; 
						 tol::T=convert(T, 1e-8),
						 maxiter::Integer=100,
						 log::Bool = false
						) where T <: AbstractFloat
	# lower and upper bounds of the dual variable mu
	# note y == -mu
	#lb = zero(T) # lb = G[end, end] - one(T)  
	lb = 0.0
	ub = 0.5 * norm(G, 2)^2   # Frobenius norm squared

	history = ConvergenceHistory(partial = !log)
	history[:tol] = tol
	IterativeSolvers.reserve!(T, history, :objval, maxiter)
	IterativeSolvers.reserve!(Float64, history, :itertime, maxiter)
	IterativeSolvers.nextiter!(history)
	
	C = copy(G)
	e = zeros(size(G)[1])
	e[end] = one(T)   # elementary unit vector 
	# if G - lb * e * e' >= 0 we are done
	BLAS.syr!('U', -lb, e, C) # C -= lb * e * e'
    tau = minimum(eigvals(Symmetric(C)))
	if (tau > -tol) 
		push!(history, :objval, zero(T))
		push!(history, :itertime, 0.0)
		IterativeSolvers.setconv(history, true)
		log && IterativeSolvers.shrink!(history)
		return lb, C, 0
	end

	#
	# Otherwise we do bisection
	#
	valG, vecG = eigen(G)  # eigendecomposition for reuse
	perm = sortperm(valG)
	valG = valG[perm]
	vecG = vecG[:, perm]
	n = size(G)[1] + 1

	(l, u) = (-ub, -lb)
	y = 0.5 * one(T) * (l + u)
	# for reuse
	P = reshape(view(Q2, 1:((n - 1) * (n - 1))), n - 1, n - 1)
	for iter = 1:maxiter # Newton iterations
		tic = time()
		## first derivative
		evec[1] = 0.0
		evec[2:end] .= e
		lam, Pte = getRank1UpdatePSDPart!(valG, vecG, y, evec, 
										  d, Q, C,
										  indxq, z, dlambda, w, Q2,
				 						  indx, indxc, indxp, coltyp,
										  S )
		deriv1 = C[end, end] - one(T)
		if ( abs(deriv1) < tol )  # converged
			toc = time()
			obj = 0.5 * norm(C, 2)^2 - y - 0.5 * norm(G, 2)^2
			push!(history, :objval, obj)
			push!(history, :itertime, toc - tic)
			IterativeSolvers.setconv(history, true)
			break
		end
		if deriv1 > zero(T)
			u = y
		else
			l = y
		end
		y = 0.5 * one(T) * (u + l)
		toc = time()
		obj = 0.5 * norm(C, 2)^2 - y - 0.5 * norm(G, 2)^2
		push!(history, :objval, obj)
		push!(history, :itertime, toc - tic)
		IterativeSolvers.nextiter!(history)
	end
	log && IterativeSolvers.shrink!(history)
	y, C, history
end

"""
	bisect(G::Matrix{T}; 
		   tol::T=convert(T, 1e-8),
		   maxiter::Integer=100
		  ) where T <: AbstractFloat

Solve the optimization problem
```
        min_X (1/2)|| G - X ||_F^2
        s.t.  e'Xe = 1, X >= 0
```
via dual bisection method.

# Input
- `G::Matrix{T}`: input symmetric matrix

# Keyword arguments
- `tol::T`: tolerance for convergence. Default is 1e-8.
- `maxiter::Integer`: maximum number of Newtwon iteration. Default is 100.
- `log::Bool`: whether to record iterate history or not. Defaut is `false`.

# Output
- `y::T`: dual variable. `-mu` in the paper. 
- `C::Matrix{T}`: dual variable. `C(mu)` in the paper. 
- `convhist::ConvergenceHistory`: convergence history. Valid only if `log == false`.
"""
function bisect(G::Matrix{T}; 
			    tol::T=convert(T, 1e-8),
			    maxiter::Integer=100,
				log::Bool = false
			   ) where T <: AbstractFloat
	n = size(G)[1] + 1

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

	y, C, convhist = bisect!(G, Q, evec, d, 
								indxq, z, dlambda, w, Q2,
								indx, indxc, indxp, coltyp, S,
								M, alph, bet, gam;
								tol=tol, maxiter=maxiter,
								log 
							)
end

"""
	mosek_ls_cov_adj(G::Matrix{T}) where T <: AbstractFloat

Solve
```
        min_X (1/2)|| G - X ||_F^2
        s.t.  e'Xe = 1, X >= 0
```
using Convex.jl and MOSEK. 

# Input
- `G::Matrix{T}`: input symmetric matrix

# Output
- `problem.optval::T`: optimal objective value
- `X.value::Matrix{T}: optimal variable
- `problem.constraints`: constraints fed to Convex.jl
"""
function mosek_ls_cov_adj(G::Matrix{T}) where T <: AbstractFloat
	X = Variable(size(G))
	problem = minimize(0.5 * sumsquares(vec(G-X)) )
	constraint = (X in :SDP)
	problem.constraints += constraint
	problem.constraints += X[end, end] == one(T)
	solve!(problem, Mosek.Optimizer(LOG=0))

	problem.optval, X.value, problem.constraints
end


"""
	prox_matrixperspective!(Omega::Matrix{T},
							eta::Vector{T},
							X::Matrix{T}, 
							y::Vector{T}, 
							sigma::T,
						 	Q::Matrix{T}, 
							evec::Vector{T},
						 	d::Vector{T}, 
							indxq::Vector{BlasInt},
						 	z::Vector{T}, 
							dlambda::Vector{T},
						 	w::Vector{T}, 
							Q2::Vector{T},
						 	indx::Vector{BlasInt},
						 	indxc::Vector{BlasInt},
						 	indxp::Vector{BlasInt},
						 	coltyp::Vector{BlasInt},
						 	S::Vector{T}, M::Matrix{T}, 
						 	alph::Vector{Bool}, 
						 	bet::Vector{Bool},
						 	gam::Vector{Bool}; 
						 	tol::T=convert(T, 1e-8),
						 	maxiter::Integer=50
						) where T <: AbstractFloat

Compute the proximity operator `(Omega, eta)` of the matrix perspective function for input `(X, y)`.

# Input
- `Omega::Matrix{T}`: output matrix
- `eta::Vector{T}`: output vector
- `X::Matrix{T}`: input matrix
- `y::Vector{T}`: input vector
- `sigma::T`: scaling factor for the matrix perspective function
- `Q::Matrix{T}`, `evec::Vector{T}`, `d::Matrix{T}`, `indxq::Vector{BlasInt}`, `z::Vector{T}`, `dlambda::Vector{T}`, `w::Vector{T}`, `Q2::Vector{T}`, `indx::Vector{BlasInt}`, `indxc::Vector{BlasInt}`, `indxp::Vector{BlasInt}`, `coltyp::Vector{BlasInt}`, `S::Vector{T}`, `M::Matrix{T}`, `alph::Vector{Bool}`, `bet::Vector{Bool}`, `gam::Vector{Bool}`: workspaces. 

# Keyword arguments
- `tol::T`: tolerance for convergence. Default is 1e-8.
- `maxiter::Integer`: maximum number of Newtwon iteration. Default is 100.

# Output
- `Omega::Matrix{T}`: matrix part of the proximity operator
- `eta::Vector{T}`  : vector part of the proximity operator
- `V:Matrix{T}`     : dual variable for `Omega`
- `w:Vector{T}`     : dual variable for `eta`
"""
function prox_matrixperspective!(Omega::Matrix{T},
								 eta::Vector{T},
								 X::Matrix{T}, 
								 y::Vector{T}, 
								 sigma::T,
						 		 Q::Matrix{T}, 
								 evec::Vector{T},
						 		 d::Vector{T}, 
								 indxq::Vector{BlasInt},
						 		 z::Vector{T}, 
								 dlambda::Vector{T},
						 		 w::Vector{T}, 
								 Q2::Vector{T},
						 		 indx::Vector{BlasInt},
						 		 indxc::Vector{BlasInt},
						 		 indxp::Vector{BlasInt},
						 		 coltyp::Vector{BlasInt},
						 		 S::Vector{T}, M::Matrix{T}, 
						 		 alph::Vector{Bool}, 
						 		 bet::Vector{Bool},
						 		 gam::Vector{Bool}; 
						 		 tol::T=convert(T, 1e-8),
						 		 maxiter::Integer=50
								) where T <: AbstractFloat
	p = size(X)[1]
	Xbar = [ -X/sigma  y/sqrt(2)/sigma; y'/sqrt(2)/sigma   1]
	@assert all(isfinite.(Xbar))
	# nu = -mu
	# C = [Xbar - mu * e * e']_{+} = U^{\star} = projection
	nu, C, iter = dual_ls_cov_adj!(Xbar, Q, evec, d,
									indxq, z, dlambda, w, Q2,
									indx, indxc, indxp, coltyp, S,
									M, alph, bet, gam; maxiter=maxiter)
	## projection operator
	#V = - C[1:p, 1:p]
	#u = sqrt(2) * C[1:p, end]
	### scaling to meet the Moreau decomposition
	#V .*= sigma
	#u .*= sigma
	#
	## proximity operator (used Moreau decomposition)
	#Omega = X - V
	#eta   = y - u
	#
	#Omega, eta, V, u

	## proximity operator (used Moreau decomposition)
	@views V = C[1:p, 1:p]
	# Omega .= sigma * V + X
	copyto!(Omega, X)  # Omega .= X
	BLAS.axpy!(sigma, V, Omega)  # Omega .= sigma * V + Omega
	@views u = C[1:p, end]
	# eta .= - sqrt(2) * sigma * u + y
	copyto!(eta, y)  # eta .= y
	BLAS.axpy!(- sigma * sqrt(2), u, eta) # eta .= -sigma*sqrt(2) * u + eta
	Omega, eta
end

"""
	prox_matrixperspective(X::Matrix{T}, 
						   y::Vector{T}, 
						   sigma::T;
						   tol::T=convert(T, 1e-8),
						   maxiter::Integer=100
						  ) where T <: Float64

Compute the proximity operator of the matrix perspective function for input `(X, y)`.

# Input
- `X::Matrix{T}`: input matrix
- `y::Vector{T}`: input vector
- `sigma::T`: scaling factor for the matrix perspective function

# Keyword arguments
- `tol::T`: tolerance for convergence. Default is 1e-8.
- `maxiter::Integer`: maximum number of Newtwon iteration. Default is 100.

# Output
- `Omega::Matrix{T}`: matrix part of the proximity operator
- `eta::Vector{T}`  : vector part of the proximity operator
- `V:Matrix{T}`     : dual variable for `Omega`
- `w:Vector{T}`     : dual variable for `eta`
"""
function prox_matrixperspective(X::Matrix{T}, 
								y::Vector{T}, 
								sigma::T;
						 	    tol::T=convert(T, 1e-8),
						 	    maxiter::Integer=100
							   ) where T <: Float64
	n = size(X)[1] + 2

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

	# primal variables
	Omega = similar(X)
	eta   = similar(y)
	
	#Omega, eta, V, w = prox_matrixperspective!(X, y, sigma, 
	prox_matrixperspective!(Omega, eta,
								X, y, sigma, 
						 		Q, evec, d, 
								indxq, z, dlambda, w, Q2,
								indx, indxc, indxp, coltyp, S,
								M, alph, bet, gam;
								tol=tol, maxiter=maxiter )
	# dual variables 
	V = X - Omega
	u = y - eta

	Omega, eta, V, u
end

