using LinearAlgebra

import LinearAlgebra.LAPACK.liblapack
import LinearAlgebra.LAPACK.chklapackerror
import LinearAlgebra.LAPACK.chkfinite
import LinearAlgebra.BLAS.BlasInt
import LinearAlgebra.BLAS.@blasfunc

"""
	laed2!( n1::BlasInt, 
			 d::AbstractVector{Float64}, 
			 Q::AbstractMatrix{Float64}, 
			 indxq::AbstractVector{BlasInt}, 
			 rho::Float64,
			 z::AbstractVector{Float64}, 
			 dlambda::AbstractVector{Float64}, 
			 w::AbstractVector{Float64}, 
			 Q2::AbstractVector{Float64}, 
			 indx::AbstractVector{BlasInt}, 
			 indxc::AbstractVector{BlasInt}, 
			 indxp::AbstractVector{BlasInt}, 
			 coltyp::AbstractVector{BlasInt} )

Wrapper function for LAPACK routine DLAED2 (based on [LAPACK 3.9.0](http://www.netlib.org/lapack/explore-html/d2/d24/group__aux_o_t_h_e_rcomputational_ga391322d1faf47723521ae1d0a72f1559.html)):

DLAED2 merges the two sets of eigenvalues together into a single
sorted set. Then it tries to deflate the size of the problem.
There are two ways in which deflation can occur: when two or more
eigenvalues are close together or if there is a tiny entry in the
Z vector.  For each such occurrence the order of the related secular
equation problem is reduced by one.

In a nutshell, `DLAED2` computes a partial eigenvalue decomposition (EVD) of
```
	diag(Q1, Q2) * (diag([d1; d2]) + rho * z * z') * diag(Q1', Q2')
```
where `norm(z) == sqrt(2)`. The order of matrix `Q1` and eigenvalue vector `d1` is `n1`. The dimension of vector `z` is `n`. 

The EVD computation reduces to that of the symmetric rank-one update of the diagonal matrix `diag(d)` where `d == [d1; d2])`. If some diagonal elements of the latter matrix are equal or those of vector `z` are zero then the corresponding eigenvalues are unchaged by the update. This phenomenon is called the *deflation*. `DLAED2` finds deflated eigenvalues and determines the corresponding eigenvectors. These eigenvectors are post-multiplied to `Q == diag(Q1, Q2)` to partially complete the desired EVD. In doing so, both the eigenvalues and the columns of `Q` are permuted to group deflated eigenvalues. The number of deflated eigenvalues is `k`, and they are placed to the first part of `d`. This permutation information is contained in the variable `indx`.

```
SUBROUTINE DLAED2( K, N, N1, D, Q, LDQ, INDXQ, RHO, Z, DLAMDA, W,
                   Q2, INDX, INDXC, INDXP, COLTYP, INFO )
```

# Arguments

- param[out] K
          K is INTEGER
         The number of non-deflated eigenvalues, and the order of the
         related secular equation. 0 <= K <=N.

- param[in] N
          N is INTEGER
         The dimension of the symmetric tridiagonal matrix.  N >= 0.

- param[in] N1
          N1 is INTEGER
         The location of the last eigenvalue in the leading sub-matrix.
         min(1,N) <= N1 <= N/2.

- param[in,out] D
          D is DOUBLE PRECISION array, dimension (N)
         On entry, D contains the eigenvalues of the two submatrices to
         be combined.
         On exit, D contains the trailing (N-K) updated eigenvalues
         (those which were deflated) sorted into increasing order.

- param[in,out] Q
          Q is DOUBLE PRECISION array, dimension (LDQ, N)
         On entry, Q contains the eigenvectors of two submatrices in
         the two square blocks with corners at (1,1), (N1,N1)
         and (N1+1, N1+1), (N,N).
         On exit, Q contains the trailing (N-K) updated eigenvectors
         (those which were deflated) in its last N-K columns.

- param[in] LDQ
          LDQ is INTEGER
         The leading dimension of the array Q.  LDQ >= max(1,N).

- param[in,out] INDXQ
          INDXQ is INTEGER array, dimension (N)
         The permutation which separately sorts the two sub-problems
         in D into ascending order.  Note that elements in the second
         half of this permutation must first have N1 added to their
         values. Destroyed on exit.

- param[in,out] RHO
          RHO is DOUBLE PRECISION
         On entry, the off-diagonal element associated with the rank-1
         cut which originally split the two submatrices which are now
         being recombined.
         On exit, RHO has been modified to the value required by
         DLAED3.

- param[in] Z
          Z is DOUBLE PRECISION array, dimension (N)
         On entry, Z contains the updating vector (the last
         row of the first sub-eigenvector matrix and the first row of
         the second sub-eigenvector matrix).
         On exit, the contents of Z have been destroyed by the updating
         process.

- param[out] DLAMDA
          DLAMDA is DOUBLE PRECISION array, dimension (N)
         A copy of the first K eigenvalues which will be used by
         DLAED3 to form the secular equation.

- param[out] W
          W is DOUBLE PRECISION array, dimension (N)
         The first k values of the final deflation-altered z-vector
         which will be passed to DLAED3.

- param[out] Q2
          Q2 is DOUBLE PRECISION array, dimension (N1**2+(N-N1)**2)
         A copy of the first K eigenvectors which will be used by
         DLAED3 in a matrix multiply (DGEMM) to solve for the new
         eigenvectors.

- param[out] INDX
          INDX is INTEGER array, dimension (N)
         The permutation used to sort the contents of DLAMDA into
         ascending order.

- param[out] INDXC
          INDXC is INTEGER array, dimension (N)
         The permutation used to arrange the columns of the deflated
         Q matrix into three groups:  the first group contains non-zero
         elements only at and above N1, the second contains
         non-zero elements only below N1, and the third is dense.

- param[out] INDXP
          INDXP is INTEGER array, dimension (N)
         The permutation used to place deflated values of D at the end
         of the array.  INDXP(1:K) points to the nondeflated D-values
         and INDXP(K+1:N) points to the deflated eigenvalues.

- param[out] COLTYP
          COLTYP is INTEGER array, dimension (N)
         During execution, a label which will indicate which of the
         following types a column in the Q2 matrix is:
         1 : non-zero in the upper half only;
         2 : dense;
         3 : non-zero in the lower half only;
         4 : deflated.
         On exit, COLTYP(i) is the number of columns of type i,
         for i=1 to 4 only.

- param[out] INFO
          INFO is INTEGER
          = 0:  successful exit.
          < 0:  if INFO = -i, the i-th argument had an illegal value.


Parameters `K`, `N`, `LDQ`, and `INFO` are omitted in the wrapper. They are inferred from the other parameters. For unomitted parameters, proper workspace must be provided.
"""
function laed2!( n1::BlasInt, 
				 d::AbstractVector{Float64}, 
				 Q::AbstractMatrix{Float64}, 
				 indxq::AbstractVector{BlasInt}, 
				 rho::Float64,
				 z::AbstractVector{Float64}, 
				 dlambda::AbstractVector{Float64}, 
				 w::AbstractVector{Float64}, 
				 Q2::AbstractVector{Float64}, 
				 indx::AbstractVector{BlasInt}, 
				 indxc::AbstractVector{BlasInt}, 
				 indxp::AbstractVector{BlasInt}, 
				 coltyp::AbstractVector{BlasInt} )
	@assert size(Q)[2] == length(d)
	@assert length(z) == length(d)
	chkfinite(Q) # balancing routines don't support NaNs and Infs
	k = Ref{BlasInt}()
	#k = Vector{BlasInt}(undef, 1)
	n = BlasInt(length(d))

	ldq   = BlasInt(max(1, stride(Q, 2)) )

	RHO = Ref{Float64}()
	RHO[] = rho
	#RHO = Vector{Float64}(undef, 1)
	#RHO[1] = rho

	info  = Ref{BlasInt}()
	
    ccall((@blasfunc(dlaed2_), liblapack), Cvoid,
		(Ptr{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64},
		 Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{Float64},
		 Ref{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
		 Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
		 Ptr{BlasInt}),
		 k, n, n1, d, 
		 Q, ldq, indxq, RHO, 
		 z, dlambda, w, Q2,
		 indx, indxc, indxp, coltyp, 
		 info )
	chklapackerror(info[])

	k[], RHO[]
	#k[1], RHO[1]
end

"""
	laed3!( k::BlasInt, n1::BlasInt, 
			 d::AbstractVector{Float64}, 
			 Q::AbstractMatrix{Float64},  
		     rho::Float64, 
			 dlambda::AbstractVector{Float64}, 
			 Q2::AbstractVector{Float64}, 
			 indx::AbstractVector{BlasInt}, 
			 ctot::AbstractVector{BlasInt}, 
			 w::AbstractVector{Float64},
			 S::AbstractVector{Float64} )

Wrapper function for LAPACK routine DLAED3 (based on [LAPACK 3.9.0](https://www.netlib.org/lapack/explore-html/d2/d24/group__aux_o_t_h_e_rcomputational_gaf999454a10e2af1e9b7b3e7ddfb73869.html):

DLAED3 finds the roots of the secular equation, as defined by the
 values in D, W, and RHO, between 1 and K.  It makes the
 appropriate calls to DLAED4 and then updates the eigenvectors by
 multiplying the matrix of eigenvectors of the pair of eigensystems
 being combined by the matrix of eigenvectors of the K-by-K system
 which is solved here.

In a nutshull, `DLAED3` completed the eigenvalue decomposition (EVD) of
```
	diag(Q1, Q2) * (diag([d1; d2]) + rho * z * z') * diag(Q1', Q2')
```
initiated by `DLAED2`. Here `norm(z) == sqrt(2)`. The order of matrix `Q1` and eigenvalue vector `d1` is `n1`. The dimension of vector `z` is `n`. 

The number of deflated eigenvalues `k` and the associated permutation `indx` found by `DLAED2` is passed to `DLAED3`. It then computes the modified eigenvalues (with a smaller dimension) by solving secular equations. The corresponding eigenvectors are updated accordingly.

```
SUBROUTINE DLAED3( K, N, N1, D, Q, LDQ, RHO, DLAMDA, Q2, INDX,
                   CTOT, W, S, INFO )
```

# Arguments

- param[in] K
          K is INTEGER
          The number of terms in the rational function to be solved by
          DLAED4.  K >= 0.

- param[in] N
          N is INTEGER
          The number of rows and columns in the Q matrix.
          N >= K (deflation may result in N>K).

- param[in] N1
          N1 is INTEGER
          The location of the last eigenvalue in the leading submatrix.
          min(1,N) <= N1 <= N/2.

- param[out] D
          D is DOUBLE PRECISION array, dimension (N)
          D(I) contains the updated eigenvalues for
          1 <= I <= K.

- param[out] Q
          Q is DOUBLE PRECISION array, dimension (LDQ,N)
          Initially the first K columns are used as workspace.
          On output the columns 1 to K contain
          the updated eigenvectors.

- param[in] LDQ
          LDQ is INTEGER
          The leading dimension of the array Q.  LDQ >= max(1,N).

- param[in] RHO
          RHO is DOUBLE PRECISION
          The value of the parameter in the rank one update equation.
          RHO >= 0 required.

- param[in,out] DLAMDA
          DLAMDA is DOUBLE PRECISION array, dimension (K)
          The first K elements of this array contain the old roots
          of the deflated updating problem.  These are the poles
          of the secular equation. May be changed on output by
          having lowest order bit set to zero on Cray X-MP, Cray Y-MP,
          Cray-2, or Cray C-90, as described above.

- param[in] Q2
          Q2 is DOUBLE PRECISION array, dimension (LDQ2*N)
          The first K columns of this matrix contain the non-deflated
          eigenvectors for the split problem.

- param[in] INDX
          INDX is INTEGER array, dimension (N)
          The permutation used to arrange the columns of the deflated
          Q matrix into three groups (see DLAED2).
          The rows of the eigenvectors found by DLAED4 must be likewise
          permuted before the matrix multiply can take place.

- param[in] CTOT
          CTOT is INTEGER array, dimension (4)
          A count of the total number of the various types of columns
          in Q, as described in INDX.  The fourth column type is any
          column which has been deflated.

- param[in,out] W
          W is DOUBLE PRECISION array, dimension (K)
          The first K elements of this array contain the components
          of the deflation-adjusted updating vector. Destroyed on
          output.

- param[out] S
          S is DOUBLE PRECISION array, dimension (N1 + 1)*K
          Will contain the eigenvectors of the repaired matrix which
          will be multiplied by the previously accumulated eigenvectors
          to update the system.

- param[out] INFO
          INFO is INTEGER
          = 0:  successful exit.
          < 0:  if INFO = -i, the i-th argument had an illegal value.
          > 0:  if INFO = 1, an eigenvalue did not converge

Parameters `N`, `LDQ`, and `INFO` are omitted in the wrapper. They are inferred from the other parameters. For unomitted parameters, proper workspace must be provided.
"""
function laed3!( k::BlasInt, n1::BlasInt, 
				 d::AbstractVector{Float64}, 
				 Q::AbstractMatrix{Float64},  
			     rho::Float64, 
				 dlambda::AbstractVector{Float64}, 
				 Q2::AbstractVector{Float64}, 
				 indx::AbstractVector{BlasInt}, 
				 ctot::AbstractVector{BlasInt}, 
				 w::AbstractVector{Float64},
				 S::AbstractVector{Float64} )

	#@assert size(Q2)[2] == length(indx)
	#@assert k == length(dlambda)
	#@assert 4 == length(ctot)
	#@assert k == length(w)
	#chkfinite(Q2) # balancing routines don't support NaNs and Infs

	n = BlasInt(length(indx))

	ldq   = BlasInt(max(1, stride(Q, 2)) )
	info  = Ref{BlasInt}()
	#info = Vector{BlasInt}(undef, 1)
    ccall((@blasfunc(dlaed3_), liblapack), Cvoid,
		(Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64},
		 Ptr{Float64}, Ref{BlasInt}, Ref{Float64}, Ptr{Float64},
		 Ref{Float64}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64},
		 Ptr{Float64}, Ptr{BlasInt}),
		 k, n, n1, d, 
		 Q, ldq, rho, dlambda,
		 Q2, indx, ctot, w,
		 S, info )
	chklapackerror(info[])
end

"""
	rank1eig!(d::AbstractVector{T}, 
			    Q::AbstractMatrix{T}, 
				rho::T, 
				x::AbstractVector{T},
				indxq::AbstractVector{BlasInt}, 
				z::AbstractVector{T},
				dlambda::AbstractVector{T},
				w::AbstractVector{T},
				Q2::AbstractVector{T},
				indx::AbstractVector{BlasInt},
				indxc::AbstractVector{BlasInt},
				indxp::AbstractVector{BlasInt},
				coltyp::AbstractVector{BlasInt},
				S::AbstractVector{T}
			  ) where T <: AbstractFloat

Eigenvalue decomposition of symmetric rank-one update
`Q * Diagonal(d) * Q' + rho * x * x'`.

# Input
- `d::AbstractVector{T}`: eigenvalue vector to be repaired. Must have length `n + 1` with `d[1] = 1.0` and `d[2:end]` sorted in ascending order.
- `Q::AbstractMatrix{T}`: eigenvector matrix to be repaired. Must be order `n + 1` with `Q[1, 1] = 1.0` and `Q[1, :] = zeros(n)`.
- `rho::T`: the scalar in the rank-one update equation.
- `x::AbstractVector{T}`: rank-one update vector. Must be of length `n + 1` with `x[1] = 0.0` and `norm(x, 2) == sqrt(2.0)`
- `indxq::AbstractVector{BlasInt}`, `z::AbstractVector{T}`, `dlambda::AbstractVector{T}`, `w::AbstractVector{T}`, `Q2::AbstractVector{T}`, `indx::AbstractVector{BlasInt}`, `indxc::AbstractVector{BlasInt}`, `indxp::AbstractVector{BlasInt}`, `coltyp::AbstractVector{BlasInt}`, `S::AbstractVector{T}`: arguments passed to `laed2!()` and `laed3!()`. Proper workspace must be prepared.

# Output
- `k::BlasInt`: number of non-deflated eigenvalues, which is the order of the related secular equation. `0 <= k <= n + 1` (see `laed3()`).
- On return, both `Q` and `d` are modified and permuted. Variable `indx` contains the permutation used to arrange the columns of the deflated `Q` matrix into three groups (see `laed2()`).

"""
function rank1eig!(d::AbstractVector{T}, 
					Q::AbstractMatrix{T}, 
					rho::T, 
					x::AbstractVector{T},
					indxq::AbstractVector{BlasInt}, 
					z::AbstractVector{T},
					dlambda::AbstractVector{T},
					w::AbstractVector{T},
					Q2::AbstractVector{T},
					indx::AbstractVector{BlasInt},
					indxc::AbstractVector{BlasInt},
					indxp::AbstractVector{BlasInt},
					coltyp::AbstractVector{BlasInt},
					S::AbstractVector{T}
				  ) where T <: AbstractFloat
	@assert size(Q)[1] == length(d)
	@assert size(Q)[2] == length(d)
	@assert length(x)  == length(d)

	n1 = 1 # cut point
	n = length(d)

	# normalize x to have length sqrt(2.0)
	sqrttwo = sqrt(2.0)
	xlen = norm(x, 2)
	x ./= xlen
	x .*= sqrttwo
	rho *= xlen^2 * 0.5
	if rho < zero(T)	
		rhoval = -rho
		d .*= -one(T)  #d .= -d	  # negate the vector
		#indxq .= [1; n-1:-1:1 ]
		indxq[1] = 1
		for i=2:n
			indxq[i] = n - i + 1
		end
	else
		rhoval = rho  
		#indxq .= [1; 1:n-1 ]
		indxq[1] = 1
		for i = 2:n
			indxq[i] = i - 1 
		end
	end

	#z .= Q' * x
	BLAS.gemv!('T', one(T), Q, x, zero(T), z)

	k, rho1 = laed2!(n1, d, Q, indxq, rhoval, z, dlambda, w, Q2, indx, indxc, indxp, coltyp )

	laed3!(k, n1, d, Q, rho1, dlambda, Q2, indxc, coltyp, w, S)

	if rho < zero(T)  # restore eigenvalues
		d .*= -one(T) # d .= -d   
	end

	k
end

"""
	rank1eig(lam::AbstractVector{T}, 
			  P::AbstractMatrix{T}, 
			  rho::T, u::AbstractVector{T}
			) where T <: AbstractFloat

Compute eigenvalue decomposition of 
```
	P * Diagonal(lam) * P' + rho * u * u'
```

# Input
- `lam::AbstractVector{T}`: eigenvalues of the `p * p` matrix to be updated
- `P::AbstractMatrix{T}`: eigenvectors of the `p * p` matrix to be update
- `rho::T`: scalar for rank-one update
- `u::AbstractVector{T}`: updating vector

# Output
- `d1::view(::Vector{T}, 2:(p + 1))`: updated eigenvalues
- `Q1::view(::Matrix{T}, 2:(p + 1), 2:(p + 1))`: updated eigenvectors
"""
function rank1eig(lam::AbstractVector{T}, P::AbstractMatrix{T}, 
					rho::T, u::AbstractVector{T}
				  ) where T <: AbstractFloat

	d = [1.0; lam]
	Q = zeros(size(P) .+ 1)
	Q[1, 1] = 1.0
	Q[2:end, 2:end] .= P

	x = [0; u] 

	# workspaces
	n = length(d)
	indxq = Vector{BlasInt}(undef, n)
	#work = Vector{T}(undef, 3 * n + 2 * n * n)
	#iwork = Vector{BlasInt}(undef, 4 * n)

	z = Vector{Float64}(undef, n)
	dlambda = Vector{Float64}(undef, n)
	w = Vector{Float64}(undef, n)
	Q2 = Vector{Float64}(undef, n * n)

	indx   = Vector{BlasInt}(undef, n + 1)
	indxc  = Vector{BlasInt}(undef, n)
	indxp  = Vector{BlasInt}(undef, n)
	coltyp = Vector{BlasInt}(undef, n)
	S = Vector{Float64}(undef, n * n)
	
	#perm = rank1eig!(d, Q, rho, x, indxq, work, iwork)
	#rank1eig!(d, Q, rho, x, indxq, work, iwork)
	#perm = view(iwork, 1:n)
	k = rank1eig!(d, Q, rho, x, indxq, z, dlambda, w, Q2,
				 indx, indxc, indxp, coltyp, S)
	#perm = view(indx, 1:n)
	#Q1 = view(Q, 2:size(Q)[1], perm .!= 1)
	#d1 = view(d, perm .!= 1)
	#d1, Q1

	#= 	
		First k eigenvalues are non-deflated. 
		The initial, artificial eigenvalue must have been deflated
		The corresponding artificial eigenvector is [1 0 ... 0]
		and unchanged.
	=#
	indxartf = -1
	for i=(k + 1):n
		if indx[i] == 1
			indxartf = i
			break
		end
	end
	# Move the artificial eigenvector to the first column
	Q[:, indxartf] .= Q[:, 1]
	Q[1, 1] = 1.0
	Q[2:end, 1] .= 0.0
	d1 = d[1]
	d[1] = d[indxartf]
	d[indxartf] = d1

	# return values
	Q1 = view(Q, 2:n, 2:n)
	d1 = view(d, 2:n)
	d1, Q1
end

