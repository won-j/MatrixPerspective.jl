using Test, MatrixPerspective
using LinearAlgebra

@testset "exterior point" begin
X = Diagonal([1.0, 0, -1])  # diagonal
y = [1.0; 2; 0]
@show maximum(eigvals(X + 0.5 * y * y'))  # `X + 0.5 * y * y'` is indefinite
Ω, η = prox_matrixperspective(Matrix(X), y, 1.0) # prox operator
# desired solution
Ω0 = [ 1.066944081426201       0.2111491993644736      0.0
       0.2111491993644736      0.6659884405376072      0.0
       0.0                     0.0                     0.0 ]       
η0 = [0.6340926854346827; 0.845886972140417; 0.0]
@test isapprox(Ω, Ω0, atol=1e-6) && isapprox(η, η0, atol=1e-6)
end

@testset "interior point" begin
X = X = -10Diagonal([1.0, 1, 1])  # diagonal
y = [1.0; 2; 0]
@show maximum(eigvals(X + 0.5 * y * y'))  # `X + 0.5 * y * y'` is indefinite
Ω, η = prox_matrixperspective(Matrix(X), y, 1.0) # prox operator
# desired solution
@show Ω, η 
@test isapprox(Ω, zeros(size(X)), atol=1e-6) && isapprox(η, zeros(size(y)) atol=1e-6)
end
