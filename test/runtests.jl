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
@test Ω ≈ Ω0 && η ≈ η0
end

@testset "interior point" begin
X = X = -10Diagonal([1.0, 1, 1])  # diagonal
y = [1.0; 2; 0]
@show maximum(eigvals(X + 0.5 * y * y'))  # `X + 0.5 * y * y'` is indefinite
Ω, η = prox_matrixperspective(Matrix(X), y, 1.0) # prox operator
# desired solution
@test Ω ≈ zeros(size(X)) && η ≈ zeros(size(y))
end
