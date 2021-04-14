# MatrixPerspective.jl

| **Documentation** | **Build Status** |
|-------------------|------------------|
| [![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://won-j.github.io/MatrixPerspective.jl/dev) | [![Build Status](https://travis-ci.org/won-j/MatrixPerspective.jl.svg?branch=master)](https://travis-ci.com/github/won-j/MatrixPerspective.jl)  | 



This Julia package implements the algorithms for the proximity operator of the matrix perspective function discussed in the following paper.

* Joong-Ho Won (2020). [Proximity Operator of the Matrix Perspective Function and its Applications](https://papers.nips.cc/paper/2020/hash/45f31d16b1058d586fc3be7207b58053-Abstract.html). *34th Conference on Neural Information Processing Systems (NeurIPS 2020), Virtual Conference.* 

The implemented algorithms for solving the dual of the proximity operator problem are
1. Semismooth Newton method
2. Bisection
3. Interior point method (requires [MOSEK](https://www.mosek.com); academic license is free).

MatrixPerspective.jl was tested with Julia v1.5.1. The most recent release of Julia is v1.5.2. See [documentation](https://won-j.github.io/MatrixPerspective.jl/dev) for usage. It is not yet registered but can be installed, in the Julia Pkg mode, by
```{julia}
(@v1.5) Pkg> add https://github.com/won-j/MatrixPerspective.jl
```

