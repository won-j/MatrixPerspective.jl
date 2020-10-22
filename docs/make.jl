using Documenter, MatrixPerspective

makedocs(
    format = Documenter.HTML(),
    sitename = "MatrixPerspective.jl",
    authors = "Joong-Ho Won",
    clean = true,
    debug = true,
    pages = [
        "index.md"
    ]
)

deploydocs(
    repo   = "github.com/won-j/MatrixPerspective.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing
)

