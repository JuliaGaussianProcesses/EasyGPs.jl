### Process examples
using Pkg
Pkg.add(
    Pkg.PackageSpec(; url = "https://github.com/JuliaGaussianProcesses/JuliaGPsDocs.jl"),
) # While the package is unregistered, it's a workaround

using JuliaGPsDocs

using EasyGPs

JuliaGPsDocs.generate_examples(EasyGPs)

### Build documentation
using Documenter

# Doctest setup
DocMeta.setdocmeta!(
    EasyGPs,
    :DocTestSetup,
    quote
        using EasyGPs
    end;  # we have to load all packages used (implicitly) within jldoctest blocks in the API docstrings
    recursive = true,
)

makedocs(;
    sitename = "EasyGPs.jl",
    format = Documenter.HTML(; size_threshold_ignore = ["examples/0-mauna-loa/index.md"]),
    modules = [EasyGPs],
    pages = [
        "Home" => "index.md",
        "Examples" => JuliaGPsDocs.find_generated_examples(EasyGPs),
    ],
    warnonly = true,
    checkdocs = :exports,
    doctestfilters = JuliaGPsDocs.DOCTEST_FILTERS,
)

deploydocs(; repo = "github.com/JuliaGaussianProcesses/EasyGPs.jl.git", push_preview = true)
