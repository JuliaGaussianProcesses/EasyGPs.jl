# AutoGPs

[![CI](https://github.com/JuliaGaussianProcesses/AutoGPs.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaGaussianProcesses/AutoGPs.jl/actions/workflows/CI.yml)
[![Codecov](https://codecov.io/gh/JuliaGaussianProcesses/AutoGPs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaGaussianProcesses/AutoGPs.jl)

AutoGPs is a front-end package for the JuliaGP ecosystem. It is geared towards making it
very easy to train GP models with a high-level API that does not require in-depth knowledge
of the low-level algorithmic choices.

NOTE: there is a similarly named, registered package AutoGP.jl, which has a very different
scope. This package will be renamed before being registered.

## Usage

In order to fit a GP, define one according to the familiar AbstractGP.jl interface and
let AutoGPs.jl handle the rest. The entry point for this is `AutoGPs.fit` (not exported):

```julia
using AutoGPs

kernel = 1.0 * with_lengthscale(SEKernel(), 1.0)
gp = with_gaussian_noise(GP(0.0, kernel), 0.1)
x = 0:0.1:10
y = sin.(x) .+ 0.1 .* randn(length(x))
fitted_gp = AutoGPs.fit(gp, x, y)
```

Under the hood, this will recognize the parameters (mean, variance, lengthscale) of the `GP`
you defined and automatically construct a parameterized model. It will then choose a cost
function, optimizer, and AD backend, and determine the optimal parameters.
