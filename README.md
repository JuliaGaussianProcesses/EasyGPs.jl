# EasyGPs

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGaussianProcesses.github.io/EasyGPs.jl/stable)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGaussianProcesses.github.io/EasyGPs.jl/dev)
[![CI](https://github.com/JuliaGaussianProcesses/EasyGPs.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaGaussianProcesses/EasyGPs.jl/actions/workflows/CI.yml)
[![Codecov](https://codecov.io/gh/JuliaGaussianProcesses/EasyGPs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaGaussianProcesses/EasyGPs.jl/tree/master)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

EasyGPs.jl is a package that defines a high-level API for the JuliaGaussianProcesses
ecosystem. It handles model parameterization and training, allowing users to focus on the
data and results without being distracted by tedious and repetitive tasks.

> [!NOTE]  
> This is an experimental package and may undergo breaking changes.

## Usage

In order to fit a GP, define one according to the familiar AbstractGP.jl interface and
let EasyGPs.jl handle the rest. The entry point for this is `EasyGPs.fit` (not exported):

```julia
using EasyGPs

kernel = 1.0 * with_lengthscale(SEKernel(), 1.0)
gp = with_gaussian_noise(GP(0.0, kernel), 0.1)
x = 0:0.1:10
y = sin.(x) .+ 0.1 .* randn(length(x))
fitted_gp = EasyGPs.fit(gp, x, y)
```

Under the hood, this will recognize the parameters (mean, variance, lengthscale) of the `GP`
you defined and automatically construct a parameterized model. It will then choose a cost
function, optimizer, and AD backend, and determine the optimal parameters.
