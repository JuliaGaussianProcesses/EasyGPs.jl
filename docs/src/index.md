# EasyGPs.jl

EasyGPs.jl is a package that defines a high-level API for the JuliaGaussianProcesses
ecosystem. It is aimed at people who want to use GPs to do exploratory analysis, model data
and make predictions without having to deal with all the low-level detail.

!!! note
    This is currently an experimental package and may undergo rapid changes.

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
