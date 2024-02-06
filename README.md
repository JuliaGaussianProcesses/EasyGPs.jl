# AutoGPs

AutoGPs is a front-end package for the JuliaGP ecosystem. It is geared towards making it
very easy to train GP models with a high-level API that does not require in-depth knowledge
of the low-level algorithmic choices.

## Usage

In order to fit a GP, define one according to the familiar AbstractGP.jl interface and
let AutoGPs.jl handle the rest. The entry point for this is `AutoGPs.fit` (not exported):

```julia
using AutoGPs

gp = GP(0.0, 1.0 * with_lengthscale(SEKernel(), 1.0))
x = 0:0.1:10
y = sin.(x) .+ 0.1 .* randn(length(x))
fitted_gp = AutoGPs.fit(gp, x, y)
```

Under the hood, this will recognize the parameters (mean, variance, lengthscale) of the `GP`
you defined and automatically construct a parameterized model. It will then choose a cost
function, optimizer, and AD backend, and determine the optimal parameters.
