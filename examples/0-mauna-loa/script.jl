# # Mauna Loa time series example
#
# In this notebook, we apply Gaussian process regression to the Mauna Loa CO₂
# dataset. It is adapted from the corresponding AbstractGPs.jl tutorial:
# https://juliagaussianprocesses.github.io/AbstractGPs.jl/stable/examples/1-mauna-loa/
# It is therefore instructive to read that first and then see how EasyGPs.jl
# simplifies the steps.

# ## Setup
#
# We make use of the following packages:

using CSV, DataFrames  # data loading
using EasyGPs  # handles all things related to GPs
using Plots  # visualisation

# Let's load and visualize the dataset.

# !!! tip
#     The `let` block [creates a new
#     scope](https://docs.julialang.org/en/v1/manual/variables-and-scoping/#scope-of-variables),
#     so any utility variables we define in here won't leak outside. This is
#     particularly helpful to keep notebooks tidy! The return value of the
#     block is given by its last expression.

(xtrain, ytrain), (xtest, ytest) = let
    data = CSV.read(joinpath(@__DIR__, "CO2_data.csv"), Tables.matrix; header=0)
    year = data[:, 1]
    co2 = data[:, 2]

    ## We split the data into training and testing set:
    idx_train = year .< 2004
    idx_test = .!idx_train

    (year[idx_train], co2[idx_train]), (year[idx_test], co2[idx_test])  # block's return value
end

function plotdata()
    plot(; xlabel="year", ylabel="CO₂ [ppm]", legend=:bottomright)
    scatter!(xtrain, ytrain; label="training data", ms=2, markerstrokewidth=0)
    return scatter!(xtest, ytest; label="test data", ms=2, markerstrokewidth=0)
end

plotdata()

# ## Prior
#
# We construct the GP prior using the same kernels and initial parameters as in the
# original tutorial. 

k_smooth_trend = exp(8.0) * with_lengthscale(SEKernel(), exp(4.0))#with_lengthscale(SEKernel(), exp(4.0))
k_seasonality = exp(2.0) * PeriodicKernel(; r=[0.5]) *
    with_lengthscale(SEKernel(), exp(4.0))
k_medium_term_irregularities = 1.0 * with_lengthscale(RationalQuadraticKernel(; α=exp(-1.0)), 1.0)
k_noise_terms = exp(-4.0) * with_lengthscale(SEKernel(), exp(-2.0)) + exp(-4.0) * WhiteKernel()
kernel = k_smooth_trend + k_seasonality + k_medium_term_irregularities + k_noise_terms
#md nothing #hide

# We construct the `GP` object with the kernel above:

gp = GP(kernel)
#md nothing #hide

# ## Posterior
#
# To construct a posterior, we can call the GP object with the usual AbstractGPs.jl API:

fpost_init = posterior(gp(xtrain), ytrain)
#md nothing #hide

# Let's visualize what the GP fitted to the data looks like, for the initial choice of kernel hyperparameters.
#
# We use the following function to plot a GP `f` on a specific range, using the
# AbstractGPs [plotting
# recipes](https://juliagaussianprocesses.github.io/AbstractGPs.jl/dev/concrete_features/#Plotting).
# By setting `ribbon_scale=2` we visualize the uncertainty band with ``\pm 2``
# (instead of the default ``\pm 1``) standard deviations.

plot_gp!(f; label) = plot!(f(1920:0.2:2030); ribbon_scale=2, linewidth=1, label)
#md nothing #hide

plotdata()
plot_gp!(fpost_init; label="posterior f(⋅)")

# A reasonable fit to the data, but poor extrapolation away from the observations!

# ## Hyperparameter Optimization
#
# We can now call `EasyGPs.fit` in order to optimize the hyperparameters. This takes care
# of all the parameterizations, automatic differentiation, and runs the optimizer for us.
# We pass an option to choose the exact same optimizer as in the original tutorial.

@time fitted_gp = EasyGPs.fit(
    gp, xtrain, ytrain;
    optimizer = Optim.LBFGS(;
        alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
        linesearch=Optim.LineSearches.BackTracking(),
    )
)
#md nothing #hide

# Let's now construct the posterior GP with the optimized hyperparameters:

fpost_opt = posterior(fitted_gp(xtrain), ytrain)
#md nothing #hide

# This is the kernel with the point-estimated hyperparameters:

fpost_opt.prior.kernel

# And, finally, we can visualize our optimized posterior GP:

plotdata()
plot_gp!(fpost_opt; label="optimized posterior f(⋅)")
