module EasyGPs

using Reexport

@reexport using AbstractGPs
@reexport using GPLikelihoods

using Optimization: Optimization
@reexport import OptimizationOptimJL: Optim
using ParameterHandling: ParameterHandling
using Enzyme: Enzyme
using Zygote: Zygote

using ApproximateGPs
using Distributions: MvNormal
using LinearAlgebra: I

export variational_gaussian, with_gaussian_noise
export SVA, SVGP

"""
    fit(object, data; kwargs...)

Fit `object` to `data`. Returns another instance of `typeof(object)` with optimized
parameters. For possible keyword arguments see `optimize`. Supported object types:
- `EasyGPs.NoisyGP`
"""
function fit(object, data; kwargs...)
    model, θ0 = parameterize(object)
    θ = optimize(model, θ0, data; kwargs...)
    return model(θ)
end

"""
    fit(object, x, y; kwargs...)

A shorthand for `fit(object, (; x, y); kwargs...)` for when the only data is `x` and `y`.
"""
fit(gp, x, y; kwargs...) = fit(gp, (; x, y); kwargs...)

struct Parameterized{T}
    object::T
end

function (p::Parameterized)(θ)
    return apply_parameters(p.object, ParameterHandling.value(θ))
end

"""
    parameterize(object) -> model, θ

Turn `object` into a callable parameterized version of itself and a parameter `θ`.
After assigning `model, θ = parameterize(object)`, calling `model(θ)` will yield the same
`object` back. 
"""
parameterize(object) = Parameterized(object), extract_parameters(object)

"""
    optimize(model, θ0, data; kwargs...) -> θ_opt

Takes a callable `model` and returns the optimal parameter, starting with initial parameters
`θ0`. In order to work, there needs to be an implementation of `EasyGPs.costfunction` taking
two arguments, the first of which is of type `typeof(model(θ0))`.
"""
function optimize(model, θ0, data; iterations=1000, optimizer=Optim.BFGS(), kwargs...)
    par0, unflatten = ParameterHandling.flatten(θ0)
    optf = Optimization.OptimizationFunction(
        (par, data) -> costfunction(model(unflatten(par)), data), Optimization.AutoZygote()
    )
    prob = Optimization.OptimizationProblem(optf, par0, data)
    sol = Optimization.solve(prob, optimizer; maxiters=iterations)
    return unflatten(sol.u)
end

"""
    _isequal(object1, object2)

Check whether two things are equal for the purposes of this library. For this to be true,
roughly speaking the objects must be of the same type and have the same parameters.
"""
_isequal(::T1, ::T2) where {T1,T2} = false

# Mean functions
extract_parameters(::ZeroMean) = nothing
apply_parameters(m::ZeroMean, θ) = m
_isequal(::ZeroMean, ::ZeroMean) = true

extract_parameters(m::ConstMean) = m.c
apply_parameters(::ConstMean, θ) = ConstMean(θ)
_isequal(m1::ConstMean, m2::ConstMean) = isapprox(m1.c, m2.c)

# Simple kernels
KernelsWithoutParameters = Union{SEKernel,Matern32Kernel,Matern52Kernel,WhiteKernel}

extract_parameters(::T) where {T<:KernelsWithoutParameters} = nothing
apply_parameters(k::T, θ) where {T<:KernelsWithoutParameters} = k
_isequal(k1::T, k2::T) where {T<:KernelsWithoutParameters} = true

extract_parameters(k::PeriodicKernel) = ParameterHandling.positive(only(k.r))
apply_parameters(::PeriodicKernel, θ) = PeriodicKernel(; r=[θ])
_isequal(k1::T, k2::T) where {T<:PeriodicKernel} = k1.r ≈ k2.r

extract_parameters(k::RationalQuadraticKernel) = ParameterHandling.positive(only(k.α))
function apply_parameters(k::RationalQuadraticKernel, θ)
    return RationalQuadraticKernel(; α=θ, metric=k.metric)
end
_isequal(k1::T, k2::T) where {T<:RationalQuadraticKernel} = true

# Composite kernels
extract_parameters(k::KernelSum) = map(extract_parameters, k.kernels)
apply_parameters(k::KernelSum, θ) = KernelSum(map(apply_parameters, k.kernels, θ))
_isequal(k1::KernelSum, k2::KernelSum) = mapreduce(_isequal, &, k1.kernels, k2.kernels)

extract_parameters(k::KernelProduct) = map(extract_parameters, k.kernels)
apply_parameters(k::KernelProduct, θ) = KernelProduct(map(apply_parameters, k.kernels, θ))
function _isequal(k1::KernelProduct, k2::KernelProduct)
    return mapreduce(_isequal, &, k1.kernels, k2.kernels)
end

function extract_parameters(k::TransformedKernel)
    return (extract_parameters(k.kernel), extract_parameters(k.transform))
end

function apply_parameters(k::TransformedKernel, θ)
    return TransformedKernel(
        apply_parameters(k.kernel, θ[1]), apply_parameters(k.transform, θ[2])
    )
end

function _isequal(k1::TransformedKernel, k2::TransformedKernel)
    return _isequal(k1.kernel, k2.kernel) && _isequal(k1.transform, k2.transform)
end

function extract_parameters(k::ScaledKernel)
    return (extract_parameters(k.kernel), ParameterHandling.positive(only(k.σ²)))
end

function apply_parameters(k::ScaledKernel, θ)
    return ScaledKernel(apply_parameters(k.kernel, θ[1]), θ[2])
end

function _isequal(k1::ScaledKernel, k2::ScaledKernel)
    return _isequal(k1.kernel, k2.kernel) && isapprox(k1.σ², k2.σ²)
end

# Transforms
extract_parameters(t::ScaleTransform) = ParameterHandling.positive(only(t.s))
apply_parameters(::ScaleTransform, θ) = ScaleTransform(θ)
_isequal(t1::ScaleTransform, t2::ScaleTransform) = isapprox(t1.s, t2.s)

# Likelihoods
extract_parameters(::BernoulliLikelihood) = nothing
apply_parameters(l::BernoulliLikelihood, θ) = l
_isequal(l1::T, l2::T) where {T<:BernoulliLikelihood} = true

extract_parameters(::PoissonLikelihood) = nothing
apply_parameters(l::PoissonLikelihood, θ) = l
_isequal(l1::T, l2::T) where {T<:PoissonLikelihood} = true

# GPs
extract_parameters(f::GP) = (extract_parameters(f.mean), extract_parameters(f.kernel))
function apply_parameters(f::GP, θ)
    return GP(apply_parameters(f.mean, θ[1]), apply_parameters(f.kernel, θ[2]))
end
costfunction(f::GP, data) = -logpdf(f(data.x), data.y)
_isequal(f1::GP, f2::GP) = _isequal(f1.mean, f2.mean) && _isequal(f1.kernel, f2.kernel)

extract_parameters(f::LatentGP) = (extract_parameters(f.f), extract_parameters(f.lik))
function apply_parameters(f::LatentGP, θ)
    return LatentGP(apply_parameters(f.f, θ[1]), apply_parameters(f.lik, θ[2]), f.Σy)
end

# Approximations
const SVA = SparseVariationalApproximation

function extract_parameters(sva::SVA, fixed_inducing_points::Bool)
    fz_par = fixed_inducing_points ? nothing : collect(sva.fz.x)
    q_par = extract_parameters(sva.q)
    return (fz_par, q_par)
end

function apply_parameters(sva::SVA, θ)
    fz = isnothing(θ[1]) ? sva.fz : sva.fz.f(θ[1])
    q = apply_parameters(sva.q, θ[2])
    return SVA(fz, q)
end

variational_gaussian(n::Int, T=Float64) = MvNormal(zeros(T, n), Matrix{T}(I, n, n))

# Distributions
extract_parameters(d::MvNormal) = (d.μ, ParameterHandling.positive_definite(d.Σ))
apply_parameters(::MvNormal, θ) = MvNormal(θ[1], θ[2])
_isequal(d1::MvNormal, d2::MvNormal) = isapprox(d1.μ, d1.μ) && isapprox(d1.Σ, d2.Σ)

# Custom wrappers
struct NoisyGP{T<:GP,Tn<:Real}
    gp::T
    obs_noise::Tn
end

(gp::NoisyGP)(x) = gp.gp(x, gp.obs_noise)

with_gaussian_noise(gp::GP, obs_noise::Real) = NoisyGP(gp, obs_noise)

function extract_parameters(f::NoisyGP)
    return (extract_parameters(f.gp), ParameterHandling.positive(f.obs_noise, exp, 1e-6))
end
apply_parameters(f::NoisyGP, θ) = NoisyGP(apply_parameters(f.gp, θ[1]), θ[2])
costfunction(f::NoisyGP, data) = -logpdf(f(data.x), data.y)
function _isequal(f1::NoisyGP, f2::NoisyGP)
    return _isequal(f1.gp, f2.gp) && isapprox(f1.obs_noise, f2.obs_noise)
end

struct SVGP{T<:LatentGP,Ts<:SVA}
    lgp::T
    sva::Ts
    fixed_inducing_points::Bool
end

SVGP(lgp, sva; fixed_inducing_points) = SVGP(lgp, sva, fixed_inducing_points)

function extract_parameters(f::SVGP)
    return (extract_parameters(f.lgp), extract_parameters(f.sva, f.fixed_inducing_points))
end

function apply_parameters(f::SVGP, θ)
    lgp = apply_parameters(f.lgp, θ[1])
    sva = apply_parameters(f.sva, θ[2])
    return SVGP(lgp, SVA(lgp(sva.fz.x).fx, sva.q), f.fixed_inducing_points)
end

costfunction(svgp::SVGP, data) = -elbo(svgp.sva, svgp.lgp(data.x), data.y)

end # module EasyGPs
