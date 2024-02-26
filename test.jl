using AbstractGPs
using Enzyme
import Zygote.gradient as zygote_gradient

x = rand(1000)
y = rand(1000)

build_gp(θ) = GP(θ.var * with_lengthscale(SEKernel(), θ.l))
θ = (var = 1., l = 1., obs_noise = 0.1)

costfun(θ, x, y) = logpdf(build_gp(θ)(x, θ.obs_noise), y)

function enzyme_gradient(costfun, θ, x, y)
    dx = make_zero(x)
    dy = make_zero(y)
    ((dθ, _, _),) = Enzyme.autodiff(Reverse, costfun, Active(θ), Duplicated(x, dx), Duplicated(y, dy))
    return (dθ, dx, dy)
end

dθ, dx, dy = enzyme_gradient(costfun, θ, x, y)
dθ_, dx_, dy_ = zygote_gradient(costfun, θ, x, y)

values(dθ) .≈ values(dθ_) # (true, true, true)
dx ≈ dx_ # true
dy ≈ dy_ # true

using BenchmarkTools

@btime costfun($θ, $x, $y); # 9.006 ms (20 allocations: 38.19 MiB)
@btime enzyme_gradient(costfun, $θ, $x, $y); # 38.544 ms (180 allocations: 99.29 MiB)
@btime zygote_gradient(costfun, $θ, $x, $y); # 43.990 ms (493 allocations: 244.31 MiB)
