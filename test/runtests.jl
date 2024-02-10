using EasyGPs
import KernelFunctions
import ParameterHandling
using Test

@testset "EasyGPs" begin

@testset "Unit tests for EasyGPs._isequal(::$(typeof(o1)), ::$(typeof(o2)))" for (o1, o2) in (
    SEKernel() => SEKernel(EasyGPs.KernelFunctions.SqEuclidean()),
    Matern32Kernel() => Matern32Kernel(EasyGPs.KernelFunctions.SqEuclidean()),
    Matern52Kernel() => Matern52Kernel(EasyGPs.KernelFunctions.SqEuclidean()),
)
    @test EasyGPs._isequal(o1, o1)
    @test EasyGPs._isequal(o2, o2)
    @test !EasyGPs._isequal(o1, o2)
end

@testset "Unit tests for EasyGPs.parameterize" begin
    @testset "$(typeof(object))" for object in (
        ZeroMean(), ConstMean(1.),
        SEKernel(), Matern32Kernel(), Matern52Kernel(),
        with_lengthscale(SEKernel(), 2.),
        2. * SEKernel(), 3. * SEKernel() + 2. * Matern32Kernel(),
        2. * Matern32Kernel() * SEKernel(),
        2. * with_lengthscale(SEKernel(), 1.) + 3. * Matern32Kernel() * Matern52Kernel(),
        BernoulliLikelihood(),
        PoissonLikelihood()
    )
        model, θ = EasyGPs.parameterize(object)
        new_object = @inferred model(θ)
        @test EasyGPs._isequal(model(θ), object)

        # Type stability in combination with ParameterHandling
        par, unflatten = ParameterHandling.flatten(θ)
        @test (@inferred unflatten(par)) == θ
    end
end

@testset "Integration tests" begin
    @testset "GP without noise" begin
        kernel = 2. * with_lengthscale(SEKernel(), 1.) + 3. * Matern32Kernel() * Matern52Kernel()
        gp = GP(3., kernel)
        x = 0.01:0.01:1.
        y = rand(gp(x, 0.1))
        fitted_gp = EasyGPs.fit(gp, x, y; iterations = 1)
        @test fitted_gp isa typeof(gp)
        @test !EasyGPs._isequal(fitted_gp, gp)
    end
    @testset "GP with Gaussian noise" begin
        kernel = 2. * with_lengthscale(SEKernel(), 1.) + 3. * Matern32Kernel() * Matern52Kernel()
        gp = with_gaussian_noise(GP(3., kernel), 0.1)
        x = 0.01:0.01:1.
        y = rand(gp.gp(x, 0.1))
        fitted_gp = EasyGPs.fit(gp, x, y; iterations = 1)
        @test fitted_gp isa typeof(gp)
        @test !EasyGPs._isequal(fitted_gp, gp)
    end
    @testset "Sparse variational 2d GP with Poisson likelihood" begin
        kernel = 1. * SEKernel()
        lgp = LatentGP(GP(0.0, kernel), PoissonLikelihood(), 1e-6)
        x = rand(100, 2) |> RowVecs
        y = round.(Int, 10 .* sum.(abs2, x))
        z = x[begin:5:end]
        sva = SVA(lgp(z).fx, variational_gaussian(length(z)))
        svgp = SVGP(lgp, sva; fixed_inducing_points = true)
        fitted_svgp = EasyGPs.fit(svgp, x, y; iterations = 1)
        @test fitted_svgp isa typeof(svgp)
        @test !EasyGPs._isequal(fitted_svgp, svgp)
        @test fitted_svgp.sva.fz.x === z
    end
end

end
