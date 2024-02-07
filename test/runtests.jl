using AutoGPs
import ParameterHandling
using Test

@testset "AutoGPs" begin

@testset "Unit tests for AutoGPs._isequal(::$(typeof(o1)), ::$(typeof(o2)))" for (o1, o2) in (
    SEKernel() => SEKernel(AutoGPs.KernelFunctions.SqEuclidean()),
    Matern32Kernel() => Matern32Kernel(AutoGPs.KernelFunctions.SqEuclidean()),
    Matern52Kernel() => Matern52Kernel(AutoGPs.KernelFunctions.SqEuclidean()),
)
    @test AutoGPs._isequal(o1, o1)
    @test AutoGPs._isequal(o2, o2)
    @test !AutoGPs._isequal(o1, o2)
end

@testset "Unit tests for AutoGPs.parameterize" begin
    @testset "$(typeof(object))" for object in (
        ZeroMean(), ConstMean(1.),
        SEKernel(), Matern32Kernel(), Matern52Kernel(),
        with_lengthscale(SEKernel(), 2.),
        2. * SEKernel(), 3. * SEKernel() + 2. * Matern32Kernel(),
        2. * Matern32Kernel() * SEKernel(),
        2. * with_lengthscale(SEKernel(), 1.) + 3. * Matern32Kernel() * Matern52Kernel(),
        BernoulliLikelihood()
    )
        model, θ = AutoGPs.parameterize(object)
        new_object = @inferred model(θ)
        @test AutoGPs._isequal(model(θ), object)

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
        fitted_gp = AutoGPs.fit(gp, x, y; iterations = 1)
        @test fitted_gp isa typeof(gp)
        @test !AutoGPs._isequal(fitted_gp, gp)
    end
    @testset "GP with Gaussian noise" begin
        kernel = 2. * with_lengthscale(SEKernel(), 1.) + 3. * Matern32Kernel() * Matern52Kernel()
        gp = with_gaussian_noise(GP(3., kernel), 0.1)
        x = 0.01:0.01:1.
        y = rand(gp.gp(x, 0.1))
        fitted_gp = AutoGPs.fit(gp, x, y; iterations = 1)
        @test fitted_gp isa typeof(gp)
        @test !AutoGPs._isequal(fitted_gp, gp)
    end
end

end
