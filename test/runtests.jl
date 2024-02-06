using AutoGPs
using Test

@testset "Integration tests" begin
    @testset "GP with Gaussian noise" begin
        kernel = 2. * with_lengthscale(SEKernel(), 1.) + 3. * Matern32Kernel() * Matern52Kernel()
        gp = with_gaussian_noise(GP(3., kernel), 0.1)
        model, θ = AutoGPs.parameterize(gp)
        @test AutoGPs._isequal(model(θ), gp)

        x = 0.01:0.01:1.
        y = rand(gp.gp(x, 0.1))
        fitted_gp = AutoGPs.fit(gp, x, y)
        @test fitted_gp isa typeof(gp)
        @test !AutoGPs._isequal(fitted_gp, gp)
    end
end
