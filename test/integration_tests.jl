@testitem "GP without noise" begin
    kernel =
        2.0 * with_lengthscale(SEKernel(), 1.0) + 3.0 * Matern32Kernel() * Matern52Kernel()
    gp = GP(3.0, kernel)
    x = 0.01:0.01:1.0
    y = rand(gp(x, 0.1))
    fitted_gp = EasyGPs.fit(gp, x, y; iterations=1)
    @test fitted_gp isa typeof(gp)
    @test !EasyGPs._isequal(fitted_gp, gp)
end

@testitem "GP with Gaussian noise" begin
    kernel = 2.0 * with_lengthscale(SEKernel(), 1.0) + 3.0 * WhiteKernel()
    gp = with_gaussian_noise(GP(3.0, kernel), 0.1)
    x = 0.01:0.01:1.0
    y = rand(gp.gp(x, 0.1))
    fitted_gp = EasyGPs.fit(gp, x, y; iterations=1)
    @test fitted_gp isa typeof(gp)
    @test !EasyGPs._isequal(fitted_gp, gp)
end

@testitem "GP with Gaussian noise and custom constraints" begin
    kernel = 1.0 * with_lengthscale(SEKernel(), fixed(2.))
    gp = with_gaussian_noise(GP(3., kernel), fixed(0.1))
    x = 0.01:0.01:1.
    y = rand(gp.gp(x, 0.1))
    fitted_gp = EasyGPs.fit(gp, x, y; iterations = 1)
    @test fitted_gp isa typeof(gp)
    @test !EasyGPs._isequal(fitted_gp, gp)

    _, θ1 = EasyGPs.parameterize(gp)
    _, θ2 = EasyGPs.parameterize(fitted_gp)

    θ1 = EasyGPs.value(θ1)
    θ2 = EasyGPs.value(θ2)
    
    # Check that parameters we asked to remain fixed are fixed, and the
    @test θ1[1][2][1][2] ≈ θ2[1][2][1][2]
    @test θ1[2] ≈ θ2[2]

    # Check that the other parameters have changed
    @test θ1[1][1] != θ2[1][1]
    @test θ1[1][2][2] != θ2[1][2][2]
end

@testitem "Sparse variational 2d GP with Poisson likelihood" begin
    kernel = 1.0 * SEKernel()
    lgp = LatentGP(GP(0.0, kernel), PoissonLikelihood(), 1e-6)
    x = RowVecs(rand(100, 2))
    y = round.(Int, 10 .* sum.(abs2, x))
    z = x[begin:5:end]
    sva = SVA(lgp(z).fx, variational_gaussian(length(z)))
    svgp = SVGP(lgp, sva; fixed_inducing_points=true)
    fitted_svgp = EasyGPs.fit(svgp, x, y; iterations=1)
    @test fitted_svgp isa typeof(svgp)
    @test !EasyGPs._isequal(fitted_svgp, svgp)
    @test fitted_svgp.sva.fz.x === z
end
