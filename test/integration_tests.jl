@testitem "GP without noise" begin
    kernel = 2. * with_lengthscale(SEKernel(), 1.) + 3. * Matern32Kernel() * Matern52Kernel()
    gp = GP(3., kernel)
    x = 0.01:0.01:1.
    y = rand(gp(x, 0.1))
    fitted_gp = EasyGPs.fit(gp, x, y; iterations = 1)
    @test fitted_gp isa typeof(gp)
    @test !EasyGPs._isequal(fitted_gp, gp)
end

@testitem "GP with Gaussian noise" begin
    kernel = 2. * with_lengthscale(SEKernel(), 1.) + 3. * WhiteKernel()
    gp = with_gaussian_noise(GP(3., kernel), 0.1)
    x = 0.01:0.01:1.
    y = rand(gp.gp(x, 0.1))
    fitted_gp = EasyGPs.fit(gp, x, y; iterations = 1)
    @test fitted_gp isa typeof(gp)
    @test !EasyGPs._isequal(fitted_gp, gp)
end

@testitem "Sparse variational 2d GP with Poisson likelihood" begin
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
