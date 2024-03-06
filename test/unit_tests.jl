@testitem "_isequal" begin
    for (o1, o2) in (
        SEKernel() => SEKernel(EasyGPs.KernelFunctions.SqEuclidean()),
        Matern32Kernel() => Matern32Kernel(EasyGPs.KernelFunctions.SqEuclidean()),
        Matern52Kernel() => Matern52Kernel(EasyGPs.KernelFunctions.SqEuclidean()),
    )
        @test EasyGPs._isequal(o1, o1)
        @test EasyGPs._isequal(o2, o2)
        @test !EasyGPs._isequal(o1, o2)
    end
end

@testitem "parameterize" begin
    import ParameterHandling
    for object in (
        ZeroMean(), ConstMean(1.0),
        SEKernel(), Matern32Kernel(), Matern52Kernel(),
        with_lengthscale(SEKernel(), 2.0),
        2.0 * SEKernel(), 3.0 * SEKernel() + 2.0 * Matern32Kernel(),
        2.0 * Matern32Kernel() * SEKernel(),
        2.0 * with_lengthscale(SEKernel(), 1.0) + 3.0 * Matern32Kernel() * Matern52Kernel(),
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
