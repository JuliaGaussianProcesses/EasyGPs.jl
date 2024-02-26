using EasyGPs

using Test
using TestItems
using TestItemRunner
@run_package_tests

include("unit_tests.jl")
include("integration_tests.jl")
