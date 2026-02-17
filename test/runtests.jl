using Test
using Random
using OSDDistributions
using ExponentialUtilities

@testset "OSD basic" begin
    M = [-1.0 1.0; -0.3 -0.2]
    cp = compound_beta(2; λ = [1.0, 0.5], α = [1.0, 1.0], β = [1.0, 1.0])
    osd_cp = OSDDistributions.osd(M, cp)
    Random.seed!(123)
    X = rand(osd_cp, 5, 50; method=:eigen)
    @test size(X) == (5, 2)
    X2 = rand(osd_cp, 3, 30; method=:naive)
    @test size(X2) == (3, 2)
end