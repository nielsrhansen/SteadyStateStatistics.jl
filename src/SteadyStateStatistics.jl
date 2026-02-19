module SteadyStateStatistics

using LinearAlgebra 
using Distributions # For Exponential, Gamma, Beta
using StatsBase # For Weights and sample
using Plots
using Tullio # For efficient tensor operations
using Cumulants # For efficient cumulant estimation
using ExponentialUtilities # For matrix exponentials 
import Base: rand

include("jump_process.jl")
include("distributions.jl")
include("osd.jl")
include("estimation.jl")

export jump_process, compound_poisson, compound_gamma, compound_signed_gamma, compound_beta, linear_estimator, quadratic_estimator, two_step_estimator, quadratic_estimator_4, two_step_shrinkage_estimator
export osd, rand, plot

end # module SteadyStateStatistics