# import Pkg; Pkg.add("Distributions")
using Random
using Distributions
Random.seed!(1234)
setprecision(500_000)  # 10k bits for math...

a = 1+sqrt(6)+0.1200795

iterations = 100_000
N = 10  # can keep going but will take a hot minute... roughly 97 seconds for 2^29

logistic_map = Vector{Float64}(undef, iterations)
logistic_map[1] = rand(Float64)

println("Generating Map...")
for i in 2:iterations
    logistic_map[i] = a*logistic_map[i-1]*(1-logistic_map[i-1])
end
println("Map Generated!")

tolerance = 10^-8
for i in 0:N
    index = 2^i
    val = abs(BigFloat(last(logistic_map)) - BigFloat(logistic_map[size(logistic_map)[1] - index]))
    println(BigFloat(logistic_map[size(logistic_map)[1] - index]), " at 2^$i")
    # if val < tolerance
    #     println("orbit repeats every $index or 2^$i")
    # end
end
