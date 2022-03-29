using Distributions
# r₁ = 1.266569242783178 repeats every 2¹
# r₂ = 1.819469242783178 repeats every 2²
# r₃ = 1.9275692427831779 repeats every 2³
# r₄ = 1.945669242783178 repeats every 2⁴
# r₅ = 1.95112341245325143 repeats every 2⁵
# r₆ = 1.95152341245325143 repeats every 2⁶
a = 1.95152341245325143
b = -0.3
println("a:", a)
iterations = 1000
N = 6
IC = [0;2];
x = zeros(iterations+1,1); y = zeros(iterations+1,1);
x[1] = IC[1]
y[1] = IC[2]

println("Generating Map...")
for i = 1:iterations
    x[i+1] = a-x[i]^2+b*y[i];
    y[i+1] = x[i];
end
println("Map Generated!")

tolerance = 10^-4
setprecision(1_000_000)
for i in 0:N
    index = 2^i
    val = abs(BigFloat(last(x)) - BigFloat(x[size(x)[1] - index]))
    # println(val)
    if val < tolerance
        println("orbit repeats every $index or 2^$i")
    end
end

rs = [1.266569242783178, 1.819469242783178, 1.9275692427831779, 1.945669242783178, 1.95112341245325143, 1.95152341245325143]
fs = []
for i in 1:length(rs)-2
    println(i)
    r1 = rs[i]
    r2 = rs[i+1]
    r3 = rs[i+2]
    append!(fs, (r2 - r1)/(r3-r2))
end
println(fs)