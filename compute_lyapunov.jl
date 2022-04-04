map(x) = 2.5*x*(1-x)
map¹(x) = 2.5 - 5*x

function get_L(x₀, iterates=450)
    xs = [x₀]
    mag_slopes = []
    for i=1:iterates
        append!(xs, map(xs[i]))
        append!(mag_slopes, abs(map¹(xs[i])))
    end
    return xs, mag_slopes
end

x₀s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
L_exp = []
for x₀ in x₀s
    xs, mags = get_L(x₀)
    L_num = prod(mags)^(1/450)
    append!(L_exp, log(L_num))
end
println(L_exp)