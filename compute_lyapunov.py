import numpy as np

def get_L(x0,y0, function, iterates=450):
    xs = [x0]
    ys = [y0]
    f_primes = []
    for i in range(iterates):
        new_x, new_y = function(xs[i], ys[i])
        # TODO: Figure out how to calculate L in two dimensions