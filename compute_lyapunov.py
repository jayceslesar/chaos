import numpy as np
from lyap_maps import henon, henon_jacobian

def get_L(x0,y0, function, iterates=450):
    xs = [x0]
    ys = [y0]
    f_primes = []
    for i in range(iterates):
        new_x, new_y = function(xs[i], ys[i])
        xs.append(new_x)
        ys.append(new_y)

        dx = xs[i+1]-xs[i]
        dy = ys[i+1]-ys[i]
        dydx = np.gradient(dy, dx)
        f_primes.append(abs(dydx))
        # TODO: Fix this so that is actually finding the L per partial differential
    L_num = np.prod(f_primes)
    L_exp = np.log(L_num**(1/iterates))

    return L_exp
