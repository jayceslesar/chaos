import numpy as np
import math
from scipy import linalg
from lyap_maps import henon, henon_jacobian

np.set_printoptions(suppress=True)


def get_lyapunov(map_func, func_jacobian):
    num_iterates = 5000
    x_vals = np.zeros(num_iterates)
    y_vals = np.zeros(num_iterates)
    w = np.array(
        [
            [1, 0],
            [0, 1]
        ])
    ws = [w]
    xs = [0.1]
    ys = [0.1]
    r1s = []
    r2s = []


    for i in range(num_iterates - 1):
        x, y = map_func(xs[i], ys[i])
        jacobian = func_jacobian(xs[i], ys[i])
        z = jacobian.dot(ws[i])
        q, r = np.linalg.qr(z)
        ws.append(q)
        r1s.append(abs(r[0][0]))
        r2s.append(abs(r[1][1]))
        xs.append(x)
        ys.append(y)

    # print(np.asarray(r1s).sum()**(1/num_iterates))
    # print(np.prod(np.asarray(r2s))**(1/num_iterates))


    print(np.asarray(r1s).prod())
    print(np.asarray(r2s).prod())


get_lyapunov(henon, henon_jacobian)