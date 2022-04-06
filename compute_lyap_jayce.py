import numpy as np
import math
from scipy import linalg
from lyap_maps import henon, henon_jacobian, tinkerbell, tinkerbell_jacobian, ikeda, ikeda_jacobian

np.set_printoptions(suppress=True)


def get_lyapunov(map_func, func_jacobian):
    num_iterates = 5000
    x_vals = np.zeros(num_iterates)
    y_vals = np.zeros(num_iterates)

    w1 = np.array(
        [
            [1],
            [0]
        ]
    )
    w2 = np.array(
        [
            [0],
            [1]
        ]
    )

    w1s = [w1]
    w2s = [w2]

    xs = [0.1]
    ys = [0.1]

    r1s = []
    r2s = []


    for i in range(num_iterates):
        jacobian = func_jacobian(xs[i], ys[i])
        z1 = jacobian.dot(w1s[i])
        z2 = jacobian.dot(w2s[i])

        y1 = z1
        y2 = z2 - z1*(y1/np.linalg.norm(y1))*(y1/np.linalg.norm(y1)) #NOTE: Second mult might be dot

        w1s.append(y1/np.linalg.norm(y1))
        w2s.append(y2/np.linalg.norm(y2))

        r1s.append(np.linalg.norm(y1))
        r2s.append(np.linalg.norm(y2))

        x, y = map_func(xs[i], ys[i])
        xs.append(x)
        ys.append(y)

    # print(np.log(np.abs(np.asarray(r1s))).sum()*(1/num_iterates))
    print(np.log(np.asarray(r1s).prod()**(1/num_iterates)))
    print(np.log(np.asarray(r2s).prod()**(1/num_iterates)))


get_lyapunov(henon, henon_jacobian)
# get_lyapunov(tinkerbell, tinkerbell_jacobian)
# get_lyapunov(ikeda, ikeda_jacobian)