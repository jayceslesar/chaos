import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt


def T(P, a, b, c, d, e, f):
    F = np.zeros((2))
    F[0] = a*P[0] + b*P[1] + c
    F[1] = d*P[0] + e*P[1] + f

    return F

def fern():
    N = 75_000
    P = np.zeros((N, 2))
    P[0, :] = [0.5, 0.5]

    for i in range(N-1):
        r = np.random.rand()

        if r < 0.05:
            P[i+1, :] = T(P[i, :], 0, 0, 0, 0, .2, 0)
        elif r < .86:
            P[i+1, :] = T(P[i, :], .85, .05, 0, -.04, .85, 1.6)
        elif r < .93:
            P[i+1, :] = T(P[i, :], .2, -.26, 0, .23, .22, 1.6)
        else:
            P[i+1, :] = T(P[i, :], -.15, .28, 0, .26, .24, .44)

    return P
