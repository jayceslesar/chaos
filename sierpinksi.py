import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go

def sierpinksi():
    A=[0,0]
    B=[4,0]
    C=[2, 2*np.sqrt(3)]

    n_max = 75_000

    P = np.zeros((n_max, 2))

    scale = 0.5

    for i in range(n_max - 1):
        r = np.random.rand()

        if r < 1/3:
            P[i+1, :] =  P[i, :] + np.multiply((A - P[i, :]), scale)
        elif r < 2/3:
            P[i+1, :] =  P[i, :] + np.multiply((B - P[i, :]), scale)
        else:
            P[i+1, :] =  P[i, :] + np.multiply((C - P[i, :]), scale)

    return P
