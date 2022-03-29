import math
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

np.random.seed(42)


title1 = f'Bifurcation for Function ax(1−x) from a = [2, 4], step size of 10^-3.'
title2 = f'Lyapunov Number Estimation for Above Bifurcation Diagram'

fig = make_subplots(rows=2, cols=1, subplot_titles=(title1, title2))


amin = 2
amax = 4
b = -0.3

N = 10**4  # steps
da = 10**-3 # step size

initial_conditions = [np.random.rand(), np.random.rand()]

x = np.zeros(N)
y = np.zeros(N)

a_space = np.arange(amin, amax, da).round(4)

lyp = []
first = go.Figure()
for i, a in enumerate(a_space):
    x[0] = initial_conditions[0]
    y[0] = initial_conditions[1]
    for i in range(N-1):
         x[i+1] = a*x[i] / (1 - x[i])
         y[i+1] = x[i]

    S = 10
    p = S
    tol = 10**(-10)
    for i in range(S):
        diff = np.linalg.norm(x[-1] - x[-1 - 2**i])
        if diff < tol:
            p = i
            break

    to_plot = y[-1 - 2**p:]
    fig.append_trace(go.Scatter(x=[a for i in range(len(to_plot))], y=to_plot, mode='markers', marker=dict(size=1, color='Red'), showlegend=False), row=1, col=1)
    # estimate = np.mean(np.log(abs(np.diff(x[101:]))))
    other_estimate = np.mean(np.log(abs(a*(1-2*x[101:]))))
    if np.isinf(other_estimate):
        other_estimate = float('nan')
        lyp.append(other_estimate)
    else:
    # new_estimate = np.linalg.norm(np.diff(y[101:]))**1/a
        lyp.append(math.e**other_estimate)
    print(a, other_estimate)


fig.append_trace(go.Scatter(x=a_space, y=lyp, mode='markers', marker=dict(size=4, color='Red'), showlegend=False), row=2, col=1)
fig.append_trace(go.Scatter(x=a_space, y=[1 for _ in a_space], mode='lines', line=dict(color='Blue', width=2), showlegend=False), row=2, col=1)
fig.show()

# fig.write_image('hello.png', width=800, height=1000, format='png')
