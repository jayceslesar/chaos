import numpy as np
import plotly.graph_objs as go
import plotly.express as px

amin = 1
amax = 2
b = -0.3

N = 10**4  # steps
da = 10**-3 # step size

initial_conditions = [0, 2]

x = np.zeros(N)
y = np.zeros(N)

a_space = np.arange(amin, amax, da).round(4)


fig = go.Figure()
for a in a_space:
    x[0] = initial_conditions[0]
    y[0] = initial_conditions[1]
    for i in range(N-1):
         x[i+1] = a - x[i]**2 + b*y[i]
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
    fig.add_trace(go.Scatter(x=[a for i in range(len(to_plot))], y=to_plot, mode='markers', marker=dict(size=1, color='Red'), showlegend=False))

fig.show()
