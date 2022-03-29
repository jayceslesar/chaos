import math
import cmath
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio


k = 10
num_iterations = 2**k

x = np.zeros(num_iterations)
y = np.zeros(num_iterations)

a_vals = [-0.5, 0, -1, 0]
b_vals = [0.3, -1, 0, 1.1]

rows = [1, 1, 2, 2]
cols = [1, 2, 1, 2]

titles = []
for a, b in zip(a_vals, b_vals):
    titles.append(f"Julia cet for c = {complex(a, b)}")

title1 = f'Bifurcation for Function ax(1−x) from a = [2, 4], step size of 10^-3.'
title2 = f'Lyapunov Number Estimation for Above Bifurcation Diagram'
fig = make_subplots(rows=2, cols=2, subplot_titles=titles)

for loc, (a, b) in enumerate(zip(a_vals, b_vals)):
    # x is the real component fo C and y is the imaginary component
    x[0] = np.real(0.5 + cmath.sqrt(0.25 - complex(a, b)))
    y[0] = np.imag(0.5 + cmath.sqrt(0.25 - complex(a, -b)))

    for i in range(num_iterations - 1):
        x1 = x[i]
        y1 = y[i]

        # euclidian distance between real and imaginary
        u = np.sqrt((x1 - a)**2 + (y1 - b)**2) / 2
        # subtract real component by initial condition, midpoint
        v = (x1 - a) / 2

        # positive and negative distance between our real and imaginary distances
        u1 = np.sqrt(u + v)
        v1 = np.sqrt(u - v)

        x[i+1] = u1
        y[i+1] = v1

        # turn our imaginary values positive if they are less than our imaginary IC
        if y[i] < b:
            y[i+1] = -y[i+1]

        # if np.random.rand() < 0.5:  # different orbits
        #     x[i+1] = -x[i+1]
        #     y[i+1] = -y[i+1]

    row, col = rows[loc], cols[loc]
    fig.append_trace(go.Scatter(x=x, y=y, mode='markers', showlegend=False), row=row, col=col)

fig.show()
