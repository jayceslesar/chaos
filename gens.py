"""Generators."""

import plotly.graph_objs as go
import plotly.express as px
import numpy as np


def generate_func_numbers(iterations: int = 100, r=3):
    out = []
    for x in range(iterations):
        math = r*x*(1-x)
        out.append(math)

    return out

iters = 20
x = [i for i in range(iters)]
y1 = generate_func_numbers(iters)
y2 = np.diff(np.asarray(y1))
y3 = np.diff(y2)

fig = go.Figure()
print(x)
fig.add_trace(go.Scatter(x=x, y=x))
fig.add_trace(go.Scatter(x=x, y=y1))

fig.show()