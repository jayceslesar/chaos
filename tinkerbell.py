import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def tinkerbell(x, y, c1=-0.3, c2=-0.6, c3=2, c4=0.5):
    new_x = x**2 - y**2 + (c1 * x) + (c2 * y)
    new_y = (2*x*y) + (c3 * x) + (c4 * y)

    return new_x, new_y


def iterate(c4):
    num_iterates = 50_000
    x_vals = np.zeros(num_iterates)
    y_vals = np.zeros(num_iterates)

    x_vals[0] = 0.1
    y_vals[0] = 0.1

    for i in range(num_iterates - 1):
        x, y = tinkerbell(x_vals[i], y_vals[i], c4=c4)
        x_vals[i+1] = x
        y_vals[i+1] = y

    return x_vals, y_vals


def fourb():
    c4s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'limegreen', 'magenta', 'gray', 'black']
    fig = make_subplots(rows=1, cols=2,  subplot_titles=("Tinkerbell Map Scatterplot", f"Tinkerbell Map Lineplot"))

    for c4, color in zip(c4s, colors):
        x, y = iterate(c4)
        fig.add_trace(go.Scattergl(x=x, y=y, mode='markers', name=f'c4={c4}', marker=dict(size=5, color=color), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scattergl(x=x, y=y, mode='lines', name=f'c4={c4}', marker=dict(size=5, color=color)), row=1, col=2)

    fig.update_layout(showlegend=True)
    fig.show()


fourb()