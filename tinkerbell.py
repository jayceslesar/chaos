import numpy as np
import plotly.graph_objs as go


def tinkerbell(x, y, c1=-0.3, c2=-0.6, c3=2, c4=0.5):
    new_x = x**2 - y**2 + (c1 * x) + (c2 * y)
    new_y = (2*x*y) + (c3 * x) + (c4 * y)

    return new_x, new_y


def iterate():
    num_iterates = 50_000
    x_vals = np.zeros(num_iterates)
    y_vals = np.zeros(num_iterates)

    x_vals[0] = 0.1
    y_vals[0] = 0.1

    for i in range(num_iterates - 1):
        x, y = tinkerbell(x_vals[i], y_vals[i])
        print(x, y)
        x_vals[i+1] = x
        y_vals[i+1] = y

    return x_vals, y_vals


if __name__ == '__main__':
    x, y = iterate()
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=x, y=y, mode='markers', marker=dict(size=2, color='magenta')))
    fig.show()
