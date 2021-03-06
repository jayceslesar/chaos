import numpy as np
import plotly.graph_objs as go


x_j = 0
y_j = 0


def tinkerbell(x, y, c1=0.9, c2=-0.6, c3=2, c4=0.5):
    new_x = x**2 - y**2 + (c1 * x) + (c2 * y)
    new_y = (2*x*y) + (c3 * x) + (c4 * y)

    return new_x, new_y


def tinkerbell_jacobian(x, y):
    jacobian = np.array(
        [
            [2*x + 0.9, -2*y - 0.6],
            [2*y + 2, 2*x + 0.5]
        ]
    )

    return jacobian


def henon(x, y, a=1.4, b=0.3):
    new_x = 1 - a * x**2 + y
    new_y = b * x

    return new_x, new_y


def henon_jacobian(x, y):
    jacobian = np.array(
        [
            [-2*x, 0.3],
            [1, 0]
        ]
    )

    return jacobian


def ikeda(x, y, R=1, c1=0.4, c2=0.9, c3=6):
    tau = c1 - (c3 / (1 + x**2 + y**2))

    new_x = R + c2 * (x * np.cos(tau) - y * np.sin(tau))
    new_y = c2 * (x * np.sin(tau) + y * np.cos(tau))

    return new_x, new_y


def ikeda_jacobian(x, y):
    u1 = 1 - (12*x*y / (1 + x**2 + y**2)**2)
    u2 = (12*x**2 / (1 + x**2 + y**2)**2)
    u3 = 1 + (12*x*y / (1 + x**2 + y**2)**2)
    u4 = (12*y**2 / (1 + x**2 + y**2)**2)

    tau = 0.4 - (6 / (1 + x**2 + y**2))

    jacobian = np.array(
        [
            [u1*np.cos(tau) - u2*np.sin(tau), -u3*np.sin(tau) - u4*np.cos(tau)],
            [u1*np.sin(tau) + u2*np.cos(tau), u3*np.cos(tau) - u4*np.sin(tau)]
        ]
    )

    return jacobian


def iterate(func):
    num_iterates = 500_000
    x_vals = np.zeros(num_iterates)
    y_vals = np.zeros(num_iterates)

    x_vals[0] = 0.1
    y_vals[0] = 0.1

    for i in range(num_iterates - 1):
        x, y = func(x_vals[i], y_vals[i])
        x_vals[i+1] = x
        y_vals[i+1] = y

    return x_vals, y_vals


def main():
    map_funcs = [tinkerbell, henon, ikeda]
    colors = ['magenta', 'limegreen', 'blue']
    titles = ['Tinkerbell Map', 'Henon Map', 'Ikdea Map']

    for i, map_func in enumerate(map_funcs):
        x, y = iterate(map_func)
        color = colors[i]
        title = titles[i]

        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=x, y=y, mode='markers', marker=dict(size=2, color=color)))
        fig.update_layout(title=title)
        fig.show()


if __name__ == '__main__':
    main()