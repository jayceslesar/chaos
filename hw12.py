import numpy as np
import plotly.graph_objs as go

import pandas as pd


def boxcount(coordmat, maxstep):
    glob_llx = np.min(coordmat[:, 0])
    glob_lly = np.min(coordmat[:, 1])
    glob_llz = np.min(coordmat[:, 2])

    glob_urx = np.max(coordmat[:, 0])
    glob_ury = np.max(coordmat[:, 1])
    glob_urz = np.max(coordmat[:, 2])

    glob_width = glob_urx - glob_llx
    glob_height = glob_ury - glob_lly
    glob_length = glob_urz - glob_llz
    x = np.zeros((maxstep+1, 1))
    y = np.zeros((maxstep+1, 1))
    for step in range(maxstep):
        n_boxes = 0
        n_sds = 2**step
        loc_width = glob_width/n_sds
        loc_height = glob_height/n_sds
        loc_length = glob_length/n_sds

        for sd_x in range(n_sds):
            loc_llx = glob_llx + sd_x*loc_width
            loc_urx = glob_llx + (sd_x + 1)*loc_width

            found_idx = np.argwhere((coordmat[:, 0] >= loc_llx) & (coordmat[:, 0] < loc_urx))
            found_y = coordmat[found_idx, 1]

            for sd_y in range(n_sds):
                loc_lly = glob_lly + sd_y*loc_height
                loc_ury = glob_lly + (sd_y + 1)*loc_height

                found_z = np.argwhere((found_y >= loc_lly) & (found_y < loc_ury))

                for sd_z in range(n_sds):
                    loc_llz = glob_llz + sd_z*loc_length
                    loc_urz = glob_llz + (sd_y + 1)*loc_length

                    inside_idx = np.argwhere((found_z >= loc_llz) & (found_z < loc_urz))

                    if len(inside_idx) > 0:
                        n_boxes += 1

        x[step + 1] = step*np.log(2)
        y[step + 1] = np.log(n_boxes)

    A = np.zeros((maxstep + 1, 2))
    A[:, 0] = x.ravel()
    A[:, 1] = np.ones((maxstep + 1, 1)).ravel()
    Q, R = np.linalg.qr(A)
    c = np.transpose(Q).dot(y)

    val, resid, rank, s = np.linalg.lstsq(R,c, rcond=None)

    return x.ravel(), y.ravel(), round(val.ravel()[0], 3)


def lorenz(x, y, z, s=10, r=28, b=8/3):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z

    return x_dot, y_dot, z_dot


def lorenz_jacobian(x, y, z, s=10, r=28, b=8/3):
    jacobian = np.array(
        [
            [-s, s, 0],
            [-z + r, -1, -x],
            [y, x, -b]
        ]
    )
    return jacobian



def q1():
    dt = 0.01
    num_steps = 10_000
    x = np.zeros(num_steps)
    y = np.zeros(num_steps)
    z = np.zeros(num_steps)

    x[0] = -12.6480
    y[0] = -13.9758
    z[0] = 30.9758

    for i in range(num_steps - 1):
        x_dot, y_dot, z_dot = lorenz(x[i], y[i], z[i])
        x[i + 1] = x[i] + (x_dot * dt)
        y[i + 1] = y[i] + (y_dot * dt)
        z[i + 1] = z[i] + (z_dot * dt)

    P = np.zeros((num_steps, 3))
    P[:, 0] = x
    P[:, 1] = y
    P[:, 2] = z

    box_x, box_y, dim = boxcount(P, 3)

    title1 = f'Lorenz Equations'
    title2 = f'Fractal Dimension: {dim}'

    lorenz_ = go.Scatter3d(x=P[:, 0],
                            y=P[:, 1],
                            z=P[:, 2],
                            mode='markers',
                            marker=dict(size=1, color='Purple'))
    frac_dim = go.Scatter(x=box_x, y=box_y, marker=dict(size=10, color='blue'))

    lorenz_fig = go.Figure(lorenz_)
    lorenz_fig.update_layout(title='Lorenz Equations')
    lorenz_fig.show()

    frac_fig = go.Figure(frac_dim)
    frac_fig.update_layout(title=f'Fractal Dimension: {dim}')
    frac_fig.update_yaxes(title=f'log number of boxes')
    frac_fig.update_xaxes(title=f'box size')
    frac_fig.show()


def q2():
    dt = 0.01
    num_steps = 10_000
    x = np.zeros(num_steps)
    y = np.zeros(num_steps)
    z = np.zeros(num_steps)

    x[0] = -12.6480
    y[0] = -13.9758
    z[0] = 30.9758

    rs = [12, 24.5, 28]

    for R in rs:

        for i in range(num_steps - 1):
            x_dot, y_dot, z_dot = lorenz(x[i], y[i], z[i], r=R)
            x[i + 1] = x[i] + (x_dot * dt)
            y[i + 1] = y[i] + (y_dot * dt)
            z[i + 1] = z[i] + (z_dot * dt)

        w = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]
        )

        lyaps_lorenz = np.zeros((num_steps, 3))

        for i in range(num_steps):
            Z = lorenz_jacobian(x[i], y[i], z[i], r=R).dot(w)
            w, r = np.linalg.qr(Z)
            diag = np.diag(r)
            lyaps_lorenz[i, 0] = np.log(np.abs(diag[0]))
            lyaps_lorenz[i, 1] = np.log(np.abs(diag[1]))
            lyaps_lorenz[i, 2] = np.log(np.abs(diag[2]))

        l1 = np.sum(lyaps_lorenz[:, 0])/num_steps
        l2 = np.sum(lyaps_lorenz[:, 1])/num_steps
        l3 = np.sum(lyaps_lorenz[:, 2])/num_steps
        print(f"Lyapunov Exponents for r={R}")
        print(f"{l1=}, {l2=}, {l3=}")
        print()


def q3():
    file_exists=True
    if not file_exists:
        sigmas = np.arange(0, 60.1, 0.1)
        bs = np.arange(0, 10.1, 0.1)

        dt = 0.01
        num_steps = 10_000

        out_sigma = []
        out_b = []
        out_z = []

        for sigma in sigmas:
            print(f"{sigma}/{sigmas[-1]}")
            for b in bs:
                for i in range(num_steps - 1):
                    x = np.zeros(num_steps)
                    y = np.zeros(num_steps)
                    z = np.zeros(num_steps)

                    x[0] = -12.6480
                    y[0] = -13.9758
                    z[0] = 30.9758
                    x_dot, y_dot, z_dot = lorenz(x[i], y[i], z[i], s=sigma, b=b)
                    x[i + 1] = x[i] + (x_dot * dt)
                    y[i + 1] = y[i] + (y_dot * dt)
                    z[i + 1] = z[i] + (z_dot * dt)

                w = np.array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]
                    ]
                )

                lyaps_lorenz = np.zeros((num_steps, 3))

                for i in range(num_steps):
                    Z = lorenz_jacobian(x[i], y[i], z[i], s=sigma, b=b).dot(w)
                    w, r = np.linalg.qr(Z)
                    diag = np.diag(r)
                    lyaps_lorenz[i, 0] = np.log(np.abs(diag[0]))
                    lyaps_lorenz[i, 1] = np.log(np.abs(diag[1]))
                    lyaps_lorenz[i, 2] = np.log(np.abs(diag[2]))

                l1 = np.sum(lyaps_lorenz[:, 0])/num_steps
                l2 = np.sum(lyaps_lorenz[:, 1])/num_steps
                l3 = np.sum(lyaps_lorenz[:, 2])/num_steps

                lyaps = [l1, l2, l3]

                out_sigma.append(sigma)
                out_b.append(b)
                out_z.append(max(lyaps))

        df = pd.DataFrame()
        df['sigma'] = out_sigma
        df['b'] = out_b
        df['z'] = out_z
        df.to_csv('lorenz_iterate.csv')
    else:
        df = pd.read_csv('lorenz_iterate.csv')
        out_sigma = df['sigma']
        out_b = df['b']
        out_z = df['z']

    fig = go.Figure(go.Scatter3d(x=out_sigma, y=out_b, z=out_z, mode='markers', marker=dict(size=3, color='blue')))
    fig.show()



def q4():
    dr = 0.1
    r = np.arange(0, 325, dr)
    dt = 0.001
    t = np.arange(0, 10, dt)

    # initialize solution arrays
    xs = np.empty(len(t) + 1)
    ys = np.empty(len(t) + 1)
    zs = np.empty(len(t) + 1)

    xs[0], ys[0], zs[0] = (1, 1, 1)

    r_maxes = []
    z_maxes = []
    r_mins = []
    z_mins = []

    for R in r:
        print(R)
        for i in range(len(t)):
            x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], r=R)
            xs[i + 1] = xs[i] + (x_dot * dt)
            ys[i + 1] = ys[i] + (y_dot * dt)
            zs[i + 1] = zs[i] + (z_dot * dt)
        for i in range(2, len(zs) - 2):
            if zs[i - 2] < zs[i] and zs[i] > zs[i + 2]:
                r_maxes.append(R)
                z_maxes.append(zs[i])
            elif zs[i - 2] > zs[i] and zs[i] < zs[i + 2]:
                r_mins.append(R)
                z_mins.append(zs[i])

        xs[0], ys[0], zs[0] = xs[i], ys[i], zs[i]

    maxes = [val for val in z_maxes if val > 28 and val < 50]
    fig = go.Figure(go.Scatter(x=maxes[:-1], y=maxes[1:], mode='markers', marker=dict(size=2, color='blue')))
    fig.update_xaxes(title='zn')
    fig.update_yaxes(title='zn + 1')
    fig.update_layout(title='Recreation of Figure 9.3')
    fig.show()

    fig = go.Figure(go.Scatter(x=r_maxes, y=z_maxes, mode='markers', marker=dict(size=2, color='blue')))
    fig.update_xaxes(title='r')
    fig.update_yaxes(title='z')
    fig.update_layout(title='Recreation of Figure 9.4')
    fig.show()


if __name__ == '__main__':
    q2()
