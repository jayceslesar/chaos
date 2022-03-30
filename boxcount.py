
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


from fern import fern
from koch import koch
from sierpinksi import sierpinksi


def boxcount(coordmat, maxstep):
    glob_llx = np.min(coordmat[:, 0])
    glob_lly = np.min(coordmat[:, 1])

    glob_urx = np.max(coordmat[:, 0])
    glob_ury = np.max(coordmat[:, 1])

    glob_width = glob_urx - glob_llx
    glob_height = glob_ury - glob_lly
    x = np.zeros((maxstep+1, 1))
    y = np.zeros((maxstep+1, 1))

    for step in range(maxstep):
        n_boxes = 0
        n_sds = 2**step
        loc_width = glob_width/n_sds
        loc_height = glob_height/n_sds

        for sd_x in range(n_sds):
            loc_llx = glob_llx + sd_x*loc_width
            loc_urx = glob_llx + (sd_x + 1)*loc_width

            found_idx = np.argwhere((coordmat[:, 0] >= loc_llx) & (coordmat[:, 0] < loc_urx))
            found_y = coordmat[found_idx, 1]

            for sd_y in range(n_sds):
                loc_lly = glob_lly + sd_y*loc_height
                loc_ury = glob_lly + (sd_y + 1)*loc_height

                inside_idx = np.argwhere((found_y >= loc_lly) & (found_y < loc_ury))

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


P = fern()
x = P[:, 0]
y = P[:, 1]

box_x, box_y, dim = boxcount(P, 10)

fig = make_subplots(rows=1, cols=2,  subplot_titles=("Barnsley's Fern", f"Fractal Dimension: {dim}"))
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=2, color='magenta')), row=1, col=1)
fig.add_trace(go.Scatter(x=box_x, y=box_y, marker=dict(size=10, color='magenta')), row=1, col=2)
fig.update_traces(showlegend=False)
fig['layout']['xaxis2']['title'] = 'n * ln(2)'
fig['layout']['yaxis2']['title'] = 'ln(n_boxes)'
fig.show()


P = sierpinksi()
x = P[:, 0]
y = P[:, 1]

box_x, box_y, dim = boxcount(P, 10)

fig = make_subplots(rows=1, cols=2,  subplot_titles=("Sierpinski Triangle", f"Fractal Dimension: {dim}"))
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=2, color='limegreen')), row=1, col=1)
fig.add_trace(go.Scatter(x=box_x, y=box_y, marker=dict(size=10, color='limegreen')), row=1, col=2)
fig.update_traces(showlegend=False)
fig['layout']['xaxis2']['title'] = 'n * ln(2)'
fig['layout']['yaxis2']['title'] = 'ln(n_boxes)'
fig.show()


P = koch()
x = P[:, 0]
y = P[:, 1]

box_x, box_y, dim = boxcount(P, 10)

fig = make_subplots(rows=1, cols=2,  subplot_titles=("Koch Curve", f"Fractal Dimension: {dim}"))
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=3, color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=box_x, y=box_y, marker=dict(size=10, color='blue')), row=1, col=2)
fig.update_traces(showlegend=False)
fig['layout']['xaxis2']['title'] = 'n * ln(2)'
fig['layout']['yaxis2']['title'] = 'ln(n_boxes)'
fig.show()
