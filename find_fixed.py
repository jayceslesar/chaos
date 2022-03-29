import numpy as np
import plotly.graph_objs as go
from fractions import Fraction


def f(x):
    if x == float('nan'):
        return x

    if 0 <= x < 1/3:
        return 3 * x

    if x == 1/3:
        return float('nan')

    if 1/3 < x <= 1:
        return 1.5 * (1 - x)



nums = np.arange(0.0, 1.0, 0.00001)

x = nums
tr1 = []
tr2 = []


tol = 10**-4

# 0, 3/5
for num in nums:
    fx = f(num)
    tr1.append(fx)
    ffx = f(fx)
    tr2.append(ffx)
    if fx != 'none':
        fx = round(fx, 10)
        ffx = round(ffx, 10)
        if abs(num - ffx) < tol:
            print(num)


fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=x, name='y=x', mode='markers'))
fig.add_trace(go.Scatter(x=x, y=tr1, name='f(x)', mode='markers'))
fig.add_trace(go.Scatter(x=x, y=tr2, name='f(f(x))', mode='markers'))
fig.show()
