import numpy as np


amin = 3.8
amax = 4
b = -0.3

N = 10**3  # steps
da = 10**-3 # step size

initial_conditions = [np.random.rand(), np.random.rand()]

x = np.zeros(N)
y = np.zeros(N)

a_space = np.arange(amin, amax, da).round(4)
for i, a in enumerate(a_space):
    x[0] = initial_conditions[0]
    y[0] = initial_conditions[1]
    for i in range(N-1):
         x[i+1] = a*x[i] * (1 - x[i])
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
    data = [a for i in range(len(to_plot))]

    if 3.961 in data:  # manually find on plot
        print(len(to_plot))
