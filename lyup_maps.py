import numpy as np


def tinkerbell(x, y, c1=-0.3, c2=-0.6, c3=2, c4=0.5):
    new_x = x**2 - y**2 + (c1 * x) + (c2 * y)
    new_y = (2*x*y) + (c3 * x) + (c4 * y)

    return new_x, new_y


def henon(x, y, a=1.4, b=0.3):
    new_x = 1 - a * x**2 + y
    new_y = b * x

    return new_x, new_y


def ikeda(x, y, R=1, c1=0.4, c2=0.9, c3=6):
    tau = (c1 - c2) / (1 + x**2 + y**2)
    pass