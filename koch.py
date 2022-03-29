import numpy as np

def koch():
    for k in range(1, 7):
        mmax = 4**k
        x = np.zeros((mmax))
        y = np.zeros((mmax))
        h = 3**-k
        angles = [
            0,
            np.pi/3,
            -np.pi/3,
            0
        ]

        for a in range(1, mmax):
            m = a - 1
            ang = 0

            for b in range(k):
                seg = np.mod(m, 4)
                m = np.floor(m / 4)
                ang += angles[int(seg)]

            x[a] = x[a-1] + h * np.cos(ang)
            y[a] = y[a-1] + h * np.sin(ang)


    P = np.zeros((len(x), 2))
    P[:, 0] = x
    P[:, 1] = y

    return P
