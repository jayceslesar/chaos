import numpy as np


def func(i, a):
    return a*i*(1-i)


def primitive():
    """Brute Force, copy of matlab logistic_period.m"""
    iterations = 10**7
    a = 1 + np.sqrt(6) + 0.1200795
    orbits = np.zeros(iterations, )
    orbits[0] = np.random.uniform(0, 1)
    for i in range(1, iterations):
        new_val = func(orbits[i - 1], a)
        orbits[i] = new_val

    S = 20
    tol = 10**(-4)

    for i in range(S):
        check_index = -1 - 2**i
        diff = np.abs(orbits[-1] - orbits[check_index])
        if diff <= tol:
            print(2**i)


def func(i, a):
    return a*i*(1-i)


def approx_diff(a):
    test = np.zeros(100, )
    test[0] = np.random.uniform(0, 1)
    for i in range(1, len(test)):
        test[i] = func(test[i-1], a)

    print(np.abs(np.diff(np.diff(test))))
    return np.mean(np.diff(np.diff(test)))
    # need to integrate this with scipy?
    # can avoid computation with this


def better():
    """Optimal Process Derived From Above."""
    N = 27
    a = 1 + np.sqrt(6)
    tolerance = 10**-6
    increment = 0.000000000000001

    acceleration = approx_diff(a)
    print(acceleration)
    return

    upper = 10**15

    done = False
    while not done:
        a = a + increment
        end = func(upper, a)
        for i in range(N):
            generated = func(2**i, a)
            diff = np.abs(end - generated)
            print(a, diff, tolerance)
            if diff < tolerance:
                print(f"orbit with a {a} repeats every 2^{i}")
                done = True


if __name__ == '__main__':
    better()
