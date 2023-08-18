""" 2D Poisson Equation """

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


def plot2d(x, y, p):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y = np.meshgrid(x, y)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.view_init(30, 225)
    ax.plot_surface(X, Y, p[:], cmap=cm.viridis)

    plt.show()


def poisson2d(p, b, dx, dy, nt):
    for _ in range(nt):
        pn = p.copy()
        p[1:-1, 1:-1] = (
            dy**2 * (pn[1:-1, 2:] + pn[1:-1, :-2])
            + dx**2 * (pn[2:, 1:-1] + pn[:-2, 1:-1])
            - b[1:-1, 1:-1] * dx**2 * dy**2
        ) / (2 * (dx**2 + dy**2))

        p_init[:, 0] = 0
        p_init[:, -1] = 0
        p_init[0, :] = 0
        p_init[-1, :] = 0

    return p


nx = 50
ny = 50
nt = 100
dx = 2 / (nx - 1)
dy = 1 / (ny - 1)

x = np.linspace(0, 2, nx)
y = np.linspace(0, 1, ny)

p_init = np.zeros((ny, nx))

p_init[:, 0] = 0
p_init[:, -1] = 0
p_init[0, :] = 0
p_init[-1, :] = 0

plot2d(x, y, p_init)

b = np.zeros((ny, nx))
b[int(ny / 4), int(nx / 4)] = 100
b[int(3 * ny / 4), int(3 * nx / 4)] = -100

p = poisson2d(p_init, b, dx, dy, nt)
plot2d(x, y, p)
