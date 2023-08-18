""" 2D Laplace Equation """

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


def laplace2d(p, y, dx, dy, l1norm_target):
    l1norm = 1
    while l1norm > l1norm_target:
        pn = p.copy()
        p[1:-1, 1:-1] = (
            dy**2 * (pn[1:-1, 2:] + pn[1:-1, :-2])
            + dx**2 * (pn[2:, 1:-1] + pn[:-2, 1:-1])
        ) / (2 * (dx**2 + dy**2))
        p[:, 0] = 0
        p[:, -1] = y
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        l1norm = (np.sum(np.abs(p[:]) - np.abs(pn[:]))) / np.sum(np.abs(pn[:]))

    return p


nx = 31
ny = 31
dx = 2 / (nx - 1)
dy = 1 / (ny - 1)

x = np.linspace(0, 2, nx)
y = np.linspace(0, 1, ny)

p_init = np.zeros((ny, nx))

p_init[:, 0] = 0
p_init[:, -1] = y
p_init[0, :] = p_init[1, :]
p_init[-1, :] = p_init[-2, :]

plot2d(x, y, p_init)

p = laplace2d(p_init, y, dx, dy, 1e-4)
plot2d(x, y, p)
