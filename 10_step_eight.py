""" 2D Burgers' Equation """

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


nx = 41
ny = 41
nt = 120
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
nu = 0.01
sigma = 0.0009
dt = sigma * dx * dy / nu

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u_init = np.ones((ny, nx))
u_init[int(0.5 / dy) : int(1 / dy + 1), int(0.5 / dx) : int(1 / dx + 1)] = 2
v_init = np.ones((ny, nx))
v_init[int(0.5 / dy) : int(1 / dy + 1), int(0.5 / dx) : int(1 / dx + 1)] = 2


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u_init[:], cmap=cm.viridis)
ax.plot_surface(X, Y, v_init[:], cmap=cm.viridis)

plt.show()


def burger2d(u, v, nt, dx, dy, dt, nu):
    for _ in range(nt + 1):
        un = u.copy()
        vn = v.copy()
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            - dt / dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2])
            - dt / dy * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[:-2, 1:-1])
            + nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])
            + nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])
        )

        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            - dt / dx * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, :-2])
            - dt / dy * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[:-2, 1:-1])
            + nu * dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2])
            + nu * dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])
        )

        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1

    return u, v


u, v = burger2d(u_init, v_init, nt, dx, dy, dt, nu)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
ax.plot_surface(X, Y, v[:], cmap=cm.viridis)

plt.show()
