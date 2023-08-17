""" 2D Convection """

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


nx = 101
ny = 101
nt = 80
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = 0.2
dt = sigma * dx

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u_init = np.ones((ny, nx))
u_init[int(0.5 / dy) : int(1 / dy + 1), int(0.5 / dx) : int(1 / dx + 1)] = 2
v_init = np.ones((ny, nx))
v_init[int(0.5 / dy) : int(1 / dy + 1), int(0.5 / dx) : int(1 / dx + 1)] = 2


def conv2d(u, v, nt, dx, dy, dt):
    for _ in range(nt + 1):
        un = u.copy()
        vn = v.copy()
        u[1:, 1:] = (
            un[1:, 1:]
            - un[1:, 1:] * dt / dx * (un[1:, 1:] - un[1:, :-1])
            - vn[1:, 1:] * dt / dy * (un[1:, 1:] - un[:-1, 1:])
        )
        v[1:, 1:] = (
            vn[1:, 1:]
            - un[1:, 1:] * dt / dx * (vn[1:, 1:] - vn[1:, :-1])
            - vn[1:, 1:] * dt / dy * (vn[1:, 1:] - vn[:-1, 1:])
        )
        u[0, :] = 1
        u[2, :] = 1
        u[:, 0] = 1
        u[:, 2] = 1

        v[0, :] = 1
        v[2, :] = 1
        v[:, 0] = 1
        v[:, 2] = 1

    return u, v


fig = plt.figure(figsize=(11, 7))

ax = fig.add_subplot(2, 2, 1, projection="3d")
X, Y = np.meshgrid(x, y)
surf_u_init = ax.plot_surface(X, Y, u_init[:], cmap=cm.viridis)
ax = fig.add_subplot(2, 2, 3, projection="3d")
X, Y = np.meshgrid(x, y)
surf_v_init = ax.plot_surface(X, Y, v_init[:], cmap=cm.viridis)


u, v = conv2d(u_init, v_init, nt, dx, dy, dt)
ax = fig.add_subplot(2, 2, 2, projection="3d")
surf_u = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
ax = fig.add_subplot(2, 2, 4, projection="3d")
surf_v = ax.plot_surface(X, Y, v[:], cmap=cm.viridis)

plt.show()
