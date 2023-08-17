""" 2D Linear Convection """

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


nx = 81
ny = 81
nt = 100
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = 0.2
dt = sigma * dx

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u_init = np.ones((ny, nx))
u_init[int(0.5 / dy) : int(1 / dy + 1), int(0.5 / dx) : int(1 / dx + 1)] = 2


def convection2d(u, nt, c, dx, dy, dt):
    for _ in range(nt + 1):
        un = u.copy()
        u[1:, 1:] = (
            un[1:, 1:]
            - c * dt / dx * (un[1:, 1:] - un[1:, :-1])
            - c * dt / dy * (un[1:, 1:] - un[:-1, 1:])
        )
        u[0, :] = 1
        u[2, :] = 1
        u[:, 0] = 1
        u[:, 2] = 1

    return u


fig = plt.figure(figsize=(11, 7))

ax = fig.add_subplot(1, 2, 1, projection="3d")
X, Y = np.meshgrid(x, y)
surf_init = ax.plot_surface(X, Y, u_init[:], cmap=cm.viridis)

u = convection2d(u_init, nt, c, dx, dy, dt)
ax = fig.add_subplot(1, 2, 2, projection="3d")
surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)

plt.show()
