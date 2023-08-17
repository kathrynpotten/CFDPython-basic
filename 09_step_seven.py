""" 2D diffusion """

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


nx = 31
ny = 31
nt = 17
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
nu = 0.05
sigma = 0.25
dt = sigma * dx * dy / nu

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u_init = np.ones((ny, nx))
u_init[int(0.5 / dy) : int(1 / dy + 1), int(0.5 / dx) : int(1 / dx + 1)] = 2


fig = plt.figure()
ax = fig.subplots(subplot_kw={"projection": "3d"})
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_zlim(1, 2.5)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u_init[:], cmap=cm.viridis)

plt.show()


def diff2d(u, nu, nt, dx, dy, dt):
    for _ in range(nt + 1):
        un = u.copy()
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            + nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])
            + nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])
        )

        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    return u


fig = plt.figure()

ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_zlim(1, 2.5)
X, Y = np.meshgrid(x, y)

u = diff2d(u_init, nu, 10, dx, dy, dt)
ax.plot_surface(X, Y, u[:], cmap=cm.viridis)

ax = fig.add_subplot(1, 3, 2, projection="3d")
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_zlim(1, 2.5)
X, Y = np.meshgrid(x, y)

u = diff2d(u_init, nu, 14, dx, dy, dt)
ax.plot_surface(X, Y, u[:], cmap=cm.viridis)


ax = fig.add_subplot(1, 3, 3, projection="3d")
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_zlim(1, 2.5)
X, Y = np.meshgrid(x, y)

u = diff2d(u_init, nu, 50, dx, dy, dt)
ax.plot_surface(X, Y, u[:], cmap=cm.viridis)

plt.show()
