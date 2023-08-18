""" Cavity Flow """

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


def build_b(b, u, v, rho, dt, dx, dy):
    b[1:-1, 1:-1] = rho * (
        1
        / dt
        * (
            (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
            + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
        )
        - (
            ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx))
            * (u[1:-1, 2:] - u[1:-1, :-2])
            / (2 * dx)
        )
        - 2
        * (
            ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy))
            * (v[1:-1, 2:] - v[1:-1, :-2])
            / (2 * dx)
        )
        - (
            ((v[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy))
            * (v[2:, 1:-1] - v[:-2, 1:-1])
            / (2 * dx)
        )
    )
    return b


def pressure_poisson(p, b, dx, dy):
    for _ in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (
            dy**2 * (pn[1:-1, 2:] + pn[1:-1, :-2])
            + dx**2 * (pn[2:, 1:-1] + pn[:-2, 1:-1])
            - b[1:-1, 1:-1] * dx**2 * dy**2
        ) / (2 * (dx**2 + dy**2))

        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[-1, :] = 0

    return p


def u_momentum_equation(un, vn, p, dx, dy, dt, rho, nu):
    return (
        un[1:-1, 1:-1]
        - dt / dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2])
        - dt / dy * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[:-2, 1:-1])
        - dt / (rho * 2 * dx) * (p[1:-1, 2:] - p[1:-1, :-2])
        + nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])
        + nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])
    )


def v_momentum_equation(un, vn, p, dx, dy, dt, rho, nu):
    return (
        vn[1:-1, 1:-1]
        - dt / dx * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, :-2])
        - dt / dy * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[:-2, 1:-1])
        - dt / (rho * 2 * dy) * (p[2:, 1:-1] - p[:-2, 1:-1])
        + nu * dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2])
        + nu * dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])
    )


def cavity_flow(u, v, p, b, dx, dy, dt, nt, rho, nu):
    for _ in range(nt + 1):
        b = build_b(b, u, v, rho, dt, dx, dy)
        p = pressure_poisson(p, b, dx, dy)

        un = u.copy()
        vn = v.copy()
        u[1:-1, 1:-1] = u_momentum_equation(u, un, vn, p, dx, dy, dt, rho, nu)
        v[1:-1, 1:-1] = v_momentum_equation(u, un, vn, p, dx, dy, dt, rho, nu)

        u[-1, :] = 1
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0

        v[-1, :] = 0
        v[0, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0

    return u, v, p


nx = 41
ny = 41
nt = 700
nit = 50
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

rho = 1
nu = 0.1
dt = 0.001

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

u_init = np.zeros((ny, nx))
v_init = np.zeros((ny, nx))
p_init = np.zeros((ny, nx))
b = np.zeros((ny, nx))


u, v, p = cavity_flow(u_init, v_init, p_init, b, dx, dy, dt, nt, rho, nu)

fig = plt.figure()
plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
plt.colorbar()
plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
plt.xlabel("X")
plt.ylabel("Y")

plt.show()

fig = plt.figure()
plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
plt.colorbar()
plt.streamplot(X, Y, u, v)
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
