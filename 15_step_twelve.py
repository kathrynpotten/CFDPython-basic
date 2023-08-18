""" Channel Flow """
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from equations_methods import build_b, u_momentum_equation, v_momentum_equation


def pressure_poisson_periodic(p, b, dx, dy):
    for _ in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (
            dy**2 * (pn[1:-1, 2:] + pn[1:-1, :-2])
            + dx**2 * (pn[2:, 1:-1] + pn[:-2, 1:-1])
            - b[1:-1, 1:-1] * dx**2 * dy**2
        ) / (2 * (dx**2 + dy**2))

        p[1:-1, -1] = (
            dy**2 * (pn[1:-1, 0] + pn[1:-1, -2])
            + dx**2 * (pn[2:, -1] + pn[:-2, -1])
            - b[1:-1, -1] * dx**2 * dy**2
        ) / (2 * (dx**2 + dy**2))

        p[1:-1, 0] = (
            dy**2 * (pn[1:-1, 1] + pn[1:-1, -1])
            + dx**2 * (pn[2:, 0] + pn[:-2, 0])
            - b[1:-1, 0] * dx**2 * dy**2
        ) / (2 * (dx**2 + dy**2))

        p[-1, :] = p[-2, :]
        p[0, :] = p[1, :]

    return p


def periodic_bc_x_umom(i, un, vn, p, dx, dy, dt, rho, nu):
    return (
        un[1:-1, i]
        - dt / dx * un[1:-1, i] * (un[1:-1, i] - un[1:-1, i - 1])
        - dt / dy * vn[1:-1, i] * (un[1:-1, i] - un[:-2, i])
        - dt / (rho * 2 * dx) * (p[1:-1, i + 1] - p[1:-1, i - 1])
        + nu * dt / dx**2 * (un[1:-1, i + 1] - 2 * un[1:-1, i] + un[1:-1, i - 1])
        + nu * dt / dy**2 * (un[2:, i] - 2 * un[1:-1, i] + un[:-2, i])
    )


def periodic_bc_x_vmom(i, un, vn, p, dx, dy, dt, rho, nu):
    return (
        vn[1:-1, i]
        - dt / dx * un[1:-1, i] * (vn[1:-1, i] - vn[1:-1, i - 1])
        - dt / dy * vn[1:-1, i] * (vn[1:-1, i] - vn[:-2, i])
        - dt / (rho * 2 * dy) * (p[2:, i] - p[:-2, i])
        + nu * dt / dx**2 * (vn[1:-1, i + 1] - 2 * vn[1:-1, i] + vn[1:-1, i - 1])
        + nu * dt / dy**2 * (vn[2:, i] - 2 * vn[1:-1, i] + vn[:-2, i])
    )


def channel_flow(u, v, p, b, F, dx, dy, dt, rho, nu):
    udiff = 1
    iter_count = 0
    while udiff > 0.001:
        b = build_b(b, u, v, rho, dt, dx, dy)
        p = pressure_poisson_periodic(p, b, dx, dy)

        un = u.copy()
        vn = v.copy()
        u[1:-1, 1:-1] = u_momentum_equation(un, vn, p, dx, dy, dt, rho, nu) + dt * F
        v[1:-1, 1:-1] = v_momentum_equation(un, vn, p, dx, dy, dt, rho, nu)

        u[1:-1, 0] = periodic_bc_x_umom(0, un, vn, p, dx, dy, dt, rho, nu) + dt * F
        u[1:-1, -1] = periodic_bc_x_umom(-1, un, vn, p, dx, dy, dt, rho, nu) + dt * F

        v[1:-1, 0] = periodic_bc_x_vmom(0, un, vn, p, dx, dy, dt, rho, nu)
        v[1:-1, -1] = periodic_bc_x_vmom(-1, un, vn, p, dx, dy, dt, rho, nu)

        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0

        udiff = (np.sum(u) - np.sum(un)) / np.sum(u)
        iter_count += 1

    print(udiff, iter_count)
    return u, v, p


nx = 41
ny = 41
nit = 50
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

rho = 1
nu = 0.1
dt = 0.01
F = 1

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

u_init = np.zeros((ny, nx))
v_init = np.zeros((ny, nx))
p_init = np.zeros((ny, nx))
b = np.zeros((ny, nx))


u, v, p = channel_flow(u_init, v_init, p_init, b, F, dx, dy, dt, rho, nu)

fig = plt.figure()

plt.quiver(X[::3, ::3], Y[::3, ::3], u[::3, ::3], v[::3, ::3])
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
