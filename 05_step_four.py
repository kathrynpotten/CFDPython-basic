""" Burgers' Equation """

import numpy as np
from matplotlib import pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify

# set up analytical solution
x, nu, t = sp.symbols("x nu t")
phi = sp.exp(-((x - 4 * t) ** 2) / (4 * nu * (t + 1))) + sp.exp(
    -((x - 4 * t - 2 * sp.pi) ** 2) / (4 * nu * (t + 1))
)
phiprime = phi.diff(x)

u = -2 * nu * (phiprime / phi) + 4
ufunc = lambdify((t, x, nu), u)


nx = 101
nt = 100
dx = 2 * np.pi / (nx - 1)
nu = 0.07
dt = dx * nu

t = 0
x = np.linspace(0, 2 * np.pi, nx)
u = np.asarray([ufunc(t, x0, nu) for x0 in x])

plt.plot(x, u, marker="o")
plt.xlim([0, 2 * np.pi])
plt.ylim([0, 10])
plt.show()


def burgers(u, nt, dx, nu, dt):
    """function to solve burgers with nt timesteps, with given initial condition"""

    for _ in range(nt):
        un = u.copy()
        u[1:-1] = (
            un[1:-1]
            - un[1:-1] * dt / dx * (un[1:-1] - un[:-2])
            + nu * dt / dx**2 * (un[2:] - 2 * un[1:-1] + un[:-2])
        )
    u[0] = (
        un[0]
        - un[0] * dt / dx * (un[0] - un[-2])
        + nu * dt / dx**2 * (un[1] - 2 * un[0] + un[-2])
    )
    u[-1] = u[0]

    return u


# analytical solution
u_analytical = np.asarray([ufunc(nt * dt, xi, nu) for xi in x])

plt.plot(x, burgers(u, nt, dx, nu, dt), marker="o", label="Computational")
plt.plot(x, u_analytical, label="Analytical")
plt.xlim([0, 2 * np.pi])
plt.ylim([0, 10])
plt.show()
