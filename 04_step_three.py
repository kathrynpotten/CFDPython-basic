""" 1-D diffusion equation """

import numpy as np
from matplotlib import pyplot as plt


def diffusion(nx, nt):
    """function to solve 1D diffusion with nx grid points and nt timesteps,
    initial condition hat function"""
    dx = 2 / (nx - 1)
    nu = 0.3
    sigma = 0.2
    dt = sigma * dx**2 / nu

    u = np.ones(nx)
    u[int(0.5 / dx) : int(1 / dx + 1)] = 2

    for _ in range(nt):
        un = u.copy()
        u[1:-1] = un[1:-1] + nu * dt / dx**2 * (un[2:] - 2 * un[1:-1] + un[:-2])
        # for i in range(1, nx - 1):
        #    u[i] = un[i] + nu * dt / dx**2 * (un[i + 1] - 2 * un[i] + un[i - 1])

    return u


plt.plot(np.linspace(0, 2, 41), diffusion(41, 20))
plt.show()
