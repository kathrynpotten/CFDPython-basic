""" CFL Condition """

import numpy as np
from matplotlib import pyplot as plt


def linearconv(nx, nt, dt):
    """function to solve linear convection with nx grid points and nt timesteps,
    for given delta t"""
    dx = 2 / (nx - 1)
    c = 1

    u = np.ones(nx)
    u[int(0.5 / dx) : int(1 / dx + 1)] = 2

    for _ in range(nt):
        un = u.copy()
        u[1:] = un[1:] - c * dt / dx * (un[1:] - un[:-1])

    return u


figure, axis = plt.subplots(3, 2)

axis[0, 0].plot(np.linspace(0, 2, 41), linearconv(41, 25, 0.025))
axis[0, 1].plot(np.linspace(0, 2, 51), linearconv(51, 25, 0.025))
axis[1, 0].plot(np.linspace(0, 2, 61), linearconv(61, 25, 0.025))
axis[1, 1].plot(np.linspace(0, 2, 71), linearconv(71, 25, 0.025))
axis[2, 0].plot(np.linspace(0, 2, 81), linearconv(81, 25, 0.025))
axis[2, 1].plot(np.linspace(0, 2, 91), linearconv(91, 25, 0.025))

plt.show()


def linearconv_courant(nx, nt):
    """function to solve linear convection with nx grid points and nt timesteps,
    with appropriate delta t calculated from number of grid points using CFL number to ensure stability.
    Initial condition hat function
    """
    dx = 2 / (nx - 1)
    c = 1
    sigma = 0.5

    dt = sigma * dx

    u = np.ones(nx)
    u[int(0.5 / dx) : int(1 / dx + 1)] = 2

    for _ in range(nt):
        un = u.copy()
        u[1:] = un[1:] - c * dt / dx * (un[1:] - un[:-1])

    return u


figure, axis = plt.subplots(3, 2)

axis[0, 0].plot(np.linspace(0, 2, 41), linearconv_courant(41, 25))
axis[0, 1].plot(np.linspace(0, 2, 51), linearconv_courant(51, 25))
axis[1, 0].plot(np.linspace(0, 2, 61), linearconv_courant(61, 25))
axis[1, 1].plot(np.linspace(0, 2, 71), linearconv_courant(71, 25))
axis[2, 0].plot(np.linspace(0, 2, 81), linearconv_courant(81, 25))
axis[2, 1].plot(np.linspace(0, 2, 91), linearconv_courant(91, 25))

plt.show()
