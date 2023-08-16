""" Non-linear convection """

import numpy as np
from matplotlib import pyplot

nx = 41
dx = 2 / (nx - 1)
nt = 250
dt = 0.0025


# initial velocity
u = np.ones(nx)
u[int(0.5 / dx) : int(1 / dx + 1)] = 2

for n in range(nt):
    un = u.copy()
    u[1:] = un[1:] - un[1:] * dt / dx * (un[1:] - un[:-1])


pyplot.plot(np.linspace(0, 2, nx), u)
pyplot.show()
