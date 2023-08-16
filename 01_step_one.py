""" 1-D linear convection equation """

import numpy as np
from matplotlib import pyplot

nx = 41
dx = 2 / (nx - 1)
nt = 25
dt = 0.025
c = 1


# initial velocity
u = np.ones(nx)
u[int(0.5 / dx) : int(1 / dx + 1)] = 2

pyplot.plot(np.linspace(0, 2, nx), u)
pyplot.show()


# run convection for specified time steps
for n in range(nt):
    un = u.copy()
    u[1:] = un[1:] - c * dt / dx * (un[1:] - un[:-1])


pyplot.plot(np.linspace(0, 2, nx), u)
pyplot.show()
