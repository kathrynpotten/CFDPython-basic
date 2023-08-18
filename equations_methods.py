""" Equations Methods """

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
