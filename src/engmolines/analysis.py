"""Analysis tools: Fourier symbols of the spatial stencils and von Neumann amplification of the fully discrete
scheme (spatial symbol composed with the integrator's stability function), modified equations (the numerical
diffusion and dispersion a scheme adds), orders of convergence from error sequences with separated space and time
refinements, and conservation and energy bookkeeping.

>>> import numpy as np
>>> from engmolines import analysis
>>> round(float(analysis.symbol("central2", np.pi, 0.1).real))   # the highest mode of u_xx: -4/dx^2
-400
"""

from __future__ import annotations

import numpy as np

from .stiffness import stability_function


def symbol(stencil: str, theta, dx: float, a: float = 1.0) -> np.ndarray:
    """Fourier symbol sigma(theta) of a stencil applied to e^{i j theta}: the operator multiplies the mode by
    sigma. 'central2' (u_xx, 3 points), 'central4' (u_xx, 5 points), 'first_central' (a u_x, centred),
    'first_upwind' (a u_x, upwind for a > 0), 'biharmonic' (u_xxxx, 5 points), 'third' (u_xxx, 4 points)."""
    th = np.asarray(theta, float)
    if stencil == "central2":
        return (2 * np.cos(th) - 2) / dx**2 + 0j
    if stencil == "central4":
        return (-30 + 32 * np.cos(th) - 2 * np.cos(2 * th)) / (12 * dx**2) + 0j
    if stencil == "first_central":
        return 1j * a * np.sin(th) / dx
    if stencil == "first_upwind":
        return a * (1 - np.exp(-1j * th)) / dx
    if stencil == "biharmonic":
        return (2 * np.cos(2 * th) - 8 * np.cos(th) + 6) / dx**4 + 0j
    if stencil == "third":
        return 1j * (np.sin(2 * th) - 2 * np.sin(th)) / dx**3
    raise ValueError(stencil)


def amplification(
    stencil: str, method: str, h: float, dx: float, theta=None, coefficient: float = 1.0, sign: float = 1.0
) -> np.ndarray:
    """|R(h * coefficient * sigma(theta))| of the fully discrete scheme for the modes theta in [0, pi]; for
    advection pass sign = -1 (the operator is -a u_x)."""
    th = np.linspace(0, np.pi, 361) if theta is None else np.asarray(theta, float)
    return np.abs(stability_function(method, h * sign * coefficient * symbol(stencil, th, dx)))


def max_stable_step(stencil: str, method: str, dx: float, coefficient: float = 1.0, sign: float = 1.0) -> float:
    """Largest h with amplification <= 1 for every mode (bisection)."""
    lo, hi = 0.0, 1.0
    while np.max(amplification(stencil, method, hi, dx, coefficient=coefficient, sign=sign)) <= 1 + 1e-12 and hi < 1e9:
        hi *= 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.max(amplification(stencil, method, mid, dx, coefficient=coefficient, sign=sign)) <= 1 + 1e-12:
            lo = mid
        else:
            hi = mid
    return lo


def modified_equation(scheme: str, a: float, dx: float) -> dict:
    """Leading error terms of the semi-discrete advection schemes u_t + a u_x = 0 (time continuous, as in the
    method of lines): 'upwind' adds diffusion (a dx / 2) u_xx; 'centred' adds dispersion -(a dx^2 / 6) u_xxx;
    'upwind2' (second-order biased) adds dispersion (a dx^2 / 3) u_xxx and diffusion -(a dx^3/4) u_xxxx."""
    if scheme == "upwind":
        return {"diffusion": a * dx / 2, "dispersion": 0.0}
    if scheme == "centred":
        return {"diffusion": 0.0, "dispersion": -a * dx**2 / 6}
    if scheme == "upwind2":
        return {"diffusion": 0.0, "dispersion": a * dx**2 / 3, "fourth": -a * dx**3 / 4}
    raise ValueError(scheme)


def observed_order(errors, ratio: float = 2.0) -> np.ndarray:
    """log(e_k / e_{k+1}) / log(ratio) for a sequence of errors under successive refinement by `ratio`."""
    e = np.asarray(errors, float)
    return np.log(e[:-1] / e[1:]) / np.log(ratio)


def convergence_study(run, ns, hs, exact) -> dict:
    """Space-time convergence table: run(n, h) -> (x, u) at the final time, exact(x) the reference. Returns the
    error matrix (rows n, columns h) and the orders measured along the finest column (space) and finest row
    (time)."""
    E = np.zeros((len(ns), len(hs)))
    for i, n in enumerate(ns):
        for j, h in enumerate(hs):
            x, u = run(n, h)
            E[i, j] = np.max(np.abs(u - exact(x)))
    return {
        "errors": E,
        "space_orders": observed_order(E[:, -1], ns[1] / ns[0]),
        "time_orders": observed_order(E[-1, :], hs[0] / hs[1]),
    }


def conservation(grid, ys) -> np.ndarray:
    """Integral of each stored state over the domain: constant for a conservative scheme with no-flux ends."""
    return np.array([grid.integrate(y) for y in ys])


def wave_energy(u, v, c: float, grid) -> float:
    """(1/2) integral (v^2 + c^2 u_x^2) dx for the wave equation u_tt = c^2 u_xx (u_x by central differences)."""
    ux = np.gradient(np.asarray(u, float), grid.x)
    return 0.5 * grid.integrate(np.asarray(v) ** 2 + c**2 * ux**2)
