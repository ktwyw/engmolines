"""Boundary conditions by ghost points. A method-of-lines right-hand side extends the state with ghost values
chosen so that the boundary condition holds to second order, then applies the same interior stencil
everywhere. Dirichlet, Neumann, Robin and periodic conditions on node-centred and cell-centred grids, with one
or two ghost points, time-dependent values, and the clamped/pinned/free end conditions of beams.

Conventions: Neumann(g) means du/dx = g at the boundary (the x-derivative, not the outward normal);
Robin(alpha, beta, g) means alpha u + beta du/dx = g. Values may be numbers or functions of t.

>>> import numpy as np
>>> from engmolines import boundary as bc
>>> bc.pad(np.array([1.0, 2.0, 3.0]), 0.5, bc.Dirichlet(0.0), bc.Neumann(0.0), kind="node")
array([-2.,  1.,  2.,  3.,  2.])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _val(v, t):
    return float(v(t)) if callable(v) else float(v)


@dataclass
class Dirichlet:
    """u = value at the boundary (value may be a number or a function of t)."""

    value: float | object = 0.0


@dataclass
class Neumann:
    """du/dx = value at the boundary (the x-derivative, not the outward normal)."""

    value: float | object = 0.0


@dataclass
class Robin:
    """alpha u + beta du/dx = value at the boundary."""

    alpha: float
    beta: float
    value: float | object = 0.0


class Periodic:
    """Periodic boundary: the state wraps around."""

    pass


def ghosts(u, dx: float, cond, side: str, kind: str, t: float, nghost: int) -> np.ndarray:
    """Ghost values beyond one end (ordered from the boundary outward) for a single condition."""
    u = np.asarray(u, float)
    n = len(u)
    inner = u if side == "left" else u[::-1]  # inner[0] is the point nearest this boundary
    s = 1.0 if side == "left" else -1.0  # outward direction sign for derivatives
    out = np.empty(nghost)
    if isinstance(cond, Dirichlet):
        g = _val(cond.value, t)
        for k in range(1, nghost + 1):
            out[k - 1] = (
                2 * g - inner[k] if kind == "node" else 2 * g - inner[k - 1]
            )  # odd reflection about the boundary
    elif isinstance(cond, (Neumann, Robin)):
        if isinstance(cond, Neumann):
            slope = _val(cond.value, t)  # du/dx at the boundary
            if kind == "node":
                for k in range(1, nghost + 1):
                    out[k - 1] = inner[k] - 2 * k * dx * s * slope  # even reflection plus the slope
            else:
                for k in range(1, nghost + 1):
                    out[k - 1] = inner[k - 1] - (2 * k - 1) * dx * s * slope
        else:
            a, b, g = cond.alpha, cond.beta, _val(cond.value, t)
            if kind == "node":  # boundary value known: convert to a slope
                slope = (g - a * inner[0]) / b
                for k in range(1, nghost + 1):
                    out[k - 1] = inner[k] - 2 * k * dx * s * slope
            else:  # boundary value (u0 + ghost)/2, slope s (u0 - ghost)/dx
                gh = (g - inner[0] * (a / 2 + b * s / dx)) / (a / 2 - b * s / dx)
                out[0] = gh
                if nghost > 1:  # further ghosts by the equivalent Neumann reflection
                    slope = s * (inner[0] - gh) / dx
                    for k in range(2, nghost + 1):
                        out[k - 1] = inner[k - 1] - (2 * k - 1) * dx * s * slope
    else:
        raise TypeError(f"unknown boundary condition {cond!r}")
    return out if n else out


def pad(u, dx: float, left, right, kind: str = "node", t: float = 0.0, nghost: int = 1) -> np.ndarray:
    """The state extended by nghost ghost values at each end (periodic: wrapped)."""
    u = np.asarray(u, float)
    if isinstance(left, Periodic) or isinstance(right, Periodic):
        return np.concatenate([u[-nghost:], u, u[:nghost]])
    gl = ghosts(u, dx, left, "left", kind, t, nghost)[::-1]
    gr = ghosts(u, dx, right, "right", kind, t, nghost)
    return np.concatenate([gl, u, gr])


def pad2d(U, dx: float, dy: float, bcs: dict, t: float = 0.0, nghost: int = 1) -> np.ndarray:
    """Ghost cells on all four sides of a 2-D field (keys 'left', 'right', 'bottom', 'top'; node grids)."""
    U = np.asarray(U, float)
    cols = [pad(U[:, j], dx, bcs["left"], bcs["right"], "node", t, nghost) for j in range(U.shape[1])]
    Ux = np.array(cols).T  # padded along x
    rows = [pad(Ux[i, :], dy, bcs["bottom"], bcs["top"], "node", t, nghost) for i in range(Ux.shape[0])]
    return np.array(rows)


def dirichlet_rhs(rhs, left, right, t: float = 0.0, kind: str = "node") -> np.ndarray:
    """On node grids a Dirichlet boundary node is not a dynamic unknown: its rate is the rate of the imposed
    value (zero for a constant). Sets those rows of the right-hand side."""
    rhs = np.asarray(rhs, float).copy()
    if kind != "node":
        return rhs
    for cond, i in ((left, 0), (right, -1)):
        if isinstance(cond, Dirichlet):
            if callable(cond.value):
                e = 1e-6
                rhs[i] = (cond.value(t + e) - cond.value(t - e)) / (2 * e)
            else:
                rhs[i] = 0.0
    return rhs


def beam_ghosts(w, dx: float, end: str, side: str) -> np.ndarray:
    """Two ghost values for an Euler-Bernoulli beam end: 'clamped' (w = w' = 0), 'pinned' (w = w'' = 0) or 'free'
    (w'' = w''' = 0), for the 5-point fourth-derivative stencil on a node grid."""
    w = np.asarray(w, float)
    inner = w if side == "left" else w[::-1]
    if end == "clamped":
        return np.array([inner[1], inner[2]])  # even reflection: w(-x) = w(x) keeps w' = 0 (with w0 = 0)
    if end == "pinned":
        return np.array([-inner[1], -inner[2]])  # odd reflection: w'' = 0 at the end
    if end == "free":  # w''(0) = 0 and w'''(0) = 0 with centred stencils
        g1 = 2 * inner[0] - inner[1]
        g2 = 4 * inner[0] - 4 * inner[1] + inner[2]  # from w''' = (w2 - 2 w1 + 2 w_-1 - w_-2)/(2 dx^3) = 0
        return np.array([g1, g2])
    raise ValueError(end)


def pad_beam(w, dx: float, left: str, right: str) -> np.ndarray:
    """A beam deflection extended by two ghost values at each end ('clamped', 'pinned' or 'free')."""
    return np.concatenate(
        [beam_ghosts(w, dx, left, "left")[::-1], np.asarray(w, float), beam_ghosts(w, dx, right, "right")]
    )
