"""Semi-discrete spatial operators, each a function u -> du/dt contribution on a grid with boundary conditions:
diffusion in conservative form with variable coefficients (harmonic face averaging) in planar, cylindrical and
spherical geometry; advection by centred, upwind and flux-limited schemes; third-derivative dispersion;
the fourth-derivative beam operator; and the 2-D Laplacian. All second-order accurate in space unless stated.

>>> import numpy as np
>>> from engmolines import grids, boundary as bc, operators as op
>>> g = grids.uniform(0, 1, 4)
>>> u = np.array([0.0, 1.0, 4.0, 9.0, 16.0])                          # u = (4x)^2, so u'' = 32
>>> op.diffusion(u, g, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(16.0))
array([ 0., 32., 32., 32.,  0.])
"""

from __future__ import annotations

import numpy as np

from . import boundary as bc

LIMITERS = {
    "minmod": lambda r: np.maximum(0, np.minimum(1, r)),
    "superbee": lambda r: np.maximum.reduce([np.zeros_like(r), np.minimum(2 * r, 1), np.minimum(r, 2)]),
    "van-leer": lambda r: (r + np.abs(r)) / (1 + np.abs(r)),
    "mc": lambda r: np.maximum(0, np.minimum.reduce([2 * r, 0.5 * (1 + r), 2 * np.ones_like(r)])),
    "koren": lambda r: np.maximum(0, np.minimum.reduce([2 * r, (1 + 2 * r) / 3, 2 * np.ones_like(r)])),
}


def _harmonic(a, b):
    """Harmonic mean, zero where both values vanish (degenerate diffusion)."""
    den = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, 2 * a * b / np.where(den > 0, den, 1.0), 0.0)


def _face_coefficient(k, g, u=None):
    """Diffusivity at the faces between neighbouring points: harmonic mean of point values (exact flux continuity
    for piecewise-constant layers); k may be a number, an array at the points, or a function k(x) or k(x, u)."""
    if callable(k):
        try:
            kp = np.asarray(k(g.x, u), float)
        except TypeError:
            kp = np.asarray(k(g.x), float)
    else:
        kp = np.broadcast_to(np.asarray(k, float), g.x.shape)
    return kp, _harmonic(kp[:-1], kp[1:])


def diffusion(u, g, k, left, right, t: float = 0.0, source=None) -> np.ndarray:
    """d/dx (k du/dx) [+ source(x, u, t)] in conservative form; for radial grids (kind 'cylindrical' or
    'spherical') (1/r^p) d/dr (r^p k du/dr) with the symmetry condition at r = 0 built in (left ignored there)."""
    u = np.asarray(u, float)
    dx = g.dx
    radial = g.kind in ("cylindrical", "spherical")
    kind = "node" if radial else g.kind
    kp, kf = _face_coefficient(k, g, u)
    up = bc.pad(u, dx, bc.Neumann(0.0) if radial else left, right, kind, t)
    kpad = np.r_[kp[0], kp, kp[-1]]
    kfp = _harmonic(kpad[:-1], kpad[1:])  # faces including the two beyond the ends
    flux = kfp * np.diff(up) / dx  # k du/dx at every face
    if radial:
        p = 1 if g.kind == "cylindrical" else 2
        r = g.x
        rf_left = np.where(
            np.arange(len(r)) == 0, 0.0, 0.5 * (np.r_[r[0] - dx, r[:-1]] + r)
        )  # inner face of each control volume (none at r = 0)
        rf_right = 0.5 * (r + np.r_[r[1:], r[-1] + dx])  # outer face (a half-cell beyond the last node)
        vol = (rf_right ** (p + 1) - rf_left ** (p + 1)) / (p + 1)
        out = (rf_right**p * flux[1:] - rf_left**p * flux[:-1]) / vol
    else:
        out = np.diff(flux) / dx
    if source is not None:
        out = out + source(g.x, u, t)
    if kind == "node":
        out = bc.dirichlet_rhs(out, bc.Neumann(0.0) if radial else left, right, t)
    return out


def advection(u, g, a, left, right, t: float = 0.0, scheme: str = "upwind") -> np.ndarray:
    """-(a u)_x for a velocity a (number or array at the points): 'centred' (second order, non-dissipative:
    unstable with explicit non-dissipative time integrators), 'upwind' (first order, monotone), or a
    flux-limited scheme ('minmod', 'superbee', 'van-leer', 'mc', 'koren') in flux form on the faces."""
    u = np.asarray(u, float)
    dx = g.dx
    a = np.broadcast_to(np.asarray(a, float), u.shape)
    if scheme == "centred":
        up = bc.pad(u, dx, left, right, g.kind, t)
        out = -(a * (up[2:] - up[:-2]) / (2 * dx))
    else:
        up = bc.pad(u, dx, left, right, g.kind, t, nghost=2)
        af = np.r_[a[0], 0.5 * (a[:-1] + a[1:]), a[-1]]  # velocity at the faces (n + 1 faces)
        uL = up[1:-2]  # left state at each face i - 1/2 ... n + 1/2  (n + 1 faces)
        uR = up[2:-1]
        if scheme != "upwind":
            phi = LIMITERS[scheme]
            dL = up[1:-2] - up[:-3]  # slope ratios for the two upwind directions
            dC = up[2:-1] - up[1:-2]
            dR = up[3:] - up[2:-1]
            with np.errstate(divide="ignore", invalid="ignore"):
                rp = np.where(dC != 0, dL / np.where(dC != 0, dC, 1), 0.0)  # flow to the right
                rm = np.where(dC != 0, dR / np.where(dC != 0, dC, 1), 0.0)  # flow to the left
            uL = uL + 0.5 * phi(rp) * dC
            uR = uR - 0.5 * phi(rm) * dC
        flux = np.where(af >= 0, af * uL, af * uR)
        out = -np.diff(flux) / dx
    if g.kind == "node":
        out = bc.dirichlet_rhs(out, left, right, t)
    return out


def dispersion(u, g, left, right, t: float = 0.0) -> np.ndarray:
    """-u_xxx by the centred 4-point stencil (second order)."""
    u = np.asarray(u, float)
    dx = g.dx
    up = bc.pad(u, dx, left, right, g.kind, t, nghost=2)
    return -(up[4:] - 2 * up[3:-1] + 2 * up[1:-3] - up[:-4]) / (2 * dx**3)


def biharmonic(w, g, left: str, right: str) -> np.ndarray:
    """w_xxxx by the 5-point stencil with beam end conditions 'clamped', 'pinned' or 'free' (node grids)."""
    w = np.asarray(w, float)
    dx = g.dx
    wp = bc.pad_beam(w, dx, left, right)
    out = (wp[4:] - 4 * wp[3:-1] + 6 * wp[2:-2] - 4 * wp[1:-3] + wp[:-4]) / dx**4
    if left in ("clamped", "pinned"):
        out[0] = 0.0
    if right in ("clamped", "pinned"):
        out[-1] = 0.0
    return out


def laplacian2d(U, g2, bcs: dict, t: float = 0.0, kappa: float = 1.0) -> np.ndarray:
    """kappa (u_xx + u_yy) on a tensor grid with ghost cells on all four sides; Dirichlet boundary nodes get a
    zero rate."""
    U = np.asarray(U, float)
    dx, dy = g2.gx.dx, g2.gy.dx
    P = bc.pad2d(U, dx, dy, bcs, t)
    out = kappa * (
        (P[2:, 1:-1] - 2 * P[1:-1, 1:-1] + P[:-2, 1:-1]) / dx**2
        + (P[1:-1, 2:] - 2 * P[1:-1, 1:-1] + P[1:-1, :-2]) / dy**2
    )
    for side, sl in (
        ("left", (0, slice(None))),
        ("right", (-1, slice(None))),
        ("bottom", (slice(None), 0)),
        ("top", (slice(None), -1)),
    ):
        if isinstance(bcs[side], bc.Dirichlet):
            out[sl] = 0.0
    return out


# ------------------------------------------------------------------ nonlinear conservation laws (finite volumes)
def conservation_law(U, g, flux, wavespeed, left, right, t: float = 0.0, limiter: str = "van-leer") -> np.ndarray:
    """-(F(U))_x on a cell-centred grid for a scalar law (U of shape (n,)) or a system (U of shape (m, n)), by the
    MUSCL-Rusanov finite-volume scheme: limited linear reconstruction of each component to the faces, then the
    local Lax-Friedrichs (Rusanov) flux F* = (F(L) + F(R))/2 - s (R - L)/2 with s the largest local wave speed.
    flux(U) -> F, wavespeed(U) -> max |eigenvalue| (per cell). Boundary conditions: 'periodic' or 'transmissive'
    (zero-gradient outflow), applied to every component. Second order in smooth regions, TVD at shocks."""
    U = np.asarray(U, float)
    scalar = U.ndim == 1
    W = U[None, :] if scalar else U
    dx = g.dx
    if left == "periodic":
        P = np.concatenate([W[:, -2:], W, W[:, :2]], axis=1)
    else:
        P = np.concatenate([W[:, :1], W[:, :1], W, W[:, -1:], W[:, -1:]], axis=1)
    phi = LIMITERS[limiter]
    dm = P[:, 1:-1] - P[:, :-2]
    dp = P[:, 2:] - P[:, 1:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dp != 0, dm / np.where(dp != 0, dp, 1.0), 0.0)
    slope = phi(r) * dp  # limited slope in each cell of P[:, 1:-1] (n + 2 cells)
    C = P[:, 1:-1]
    UL = (C + 0.5 * slope)[:, :-1]  # left state at each of the n + 1 faces
    UR = (C - 0.5 * slope)[:, 1:]
    FL, FR = flux(UL), flux(UR)
    s = np.maximum(wavespeed(UL), wavespeed(UR))
    F = 0.5 * (FL + FR) - 0.5 * s * (UR - UL)
    out = -(F[:, 1:] - F[:, :-1]) / dx
    return out[0] if scalar else out
