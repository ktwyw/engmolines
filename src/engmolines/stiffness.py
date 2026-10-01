"""Stiffness analysis of semi-discrete systems: the eigenvalue spectrum of the Jacobian, the stiffness ratio,
stability functions of the explicit integrators and the largest stable step for a given spectrum, the
theoretical extreme eigenvalues of the standard operators, and a cost model comparing explicit and implicit
integration.

>>> import numpy as np
>>> from engmolines import stiffness
>>> round(stiffness.explicit_step_limit(np.array([-100.0, -1.0]), "rk4"), 5)         # 2.785 / 100
0.02785
"""

from __future__ import annotations

import numpy as np

from . import linalg as la


def jacobian(f, y, t: float = 0.0, band=None) -> np.ndarray:
    """Dense Jacobian of f(t, y) at y (through the banded coloured differences when band is given)."""
    g = lambda u: f(t, u)  # noqa: E731
    if band is not None:
        return la.from_banded(la.banded_jacobian(g, y, *band), *band)
    return la.dense_jacobian(g, y)


def spectrum(f, y, t: float = 0.0, band=None) -> np.ndarray:
    """Eigenvalues of the Jacobian, sorted by real part (most negative first)."""
    lam = np.linalg.eigvals(jacobian(f, y, t, band))
    return lam[np.argsort(lam.real)]


def stiffness_ratio(lam) -> float:
    """max |Re lambda| / min |Re lambda| over the eigenvalues with non-negligible real part."""
    re = np.abs(np.asarray(lam).real)
    re = re[re > 1e-12 * re.max()]
    return float(re.max() / re.min())


def stability_function(method: str, z):
    """R(z) for the explicit methods: 'euler', 'heun', 'rk3', 'rk4', 'dopri' (the 5th-order Dormand-Prince
    solution), and the implicit 'backward_euler', 'trapezoidal', 'bdf2' (its principal root)."""
    z = np.asarray(z, complex)
    if method == "euler":
        return 1 + z
    if method == "heun":
        return 1 + z + z**2 / 2
    if method == "rk3":
        return 1 + z + z**2 / 2 + z**3 / 6
    if method == "rk4":
        return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    if method == "dopri":
        return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24 + z**5 / 120 + z**6 / 600
    if method == "backward_euler":
        return 1 / (1 - z)
    if method == "trapezoidal":
        return (1 + z / 2) / (1 - z / 2)
    if method == "bdf2":  # principal root of (3/2 - z) r^2 - 2 r + 1/2 = 0
        a = 1.5 - z
        return (2 + np.sqrt(4 - 2 * a)) / (2 * a)
    raise ValueError(method)


def explicit_step_limit(lam, method: str = "rk4") -> float:
    """Largest h such that |R(h lambda)| <= 1 for every eigenvalue with negative real part (bisection on h).
    Eigenvalues with Re lambda >= 0 describe modes that grow or persist physically (neutral translation modes
    of fronts, unstable states); they are excluded, because no step could make them decay."""
    lam = np.asarray(lam, complex)
    lam = lam[lam.real < 0]
    if len(lam) == 0:
        return np.inf

    def stable(h):
        return bool(np.all(np.abs(stability_function(method, h * lam)) <= 1 + 1e-12))

    hi = 1.0 / np.max(np.abs(lam))
    if not stable(hi):
        lo, hi = 0.0, hi
    else:
        lo = hi
        while stable(hi * 2) and hi < 1e12:
            hi *= 2
        lo, hi = hi, 2 * hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


def operator_extremes(kind: str, dx: float, coefficient: float = 1.0) -> float:
    """Theoretical extreme eigenvalue of a standard second-order operator on a uniform grid: 'diffusion' gives
    -4 kappa/dx^2, 'biharmonic' -16/dx^4, 'advection_centred' the imaginary extent +-i a/dx, 'advection_upwind'
    the real extent -2 a/dx (its eigenvalues lie on a circle of radius a/dx centred at -a/dx)."""
    if kind == "diffusion":
        return -4 * coefficient / dx**2
    if kind == "biharmonic":
        return -16 * coefficient / dx**4
    if kind == "advection_centred":
        return coefficient / dx
    if kind == "advection_upwind":
        return -2 * coefficient / dx
    raise ValueError(kind)


def cost_model(
    n: int,
    lam,
    t_end: float,
    bandwidth: int,
    explicit: str = "rk4",
    implicit_h: float | None = None,
    stages_explicit: int = 4,
    newton_per_step: float = 2.0,
) -> dict:
    """Rough flop counts: explicit at its stability limit (stages * n * stencil work per step) versus implicit at
    the step the accuracy needs (Newton iterations, each a banded solve of n (2b+1) flops, plus one coloured
    Jacobian and banded LU of n b^2 flops, amortised over the run)."""
    h_exp = explicit_step_limit(lam, explicit)
    steps_exp = int(np.ceil(t_end / h_exp))
    work_exp = steps_exp * stages_explicit * n * (2 * bandwidth + 1)
    h_imp = implicit_h if implicit_h is not None else t_end / 100
    steps_imp = int(np.ceil(t_end / h_imp))
    work_imp = (
        steps_imp * newton_per_step * n * (2 * bandwidth + 1) * 2
        + n * bandwidth**2
        + (2 * bandwidth + 1) * n * (2 * bandwidth + 1)
    )
    return {
        "explicit_step_limit": h_exp,
        "explicit_steps": steps_exp,
        "explicit_flops": work_exp,
        "implicit_steps": steps_imp,
        "implicit_flops": work_imp,
        "ratio": work_exp / work_imp,
    }
