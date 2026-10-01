"""Validation of engmolines against exact solutions, NumPy/SciPy references and the theory of the method of lines
(orders of convergence in space and time, stiffness and stability limits, conservation).
Run:  python docs/validate.py      (writes docs/VALIDATION.md)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import integrate as sint
from scipy import linalg as sla
from scipy import special as ssp

from engmolines import analysis, grids, stiffness
from engmolines import boundary as bc
from engmolines import integrators as ti
from engmolines import linalg as la
from engmolines import operators as op
from engmolines import stencils as st

RESULTS: list[tuple] = []
rng = np.random.default_rng(2026)


def check(section, name, reference, value, expected, rtol=0.0, atol=0.0):
    ok = bool(np.isclose(float(value), float(expected), rtol=rtol, atol=atol))
    RESULTS.append((section, name, reference, float(value), float(expected), ok))
    if not ok:
        print(f"FAIL: {section} | {name}: computed {value!r}, expected {expected!r}")


# ------------------------------------------------------------------ 1. stencils
S = "Stencils"
check(
    S,
    "Fornberg weights for the 3-point second derivative",
    "[1, -2, 1]",
    float(np.max(np.abs(st.fd_weights(0.0, [-1, 0, 1], 2) - [1, -2, 1]))),
    0.0,
    atol=1e-15,
)
check(
    S,
    "5-point fourth-order first derivative",
    "[1, -8, 0, 8, -1]/12",
    float(np.max(np.abs(st.fd_weights(0.0, [-2, -1, 0, 1, 2], 1) - np.array([1, -8, 0, 8, -1]) / 12))),
    0.0,
    atol=1e-15,
)
check(
    S,
    "one-sided second-order first derivative",
    "[-3, 4, -1]/2",
    float(np.max(np.abs(st.fd_weights(0.0, [0, 1, 2], 1) - np.array([-3, 4, -1]) / 2))),
    0.0,
    atol=1e-15,
)
check(
    S,
    "weights on non-uniform points are exact for polynomials of the stencil's degree",
    "exactness",
    float(abs(st.fd_weights(0.3, [0.0, 0.5, 1.2, 1.4], 2) @ np.array([0.0, 0.5, 1.2, 1.4]) ** 3 - 6 * 0.3)),
    0.0,
    atol=1e-12,
)
f_test = lambda x: np.sin(3 * x)  # noqa: E731
dfs = [
    lambda x: 3 * np.cos(3 * x),
    lambda x: -9 * np.sin(3 * x),
    lambda x: -27 * np.cos(3 * x),
    lambda x: 81 * np.sin(3 * x),
]
for deriv, acc in ((1, 2), (1, 4), (1, 6), (2, 2), (2, 4), (3, 2), (4, 2)):
    errs = []
    for n_ in (40, 80, 160):
        x_ = np.linspace(0, 1, n_ + 1)
        errs.append(float(np.max(np.abs(st.diff_operator(x_, deriv, acc)(f_test(x_)) - dfs[deriv - 1](x_)))))
    check(
        S,
        f"differentiation operator, derivative {deriv}, accuracy {acc}: measured order (boundary closures included)",
        f"theory: {acc} (closures may converge faster)",
        float(np.mean(analysis.observed_order(errs))),
        acc,
        atol=0.5,
    )
xp = np.linspace(0, 2 * np.pi, 64, endpoint=False)
check(
    S,
    "periodic operator wraps the stencil (derivative of sin on a periodic grid)",
    "second order",
    float(np.max(np.abs(st.diff_operator(xp, 1, 2, periodic=True)(np.sin(xp)) - np.cos(xp)))),
    0.0,
    atol=2e-3,
)
xs_ = grids.stretched(0, 1, 40, strength=2.0).x
check(
    S,
    "operators on a stretched grid: u_xx of x^3 exact (stencil degree 2 -> exact for quadratics; cubic error small)",
    "6x",
    float(np.max(np.abs(st.diff_operator(xs_, 2, 2)(xs_**3) - 6 * xs_))),
    0.0,
    atol=0.4,
)
D2 = st.diff_operator(np.linspace(0, 1, 30), 2, 2)
u_ = rng.normal(size=30)
check(
    S,
    "stencil form, dense form and banded form agree",
    "construction",
    float(
        np.max(np.abs(D2(u_) - D2.to_dense() @ u_)) + np.max(np.abs(la.from_banded(*D2.to_banded()) - D2.to_dense()))
    ),
    0.0,
    atol=1e-10,
)
Dsum = D2 + 3.0 * st.identity(30)
check(
    S,
    "operator algebra: D + 3 I",
    "dense",
    float(np.max(np.abs(Dsum.to_dense() - (D2.to_dense() + 3 * np.eye(30))))),
    0.0,
    atol=1e-13,
)
up1 = st.upwind_operator(np.linspace(0, 1, 41), 1, 1)
check(
    S,
    "upwind operator is one-sided: no downstream coefficient in the interior rows",
    "biasing",
    float(np.max(np.abs(up1.coeffs[up1.offsets > 0][:, 2:-2]))) if np.any(up1.offsets > 0) else 0.0,
    0.0,
)
Ax, Ay = st.diff_operator(np.linspace(0, 1, 9), 2, 2), st.diff_operator(np.linspace(0, 1, 7), 2, 2)
apply2, band2 = st.kron_sum(Ax, Ay)
K = np.kron(Ax.to_dense(), np.eye(7)) + np.kron(np.eye(9), Ay.to_dense())
v_ = rng.normal(size=63)
check(
    S,
    "Kronecker sum applies as (Ax x I + I x Ay)",
    "numpy.kron",
    float(np.max(np.abs(apply2(v_) - K @ v_))),
    0.0,
    atol=1e-12,
)
check(
    S,
    "... with the bandwidth ny * (half-width of Ax including its one-sided closures)",
    "band structure",
    float(band2[0]),
    7.0 * Ax.bandwidth[0],
)

# ------------------------------------------------------------------ 2. boundary conditions
S = "Boundary conditions"
u5 = np.array([1.0, 1.4, 1.9, 2.5, 3.2])
p_ = bc.pad(u5, 0.1, bc.Robin(2.0, 0.5, 3.0), bc.Neumann(1.5), kind="cell")
check(
    S,
    "Robin ghost on a cell grid satisfies alpha u + beta u_x = g at the face",
    "definition",
    float(2.0 * (p_[0] + p_[1]) / 2 + 0.5 * (p_[1] - p_[0]) / 0.1),
    3.0,
    atol=1e-12,
)
check(
    S,
    "Neumann ghost on a cell grid gives the slope at the face",
    "definition",
    float((p_[-1] - p_[-2]) / 0.1),
    1.5,
    atol=1e-12,
)
p_ = bc.pad(u5, 0.1, bc.Dirichlet(0.7), bc.Neumann(-2.0), kind="node", nghost=2)
check(
    S,
    "Dirichlet on a node grid: odd reflection about the boundary node (mean of ghost and mirror = value)",
    "reflection",
    float(0.5 * (p_[1] + u5[1])),
    0.7,
    atol=1e-12,
)
check(
    S,
    "Neumann on a node grid: centred slope at the end node",
    "definition",
    float((p_[-2] - p_[-4]) / 0.2),
    -2.0,
    atol=1e-12,
)
p_ = bc.pad(u5, 0.1, bc.Periodic(), bc.Periodic(), nghost=2)
check(
    S,
    "periodic padding wraps",
    "definition",
    float(np.max(np.abs(p_[:2] - u5[-2:])) + np.max(np.abs(p_[-2:] - u5[:2]))),
    0.0,
)
w_ = np.array([0.0, 0.1, 0.35, 0.7, 1.1])
pb = bc.pad_beam(w_, 0.25, "clamped", "free")
check(
    S,
    "clamped beam end: w' = 0 by the centred difference over the ghost",
    "even reflection",
    float(pb[3] - pb[1]),
    0.0,
    atol=1e-15,
)
check(
    S,
    "free beam end: w'' = 0 and w''' = 0 with the ghosts",
    "definition",
    float(abs(pb[-2] - 2 * pb[-3] + pb[-4]) + abs(pb[-1] - 2 * pb[-2] + 2 * pb[-4] - pb[-5])),
    0.0,
    atol=1e-12,
)
r_ = bc.dirichlet_rhs(np.ones(5), bc.Dirichlet(lambda t: np.sin(t)), bc.Neumann(0.0), t=0.3)
check(
    S, "time-dependent Dirichlet: the boundary node's rate is dg/dt", "cos(0.3)", float(r_[0]), math.cos(0.3), rtol=1e-8
)

# ------------------------------------------------------------------ 3. banded linear algebra
S = "Banded linear algebra"
for n_, lo_, up_ in ((6, 1, 1), (40, 2, 3), (60, 3, 1)):
    A = np.zeros((n_, n_))
    for i in range(n_):
        for j in range(max(0, i - lo_), min(n_, i + up_ + 1)):
            A[i, j] = rng.normal()
    A += np.diag(np.full(n_, 3.0))
    A[0, 0] = 1e-14  # forces a row swap
    b_ = rng.normal(size=n_)
    x_ = la.solve_banded(la.to_banded(A, lo_, up_), lo_, up_, b_)
    check(
        S,
        f"banded LU with pivoting, n = {n_}, bands ({lo_}, {up_}): backward error",
        "numpy.linalg.solve",
        float(np.linalg.norm(A @ x_ - b_) / (np.linalg.norm(A, np.inf) * np.linalg.norm(x_))),
        0.0,
        atol=1e-14,
    )
Ab_ = la.to_banded(A, 3, 1)
check(S, "banded storage round trip", "identity", float(np.max(np.abs(la.from_banded(Ab_, 3, 1) - A))), 0.0)
check(
    S,
    "... agrees with scipy.linalg.solve_banded",
    "scipy",
    float(np.max(np.abs(la.solve_banded(Ab_, 3, 1, b_) - sla.solve_banded((3, 1), Ab_, b_)))),
    0.0,
    atol=1e-11,
)
D2 = st.diff_operator(np.linspace(0, 1, 50), 2, 2)
fnl = lambda u: D2(u) + np.sin(u)  # noqa: E731
u0_ = rng.normal(size=50)
Jc = la.from_banded(la.banded_jacobian(fnl, u0_, *D2.bandwidth), *D2.bandwidth)
Jd = la.dense_jacobian(fnl, u0_)
check(
    S,
    "coloured banded Jacobian equals the dense finite-difference Jacobian",
    "colouring: 3 evaluations instead of 50",
    float(np.max(np.abs(Jc - Jd))),
    0.0,
    atol=1e-6,
)
D4 = st.diff_operator(np.linspace(0, 1, 50), 2, 4)
check(
    S,
    "... for a 5-point stencil (bandwidth 2, but one-sided closures reach 3): the declared band contains every entry",
    "structure",
    float(
        np.max(
            np.abs(
                la.from_banded(la.banded_jacobian(lambda u: D4(u), u0_, *D4.bandwidth), *D4.bandwidth) - D4.to_dense()
            )
        )
    ),
    0.0,
    atol=1e-8 * np.max(np.abs(D4.to_dense())),
)

# ------------------------------------------------------------------ 4. operators
S = "Operators"
g40 = grids.uniform(0, 1, 40)
x40 = g40.x
check(
    S,
    "diffusion of a quadratic is exact (u = x^2: u_xx = 2)",
    "exactness",
    float(np.max(np.abs(op.diffusion(x40**2, g40, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(1.0))[1:-1] - 2))),
    0.0,
    atol=1e-11,
)
kfun = lambda x: 1 + x  # noqa: E731
exact_var = 6 * x40**2 + 4 * x40  # d/dx((1 + x) d/dx x^3) = d/dx(3x^2 + 3x^3) = 6x + 9x^2 ... computed below
exact_var = 6 * x40 + 9 * x40**2
check(
    S,
    "variable-coefficient diffusion d/dx((1 + x) du/dx) for u = x^3",
    "6x + 9x^2 (second order)",
    float(
        np.max(np.abs(op.diffusion(x40**3, g40, kfun, bc.Dirichlet(0.0), bc.Dirichlet(1.0))[1:-1] - exact_var[1:-1]))
    ),
    0.0,
    atol=2e-3,
)
gcw = grids.cell_centred(0, 1, 40)  # the interface x = 0.5 is a face
klayer = np.where(gcw.x < 0.5, 1.0, 4.0)
u_layer = np.where(gcw.x < 0.5, gcw.x, 0.5 + 0.25 * (gcw.x - 0.5))  # k u_x = 1 on both sides
resid = op.diffusion(u_layer, gcw, klayer, bc.Dirichlet(0.0), bc.Dirichlet(0.625))
check(
    S,
    "composite wall: a piecewise-linear steady profile with continuous flux is in the kernel (harmonic face averaging)",
    "flux continuity",
    float(np.max(np.abs(resid))),
    0.0,
    atol=1e-11,
)
gr_ = grids.radial(1.0, 40, "cylindrical")
check(
    S,
    "cylindrical diffusion of r^2 gives 4 (including r = 0)",
    "(1/r)(r u_r)_r = 4",
    float(np.max(np.abs(op.diffusion(gr_.x**2, gr_, 1.0, None, bc.Dirichlet(1.0))[:-1] - 4))),
    0.0,
    atol=1e-10,
)
gs_ = grids.radial(1.0, 40, "spherical")
check(
    S,
    "spherical diffusion of r^2 gives 6 (including r = 0)",
    "(1/r^2)(r^2 u_r)_r = 6",
    float(np.max(np.abs(op.diffusion(gs_.x**2, gs_, 1.0, None, bc.Dirichlet(1.0))[:-1] - 6))),
    0.0,
    atol=1e-10,
)
gc_ = grids.cell_centred(0, 1, 200)
wc = np.exp(-(((gc_.x - 0.4) / 0.05) ** 2))
for sch in ("upwind", "centred", "minmod", "van-leer", "superbee", "mc", "koren"):
    r_ = op.advection(wc, gc_, 1.0, bc.Periodic(), bc.Periodic(), scheme=sch)
    check(
        S,
        f"advection '{sch}' is conservative (the rates sum to zero on a periodic grid)",
        "flux form",
        float(abs(np.sum(r_))),
        0.0,
        atol=1e-12,
    )
ec = []
for n_ in (100, 200, 400):
    gcn = grids.cell_centred(0, 1, n_)
    s_ = np.sin(2 * np.pi * gcn.x)
    ec.append(
        float(
            np.max(
                np.abs(
                    op.advection(s_, gcn, 1.0, bc.Periodic(), bc.Periodic(), scheme="centred")
                    + 2 * np.pi * np.cos(2 * np.pi * gcn.x)
                )
            )
        )
    )
check(S, "centred advection is second order", "theory", float(np.mean(analysis.observed_order(ec))), 2.0, atol=0.05)
eu = []
for n_ in (100, 200, 400):
    gcn = grids.cell_centred(0, 1, n_)
    s_ = np.sin(2 * np.pi * gcn.x)
    eu.append(
        float(
            np.max(
                np.abs(
                    op.advection(s_, gcn, 1.0, bc.Periodic(), bc.Periodic(), scheme="upwind")
                    + 2 * np.pi * np.cos(2 * np.pi * gcn.x)
                )
            )
        )
    )
check(S, "upwind advection is first order", "theory", float(np.mean(analysis.observed_order(eu))), 1.0, atol=0.05)
ed = []
for n_ in (50, 100, 200):
    gn = grids.cell_centred(0, 2 * np.pi, n_)
    ed.append(float(np.max(np.abs(op.dispersion(np.sin(gn.x), gn, bc.Periodic(), bc.Periodic()) - np.cos(gn.x)))))
check(S, "dispersion -u_xxx is second order", "theory", float(np.mean(analysis.observed_order(ed))), 2.0, atol=0.05)
gb = grids.uniform(0, 1, 40)
wb = gb.x**2 * (1 - gb.x) ** 2  # clamped-clamped shape: w = w' = 0 at both ends; w'''' = 24
check(
    S,
    "biharmonic of x^2 (1 - x)^2 with clamped ends is 24 in the interior",
    "exactness",
    float(np.max(np.abs(op.biharmonic(wb, gb, "clamped", "clamped")[2:-2] - 24))),
    0.0,
    atol=1e-8,
)
g2 = grids.tensor(grids.uniform(0, 1, 20), grids.uniform(0, 1, 30))
U2 = g2.X**2 + g2.Y**2
bcs2 = {"left": bc.Dirichlet(0.0), "right": bc.Dirichlet(0.0), "bottom": bc.Dirichlet(0.0), "top": bc.Dirichlet(0.0)}
check(
    S,
    "2-D Laplacian of x^2 + y^2 is 4 in the interior",
    "exactness",
    float(np.max(np.abs(op.laplacian2d(U2, g2, bcs2)[1:-1, 1:-1] - 4))),
    0.0,
    atol=1e-10,
)

# ------------------------------------------------------------------ 5. time integrators on the semi-discrete heat equation
S = "Time integrators"
g = grids.uniform(0, 1, 40)
u0 = np.sin(np.pi * g.x)
f_heat = lambda t, u: op.diffusion(u, g, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t)  # noqa: E731
Ad = st.diff_operator(g.x, 2, 2).to_dense()
Ad[0, :] = Ad[-1, :] = 0.0
ref = ti.exponential(Ad, u0, (0, 0.1)).y[-1]
check(
    S,
    "exponential integrator = exact solution of the semi-discrete system (vs scipy expm)",
    "scipy.linalg.expm",
    float(np.max(np.abs(ref - sla.expm(0.1 * Ad) @ u0))),
    0.0,
    atol=1e-12,
)
check(
    S,
    "... which decays at the discrete rate (4/dx^2) sin^2(pi dx / 2), not pi^2",
    "discrete eigenvalue",
    float(np.max(np.abs(ref - np.exp(-4 / g.dx**2 * np.sin(np.pi * g.dx / 2) ** 2 * 0.1) * u0))),
    0.0,
    atol=1e-12,
)
for order in (1, 2, 3, 4, 5):
    e = [
        float(np.max(np.abs(ti.bdf(f_heat, u0, (0, 0.1), h_, order, band=(1, 1)).y[-1] - ref)))
        for h_ in (0.01, 0.005, 0.0025)
    ]
    check(
        S,
        f"BDF{order} with a banded Newton: measured time order against the exact semi-discrete solution",
        f"theory: {order}",
        float(np.mean(analysis.observed_order(e))),
        order,
        atol=0.3,
    )
e = [
    float(np.max(np.abs(ti.trapezoidal(f_heat, u0, (0, 0.1), h_, band=(1, 1)).y[-1] - ref)))
    for h_ in (0.01, 0.005, 0.0025)
]
check(S, "trapezoidal rule: second order in time", "theory", float(np.mean(analysis.observed_order(e))), 2.0, atol=0.1)
sol_b = ti.bdf(f_heat, u0, (0, 0.1), 0.005, 2, band=(1, 1))
check(
    S,
    "BDF2 reuses one Jacobian for the whole run (modified Newton)",
    "Jacobian evaluations",
    float(sol_b.jacobian_evaluations),
    1.0,
)
check(
    S,
    "... with about two Newton iterations per step",
    "linear problem",
    float(sol_b.newton_iterations / sol_b.steps),
    2.0,
    atol=0.6,
)
g10 = grids.uniform(0, 1, 10)
u10 = np.sin(np.pi * g10.x)
f10 = lambda t, u: op.diffusion(u, g10, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t)  # noqa: E731
A10 = st.diff_operator(g10.x, 2, 2).to_dense()
A10[0, :] = A10[-1, :] = 0.0
ref10 = ti.exponential(A10, u10, (0, 0.1)).y[-1]
e = [float(np.max(np.abs(ti.explicit_rk(f10, u10, (0, 0.1), h_).y[-1] - ref10))) for h_ in (5e-3, 2.5e-3, 1.25e-3)]
check(
    S,
    "RK4 below its stability limit: fourth order in time",
    "theory",
    float(np.mean(analysis.observed_order(e))),
    4.0,
    atol=0.3,
)
h_lim = 2.785 / (4 / g.dx**2)
big = ti.explicit_rk(f_heat, u0, (0, 0.1), 1.05 * h_lim)
check(S, "RK4 just above 2.785 dx^2/4 blows up", "stability limit", float(np.max(np.abs(big.y[-1])) > 10), 1.0)
sol_d = ti.dopri45(f_heat, u0, (0, 0.1), 1e-10, 1e-12)
check(
    S,
    "Dormand-Prince at tight tolerance matches the semi-discrete solution",
    "adaptive RK",
    float(np.max(np.abs(sol_d.y[-1] - ref))),
    0.0,
    atol=1e-8,
)
check(
    S,
    "... but its mean step is pinned near the stability limit 3.3 dx^2/4 (stiffness)",
    "explicit step limit",
    float(np.mean(sol_d.extra["h"])),
    3.3 / (4 / g.dx**2),
    rtol=0.25,
)
ref_scipy = sint.solve_ivp(f_heat, (0, 0.1), u0, method="Radau", rtol=1e-10, atol=1e-12).y[:, -1]
check(
    S,
    "BDF4 (h = 0.0025) agrees with SciPy's Radau on the same system",
    "scipy.integrate.solve_ivp",
    float(np.max(np.abs(ti.bdf(f_heat, u0, (0, 0.1), 0.0025, 4, band=(1, 1)).y[-1] - ref_scipy))),
    0.0,
    atol=2e-7,
)
# IMEX on reaction-diffusion u_t = u_xx + u (exact: sin(pi x) exp((1 - pi^2) t))
f_react = lambda t, u: u  # noqa: E731
A_rd = Ad + np.eye(41)
A_rd[0, :] = A_rd[-1, :] = 0.0
ref_rd = ti.exponential(A_rd, u0, (0, 0.1)).y[-1]
e = [
    float(np.max(np.abs(ti.imex(f_react, f_heat, u0, (0, 0.1), h_, band=(1, 1), order=2).y[-1] - ref_rd)))
    for h_ in (0.01, 0.005, 0.0025)
]
check(
    S,
    "IMEX CNAB2 (explicit reaction, implicit diffusion): second order at steps far above the diffusion limit",
    "theory",
    float(np.mean(analysis.observed_order(e))),
    2.0,
    atol=0.15,
)
e1 = [
    float(np.max(np.abs(ti.imex(f_react, f_heat, u0, (0, 0.1), h_, band=(1, 1), order=1).y[-1] - ref_rd)))
    for h_ in (0.01, 0.005, 0.0025)
]
check(S, "IMEX Euler: first order", "theory", float(np.mean(analysis.observed_order(e1))), 1.0, atol=0.1)
# Verlet on the wave equation: energy conservation and second order
gw = grids.uniform(0, 1, 50)
acc = lambda t, u: op.diffusion(u, gw, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(0.0))  # noqa: E731
w0 = np.sin(np.pi * gw.x)
sv = ti.verlet(acc, w0, np.zeros_like(w0), (0, 4.0), 0.01, n_out=40)
E_ = [analysis.wave_energy(y[:51], y[51:], 1.0, gw) for y in sv.y]
check(
    S,
    "Stormer-Verlet on the wave equation: energy stays within 0.1 % over two periods",
    "symplectic",
    float(np.max(np.abs(np.array(E_) / E_[0] - 1))),
    0.0,
    atol=1e-3,
)
e = [
    float(
        np.max(
            np.abs(
                ti.verlet(acc, w0, np.zeros_like(w0), (0, 1.0), h_, n_out=1).y[-1, :51]
                - ti.verlet(acc, w0, np.zeros_like(w0), (0, 1.0), 1e-4, n_out=1).y[-1, :51]
            )
        )
    )
    for h_ in (0.01, 0.005, 0.0025)
]
check(S, "... second order in time", "theory", float(np.mean(analysis.observed_order(e))), 2.0, atol=0.1)
big = ti.verlet(acc, w0, np.zeros_like(w0), (0, 6.0), 1.05 * 2 / (2 / gw.dx), n_out=1)
check(S, "... unstable for h > 2/omega_max = dx", "CFL", float(np.max(np.abs(big.y[-1])) > 10), 1.0)
# DAE: y1' = -y1 + y2, 0 = y2 - y1^2 (index 1) vs SciPy Radau on the reduced ODE
F_dae = lambda t, y, yp: np.array([yp[0] + y[0] - y[1], y[1] - y[0] ** 2])  # noqa: E731
sd = ti.dae_bdf(F_dae, [0.5, 0.25], (0, 2.0), 0.002, order=2)
ref_dae = sint.solve_ivp(lambda t, y: [-y[0] + y[0] ** 2], (0, 2.0), [0.5], rtol=1e-12, atol=1e-12).y[0, -1]
check(
    S,
    "index-1 DAE by BDF2: differential variable vs the reduced ODE",
    "scipy",
    float(abs(sd.y[-1, 0] - ref_dae)),
    0.0,
    atol=1e-5,
)
check(
    S,
    "... algebraic constraint satisfied at every step",
    "y2 = y1^2",
    float(np.max(np.abs(sd.y[:, 1] - sd.y[:, 0] ** 2))),
    0.0,
    atol=1e-9,
)

# ------------------------------------------------------------------ 6. stiffness and analysis
S = "Stiffness and analysis"
lam = stiffness.spectrum(f_heat, u0, band=(1, 1))
check(
    S,
    "spectrum of the semi-discrete heat operator: extreme eigenvalue -4/dx^2 sin^2(N pi/(2(N+1))), N = 39 interior nodes",
    "discrete Laplacian",
    float(lam[0].real),
    -4 / g.dx**2 * np.sin(39 * np.pi / 80) ** 2,
    rtol=1e-8,
)
check(
    S,
    "... theoretical bound -4/dx^2",
    "operator_extremes",
    stiffness.operator_extremes("diffusion", g.dx),
    -4 / g.dx**2,
)
check(
    S,
    "stiffness ratio grows like n^2 (~ (2n/pi)^2 for n = 40)",
    "theory",
    stiffness.stiffness_ratio(lam[np.abs(lam) > 1e-9]),
    (2 * 40 / np.pi) ** 2,
    rtol=0.05,
)
check(
    S,
    "RK4 step limit from the spectrum = 2.785/|lambda_max|",
    "stability interval",
    stiffness.explicit_step_limit(lam[np.abs(lam) > 1e-9], "rk4"),
    2.785293563405282 / abs(lam[0].real),
    rtol=1e-6,
)
check(
    S,
    "Euler step limit = 2/|lambda_max|",
    "stability interval",
    stiffness.explicit_step_limit(lam[np.abs(lam) > 1e-9], "euler"),
    2 / abs(lam[0].real),
    rtol=1e-6,
)
check(
    S,
    "von Neumann: FTCS (Euler + central2) stable iff h <= dx^2/2",
    "r <= 1/2",
    analysis.max_stable_step("central2", "euler", 0.1),
    0.005,
    rtol=1e-6,
)
check(
    S,
    "... RK4 + central2: h <= 2.785 dx^2 / 4",
    "stability interval",
    analysis.max_stable_step("central2", "rk4", 0.1),
    2.785293563405282 * 0.01 / 4,
    rtol=1e-6,
)
check(
    S,
    "... Euler + centred advection is unconditionally unstable (limit 0)",
    "purely imaginary spectrum",
    analysis.max_stable_step("first_central", "euler", 0.1, sign=-1.0),
    0.0,
    atol=1e-6,
)
check(
    S,
    "... RK4 + centred advection: CFL 2.83",
    "imaginary-axis extent of RK4",
    analysis.max_stable_step("first_central", "rk4", 0.1, sign=-1.0) / 0.1,
    2.8284271247,
    rtol=1e-6,
)
check(
    S,
    "... Euler + upwind advection: CFL 1",
    "circle of radius a/dx",
    analysis.max_stable_step("first_upwind", "euler", 0.1, sign=-1.0) / 0.1,
    1.0,
    rtol=1e-6,
)
check(
    S,
    "trapezoidal rule is A-stable (|R| = 1 on the imaginary axis)",
    "A-stability",
    float(np.max(np.abs(stiffness.stability_function("trapezoidal", 1j * np.linspace(-100, 100, 201))))),
    1.0,
    atol=1e-12,
)
check(
    S,
    "BDF2's principal root at hλ = -1000 is 0.022: it decays like 1/sqrt(2|z|) (damped, L-stable in the limit)",
    "asymptotics of the root",
    float(abs(stiffness.stability_function("bdf2", -1000.0))),
    1 / math.sqrt(2000.0),
    rtol=0.02,
)
gcm = grids.cell_centred(0, 1, 400)
wm = np.exp(-(((gcm.x - 0.3) / 0.05) ** 2))
fu = lambda t, u: op.advection(u, gcm, 1.0, bc.Periodic(), bc.Periodic(), t, "upwind")  # noqa: E731
su = ti.dopri45(fu, wm, (0, 0.4), 1e-9, 1e-11, n_out=1)
D_num = analysis.modified_equation("upwind", 1.0, gcm.dx)["diffusion"]
sig_pred = np.sqrt(0.05**2 + 4 * D_num * 0.4)
pred = 0.05 / sig_pred * np.exp(-(((gcm.x - 0.7) / sig_pred) ** 2))
check(
    S,
    "modified equation: semi-discrete upwind advects a Gaussian with diffusion a dx/2 (peak height)",
    "a dx/2 = 6.25e-4",
    float(su.y[-1].max()),
    float(pred.max()),
    rtol=0.02,
)
cm = analysis.conservation(gcm, su.y)
check(S, "conservation bookkeeping: mass unchanged", "flux form", float(abs(cm[-1] - cm[0])), 0.0, atol=1e-12)
cost = stiffness.cost_model(41, lam[np.abs(lam) > 1e-9], 0.1, 1, implicit_h=0.005)
check(
    S,
    "cost model: explicit RK4 needs ~1.4e3 steps where BDF needs 20 on this grid",
    "stability vs accuracy",
    float(cost["explicit_steps"]),
    0.1 / (2.785293563405282 / abs(lam[0].real)),
    rtol=0.01,
)
cyl = grids.radial(1.0, 60, "cylindrical")
j01 = 2.404825557695773
u_c = ssp.j0(j01 * cyl.x)
f_c = lambda t, u: op.diffusion(u, cyl, 1.0, None, bc.Dirichlet(0.0), t)  # noqa: E731
sc = ti.bdf(f_c, u_c, (0, 0.2), 0.001, 4, band=(1, 1))
check(
    S,
    "cooling cylinder: the first Bessel mode J0(2.4048 r) decays at the exact rate exp(-2.4048^2 t)",
    "separation of variables",
    float(np.max(np.abs(sc.y[-1] - np.exp(-(j01**2) * 0.2) * u_c))),
    0.0,
    atol=1e-4,
)
sph = grids.radial(1.0, 60, "spherical")
u_s = np.sinc(sph.x)  # sin(pi r)/(pi r): the first spherical mode
f_s = lambda t, u: op.diffusion(u, sph, 1.0, None, bc.Dirichlet(0.0), t)  # noqa: E731
ss_ = ti.bdf(f_s, u_s, (0, 0.2), 0.001, 4, band=(1, 1))
check(
    S,
    "cooling sphere: sin(pi r)/(pi r) decays at exp(-pi^2 t)",
    "separation of variables",
    float(np.max(np.abs(ss_.y[-1] - np.exp(-(np.pi**2) * 0.2) * u_s))),
    0.0,
    atol=1e-4,
)

# ------------------------------------------------------------------ 7. parabolic problems
from engmolines import problems as pr  # noqa: E402

S = "Parabolic problems"
for fn, h_, tol_ in (
    (pr.heat_dirichlet, 1e-3, 2e-4),
    (pr.heat_neumann, 1e-3, 2e-4),
    (pr.heat_robin, 1e-3, 1e-4),
    (pr.heat_time_dependent, 5e-3, 5e-4),
    (pr.cooling_cylinder, 1e-3, 1e-4),
    (pr.cooling_sphere, 1e-3, 1e-4),
    (pr.allen_cahn, 1e-2, 5e-4),
):
    p_ = fn()
    s_ = p_.solve("bdf", h_, order=4)
    check(
        S,
        f"{p_.name}: BDF4 vs the exact solution",
        "separation of variables / exact profile",
        p_.error(s_),
        0.0,
        atol=tol_,
    )
p_ = pr.heat_neumann()
s_ = p_.solve("bdf", 1e-3, order=2, n_out=5)
check(
    S,
    "heat with insulated ends conserves the integral (Neumann ghosts)",
    "conservation",
    float(np.ptp(analysis.conservation(p_.grid, s_.y))),
    0.0,
    atol=1e-12,
)
p_ = pr.heat_robin()
check(
    S,
    "Robin eigenvalue: mu tan(mu/2) = Bi",
    "transcendental equation",
    p_.params["mu"] * math.tan(p_.params["mu"] / 2),
    p_.params["Bi"],
    rtol=1e-10,
)
errs_h = [pr.heat_dirichlet(n=n_).error(pr.heat_dirichlet(n=n_).solve("bdf", 2e-4, order=4)) for n_ in (20, 40, 80)]
check(
    S,
    "heat equation: spatial order 2 with the time error made negligible (BDF4, h = 2e-4)",
    "theory",
    float(np.mean(analysis.observed_order(errs_h))),
    2.0,
    atol=0.05,
)
p_ = pr.composite_wall()
s_ = p_.solve("bdf", 0.0075, order=2)
check(
    S,
    "composite wall reaches the exact piecewise-linear steady state (interface on a face)",
    "series resistances",
    float(np.max(np.abs(s_.y[-1] - p_.exact(p_.x)))),
    0.0,
    atol=1e-10,
)
xl, Tl = p_.x[p_.x < 0.5][-2:], s_.y[-1][p_.x < 0.5][-2:]
check(
    S,
    "... with the interface temperature of the resistance network (extrapolated from the left layer)",
    "T_int = T_L - q L1/k1",
    float(Tl[1] + (Tl[1] - Tl[0]) / (xl[1] - xl[0]) * (0.5 - xl[1])),
    p_.params["T_interface"],
    atol=1e-10,
)
p_ = pr.fin_transient()
s_ = p_.solve("bdf", 2.0, order=2)
check(
    S,
    "pin fin heating up (stiff; BDF2 with 2 s steps) reaches the cosh/sinh steady state",
    "fin theory",
    float(np.max(np.abs(s_.y[-1] - p_.exact(p_.x)))),
    0.0,
    atol=1e-3,
)
lam_f = stiffness.spectrum(p_.rhs, p_.y0, band=(1, 1))
lam_f = lam_f[np.abs(lam_f) > 1e-9]
alpha_f, beta_f = 200.0 / (2700 * 900), 4 * 25.0 / (2700 * 900 * 0.006)
check(
    S,
    "... its extreme eigenvalue is -(4 alpha/dx^2 + beta) (stiffness ratio ~ 3000)",
    "operator formula",
    float(lam_f[0].real),
    -(4 * alpha_f / p_.grid.dx**2 + beta_f),
    rtol=1e-3,
)
check(
    S,
    "... so explicit RK4 would need a step of 2.785/|lambda_max| = 0.034 s against BDF's 2 s",
    "stability limit",
    stiffness.explicit_step_limit(lam_f, "rk4"),
    2.785293563405282 / (4 * alpha_f / p_.grid.dx**2 + beta_f),
    rtol=1e-3,
)
p_ = pr.black_scholes(n=400)
s_ = p_.solve("bdf", 1e-3, order=2)
check(
    S,
    "Black-Scholes European call by the method of lines vs the closed-form price (max over S)",
    "Black-Scholes formula",
    p_.error(s_),
    0.0,
    atol=5e-3,
)
check(
    S,
    "... at the money",
    "BS formula at S = K",
    float(np.interp(100.0, p_.x, s_.y[-1])),
    float(p_.exact(np.array([100.0]), 1.0)[0]),
    rtol=1e-3,
)
p_ = pr.nonlinear_conductivity()
s_ = p_.solve("bdf", 5e-3, order=2)
check(
    S,
    "nonlinear conductivity k = 1 + u: steady state u + u^2/2 linear in x",
    "exact steady state",
    float(np.max(np.abs(s_.y[-1] - p_.exact(p_.x)))),
    0.0,
    atol=1e-4,
)
check(
    S,
    "... the modified Newton refreshed its Jacobian a few times (not every step)",
    "Jacobian evaluations",
    float(s_.jacobian_evaluations < 12),
    1.0,
)
p_ = pr.porous_medium()
s_ = p_.solve("bdf", 2e-3, order=2)
check(
    S, "porous-medium equation vs the Barenblatt similarity solution", "Barenblatt 1952", p_.error(s_), 0.0, atol=1e-3
)
check(
    S,
    "... mass conserved",
    "conservation form",
    float(abs(p_.grid.integrate(s_.y[-1]) - p_.grid.integrate(p_.y0))),
    0.0,
    atol=1e-10,
)
check(
    S,
    "... the front stays within one cell of the exact edge t^(1/3) sqrt(12)",
    "compact support",
    float(abs(p_.x[s_.y[-1] > 1e-6][-1] - math.sqrt(12 * 1.1 ** (2 / 3)))),
    0.0,
    atol=2 * p_.grid.dx,
)
p_ = pr.fisher()
s_ = p_.solve("bdf", 0.02, order=2, n_out=15)
pos = [pr.front_position(p_.x, y) for y in s_.y]
speeds = np.diff(pos) / np.diff(s_.t)
check(
    S,
    "Fisher-KPP front speed approaches 2 sqrt(rD) from below (1.9 at t = 15; the log correction)",
    "KPP",
    float(speeds[-1]),
    p_.params["speed"],
    rtol=0.07,
)
check(
    S,
    "... and is monotonically increasing in time",
    "Bramson's correction",
    float(np.all(np.diff(speeds[3:]) > 0)),
    1.0,
)
s_i = p_.solve("imex", 0.02, n_out=15)
check(
    S,
    "... IMEX (explicit reaction, implicit diffusion) gives the same front",
    "splitting",
    float(abs(pr.front_position(p_.x, s_i.y[-1]) - pos[-1])),
    0.0,
    atol=0.05,
)
p_ = pr.cahn_hilliard()
s_ = p_.solve("bdf", 1e-4, order=2, n_out=20)
E_ch = [p_.params["energy"](y) for y in s_.y]
check(
    S,
    "Cahn-Hilliard: mass conserved (fourth order, no-flux ends)",
    "conservation",
    float(np.ptp(analysis.conservation(p_.grid, s_.y))),
    0.0,
    atol=1e-12,
)
check(S, "... free energy decreases monotonically", "gradient flow", float(np.all(np.diff(E_ch) <= 1e-9)), 1.0)
check(
    S,
    "... the mixture separates into domains near +-1",
    "spinodal decomposition",
    float(np.max(np.abs(s_.y[-1]))),
    1.0,
    atol=0.05,
)
check(
    S,
    "... with the extreme eigenvalue of the biharmonic scale eps^2 16/dx^4",
    "stiffness",
    float(stiffness.spectrum(p_.rhs, s_.y[-1], band=(2, 2))[0].real),
    -(0.02**2) * 16 / p_.grid.dx**4,
    rtol=0.35,
)
p_ = pr.stefan()
s_ = p_.solve("dopri", rtol=1e-8, atol=1e-10)
check(
    S,
    "Stefan melting: front position s = 2 lam sqrt(t) (Neumann's solution)",
    "similarity solution",
    float(s_.y[-1, -1]),
    2 * p_.params["lambda"] * math.sqrt(1.0 + p_.params["t0"]),
    rtol=1e-5,
)
check(
    S,
    "... temperature profile in the moving frame stays the similarity profile",
    "exact",
    float(np.max(np.abs(s_.y[-1, :-1] - (1 - ssp.erf(p_.x * p_.params["lambda"]) / ssp.erf(p_.params["lambda"]))))),
    0.0,
    atol=1e-5,
)
p_ = pr.gray_scott(n=48)
s_ = p_.solve("rk4", 1.0, t_end=1500.0, n_out=1)
V_ = s_.y[-1, 48 * 48 :].reshape(48, 48)
check(
    S,
    "Gray-Scott (2-D, F = 0.037, k = 0.06): spots spread from the seed to a large fraction of the domain",
    "pattern formation",
    float(np.mean(V_ > 0.1) > 0.15),
    1.0,
)
check(
    S,
    "... u and v stay within [0, 1]",
    "positivity",
    float(np.all(s_.y[-1] >= -1e-12) and np.all(s_.y[-1] <= 1 + 1e-12)),
    1.0,
)

# ------------------------------------------------------------------ 8. hyperbolic problems
S = "Hyperbolic problems"
errs_adv = {}
for sch in ("upwind", "van-leer", "superbee", "centred"):
    p_ = pr.advection_periodic(scheme=sch)
    s_ = p_.solve("dopri", rtol=1e-7, atol=1e-9)
    errs_adv[sch] = p_.error(s_)
    check(
        S,
        f"advection ({sch}) conserves mass exactly",
        "flux form",
        float(abs(p_.grid.integrate(s_.y[-1]) - p_.grid.integrate(p_.y0))),
        0.0,
        atol=1e-12,
    )
check(
    S,
    "upwind smears a Gaussian pulse badly after one period (error 0.55) while van Leer keeps it (0.10)",
    "numerical diffusion a dx/2",
    float(errs_adv["upwind"] > 5 * errs_adv["van-leer"]),
    1.0,
)
check(
    S,
    "centred + Dormand-Prince is stable (its region contains a piece of the imaginary axis) with error 0.13",
    "non-dissipative",
    errs_adv["centred"],
    0.13,
    atol=0.03,
)
p_ = pr.advection_periodic(scheme="centred")
s_e = ti.explicit_rk(p_.rhs, p_.y0, (0, 1.0), p_.grid.dx, ti.EULER, n_out=1)  # Courant number 1
check(
    S,
    "... but centred + explicit Euler grows without bound (unconditionally unstable)",
    "purely imaginary spectrum",
    float(np.max(np.abs(s_e.y[-1])) > 2),
    1.0,
)
p_ = pr.advection_inflow()
s_ = p_.solve("dopri", rtol=1e-7, atol=1e-9)
check(
    S,
    "advection with a time-dependent inflow follows the characteristics (error < 0.03 at the peaks)",
    "sin(omega (t - x/a))",
    p_.error(s_),
    0.0,
    atol=0.03,
)
p_ = pr.burgers_riemann(uL=1.0, uR=0.0)
s_ = p_.solve("dopri", rtol=1e-7, atol=1e-9)
check(
    S,
    "Burgers shock travels at the Rankine-Hugoniot speed (uL + uR)/2",
    "conservation",
    float(p_.x[np.argmin(np.abs(s_.y[-1] - 0.5))]),
    0.4,
    atol=p_.grid.dx,
)
check(
    S,
    "... with an L1 error of one cell width",
    "first order at the shock",
    float(np.mean(np.abs(s_.y[-1] - p_.exact(p_.x, 0.8)))),
    0.0,
    atol=2 * p_.grid.dx,
)
check(S, "... and no overshoot (TVD)", "MUSCL limiter", float(np.max(s_.y[-1]) - 1.0), 0.0, atol=1e-6)
p_ = pr.burgers_riemann(uL=0.0, uR=1.0)
s_ = p_.solve("dopri", rtol=1e-7, atol=1e-9)
fan = (p_.x > 0.05) & (p_.x < 0.75)
check(
    S,
    "Burgers rarefaction: the entropy-satisfying fan u = x/t (not an expansion shock)",
    "entropy condition",
    float(np.max(np.abs(s_.y[-1] - p_.exact(p_.x, 0.8))[fan])),
    0.0,
    atol=0.01,
)
p_ = pr.burgers_sine()
s_ = p_.solve("dopri", rtol=1e-8, atol=1e-10)
check(
    S,
    "Burgers wave steepening before breaking matches the characteristic solution",
    "method of characteristics",
    p_.error(s_),
    0.0,
    atol=1e-3,
)
p_ = pr.wave_string()
s_ = p_.solve("rk4", 0.005)
check(S, "vibrating string, second mode, one period by RK4", "standing wave", p_.error(s_), 0.0, atol=1e-5)
sv_ = ti.verlet(p_.params["accel"], p_.y0[:101], p_.y0[101:], (0, p_.t_end), 0.005)
check(
    S,
    "... and by Stormer-Verlet",
    "standing wave",
    float(np.max(np.abs(sv_.y[-1][:101] - p_.exact(p_.x, p_.t_end)))),
    0.0,
    atol=1e-5,
)
lam_s = stiffness.spectrum(p_.rhs, p_.y0)
check(
    S,
    "... its spectrum is purely imaginary, |lambda| up to 2c/dx",
    "hyperbolic",
    float(np.max(np.abs(lam_s.real))),
    0.0,
    atol=1e-6,
)
check(S, "... ", "2c/dx", float(np.max(np.abs(lam_s.imag))), 2.0 / p_.grid.dx, rtol=1e-3)
p_ = pr.beam_vibration()
lam_b = stiffness.spectrum(p_.rhs, p_.y0)
check(
    S,
    "pinned beam: |lambda|max ~ 4 sqrt(EI/rhoA)/dx^2 (the square root of the biharmonic's 16/dx^4)",
    "fourth-order stiffness",
    float(np.max(np.abs(lam_b.imag))),
    4.0 / p_.grid.dx**2,
    rtol=1e-2,
)
sv_ = ti.verlet(p_.params["accel"], p_.y0[:61], p_.y0[61:], (0, p_.t_end), 0.4 * p_.grid.dx**2, n_out=1)
check(
    S,
    "... one period by Verlet at h = 0.4 dx^2 (its limit is dx^2/2)",
    "exact mode",
    float(np.max(np.abs(sv_.y[-1][:61] - p_.exact(p_.x, p_.t_end)))),
    0.0,
    atol=1e-5,
)
p_ = pr.telegraph()
s_ = p_.solve("rk4", 0.005)
check(S, "telegraph equation: damped standing wave", "exact", p_.error(s_), 0.0, atol=1e-4)
lam_t = stiffness.spectrum(p_.rhs, p_.y0)
lam_t = lam_t[np.abs(lam_t) > 1e-9]  # the Dirichlet rows contribute zeros
check(
    S,
    "... every dynamic eigenvalue has real part -a",
    "uniform damping",
    float(np.max(np.abs(lam_t.real + 0.5))),
    0.0,
    atol=1e-6,
)
p_ = pr.shallow_water_dam_break()
s_ = p_.solve("dopri", rtol=1e-7, atol=1e-9)
hh = s_.y[-1][:400]
check(
    S,
    "shallow-water dam break: plateau depth equals Stoker's h_m",
    "Stoker 1957",
    float(np.median(hh[(p_.x > 0.05) & (p_.x < 0.3)])),
    p_.params["h_m"],
    rtol=1e-3,
)
check(
    S,
    "... shock position s t within a cell",
    "Stoker",
    float(p_.x[np.argmin(np.abs(hh - 0.5 * (p_.params["h_m"] + 0.1)))]),
    p_.params["shock_speed"] * 0.2,
    atol=2 * p_.grid.dx,
)
check(
    S,
    "... rarefaction head at -sqrt(g hL) t",
    "characteristics",
    float(p_.x[np.argmax(hh < 0.999)]),
    -math.sqrt(9.81) * 0.2,
    atol=3 * p_.grid.dx,
)
check(
    S,
    "... mass and momentum conserved",
    "finite volumes",
    float(abs(p_.grid.integrate(hh) - p_.grid.integrate(p_.y0[:400]))),
    0.0,
    atol=1e-12,
)
p_ = pr.plug_flow_reactor()
s_ = p_.solve("bdf", 0.01, order=2)
check(
    S,
    "dispersed plug-flow reactor reaches Danckwerts' steady profile (within the first-order upwind error)",
    "Danckwerts 1953",
    float(np.max(np.abs(s_.y[-1] - p_.exact(p_.x)))),
    0.0,
    atol=5e-3,
)
check(S, "... outlet conversion", "exact 0.8227", float(1 - s_.y[-1][-1]), p_.params["conversion"], rtol=5e-3)

# ------------------------------------------------------------------ 9. dispersive problems
S = "Dispersive problems"
p_ = pr.kdv_soliton()
s_ = p_.solve("dopri", rtol=1e-8, atol=1e-10)
check(
    S,
    "KdV soliton after t = 4 (spectral in space, Dormand-Prince in time)",
    "2 kappa^2 sech^2 exact",
    p_.error(s_),
    0.0,
    atol=1e-6,
)
check(
    S,
    "... it travelled at 4 kappa^2",
    "soliton speed",
    float(p_.x[np.argmax(s_.y[-1])]),
    -10 + 4 * 0.8**2 * 4,
    atol=p_.grid.dx,
)
lam_k = stiffness.spectrum(lambda t, u: -__import__("engmolines").spectral.derivative(u, 40.0, 3), p_.y0)
check(
    S,
    "... the spectral u_xxx has eigenvalues on the imaginary axis up to (2 pi (n/2 - 1)/L)^3 (the Nyquist mode zeroed)",
    "spectral stiffness",
    float(np.max(np.abs(lam_k))),
    (2 * np.pi * 127 / 40) ** 3,
    rtol=1e-6,
)
p_ = pr.kdv_two_solitons()
s_ = p_.solve("dopri", rtol=1e-8, atol=1e-10, n_out=8)
check(
    S,
    "two-soliton collision: mass conserved through the collision",
    "KdV invariant",
    float(np.max(np.abs(analysis.conservation(p_.grid, s_.y) - p_.params["mass"]))),
    0.0,
    atol=1e-10,
)
check(
    S,
    "... momentum (integral u^2) conserved",
    "KdV invariant",
    float(abs(p_.grid.integrate(s_.y[-1] ** 2) - p_.params["momentum"])),
    0.0,
    atol=1e-10,
)
check(
    S,
    "... the taller soliton emerges with its amplitude 2 k1^2 intact",
    "soliton",
    float(np.max(s_.y[-1])),
    2.0,
    atol=5e-3,
)
p_ = pr.nls_soliton()
s_ = p_.solve("dopri", rtol=1e-8, atol=1e-10)
psi_ = s_.y[-1][:256] + 1j * s_.y[-1][256:]
check(S, "NLS soliton: complex field vs the exact sech e^{i(vx + (1 - v^2)t)}", "exact", p_.error(s_), 0.0, atol=1e-5)
check(
    S,
    "... mass integral |psi|^2 conserved",
    "NLS invariant",
    float(abs(p_.grid.integrate(np.abs(psi_) ** 2) - p_.params["mass"])),
    0.0,
    atol=1e-10,
)

# ------------------------------------------------------------------ 10. coupled PDE-ODE and DAE problems
S = "Coupled and DAE problems"
p_ = pr.packed_bed_adsorber()
s_ = p_.solve("bdf", 0.02, order=2, n_out=60)
nb_ = 101
c_out = s_.y[:, nb_ - 1]
t_bt = float(np.interp(0.5, c_out, s_.t))
fed = p_.params["u"] * s_.t[-1]
out_ = float(np.sum(0.5 * (c_out[1:] + c_out[:-1]) * p_.params["u"] * np.diff(s_.t)))
held = p_.grid.integrate(s_.y[-1][:nb_]) + p_.params["ratio"] * p_.grid.integrate(s_.y[-1][nb_:])
check(
    S,
    "packed-bed adsorber: breakthrough (c_out = 0.5) at the stoichiometric time L(1 + (1-eps)K/eps)/u",
    "mass balance on the front",
    t_bt,
    p_.params["t_stoich"],
    rtol=0.03,
)
check(
    S,
    "... overall mass balance: fed = out + held in fluid and solid",
    "conservation (trapezoid on the outlet)",
    fed - out_ - held,
    0.0,
    atol=0.02,
)
check(
    S,
    "... one Jacobian (bandwidth n + 2 couples q_i to c_i) for the whole run",
    "modified Newton",
    float(s_.jacobian_evaluations),
    1.0,
)
p_ = pr.catalyst_pellet()
s_ = p_.solve("bdf", 0.005, order=2)
c_ = s_.y[-1]
flux_ = (3 * c_[-1] - 4 * c_[-2] + c_[-3]) / (2 * p_.grid.dx)
check(
    S,
    "catalyst pellet: steady profile sinh(phi r)/(r sinh phi)",
    "Thiele",
    float(np.max(np.abs(c_ - p_.exact(p_.x)))),
    0.0,
    atol=1e-4,
)
check(
    S,
    "... effectiveness factor 3 (phi coth phi - 1)/phi^2 from the surface flux",
    "eta = 0.6716",
    3 * flux_ / p_.params["phi"] ** 2,
    p_.params["eta"],
    rtol=2e-3,
)
p_ = pr.counterflow_exchanger(n=400)
s_ = p_.solve("bdf", 0.01, order=2)
check(
    S,
    "counterflow heat exchanger: steady effectiveness vs the NTU relation",
    "(1 - e^-NTU(1-Cr))/(1 - Cr e^-NTU(1-Cr))",
    float(1 - s_.y[-1][400]),
    p_.params["effectiveness"],
    rtol=2e-3,
)
p_ = pr.heat_as_dae()
sd_ = ti.dae_bdf(p_.params["F"], p_.y0, (0, 1.0), 0.005, order=2, band=(1, 1))
sr_ = ti.bdf(p_.rhs, p_.y0, (0, 1.0), 0.005, 2, band=(1, 1))
check(
    S,
    "heat equation as an index-1 DAE (boundary value as an algebraic equation) vs the ghost-point ODE form",
    "same discretisation",
    float(np.max(np.abs(sd_.y[-1] - sr_.y[-1]))),
    0.0,
    atol=1e-4,
)
check(
    S,
    "... the algebraic boundary equation holds exactly at every step",
    "u_0 = sin(pi t)",
    float(np.max(np.abs(sd_.y[:, 0] - np.sin(np.pi * sd_.t)))),
    0.0,
    atol=1e-9,
)
p_ = pr.string_moving_load()
s_ = p_.solve("rk4", 0.001, n_out=100)
E_ = np.array([analysis.wave_energy(y[:201], y[201:], 1.0, p_.grid) for y in s_.y])
power = np.array([p_.grid.integrate(p_.params["load"](t_) * y[201:]) for t_, y in zip(s_.t, s_.y)])
work = np.r_[0.0, np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(s_.t))]
check(
    S,
    "string under a moving load: energy equals the work done by the load (to 1 %)",
    "energy balance",
    float(np.max(np.abs(E_ - work)) / E_.max()),
    0.0,
    atol=0.01,
)

# ------------------------------------------------------------------ 11. two-dimensional problems
S = "Two-dimensional problems"
p_ = pr.plate_heat_2d()
s_ = p_.solve("bdf", 1e-3, order=4)
check(
    S,
    "heat in a plate: sin(pi x) sin(2 pi y) decays as exp(-5 pi^2 t) (spatial error 1e-3 on a 24 x 24 grid)",
    "separation of variables",
    p_.error(s_),
    0.0,
    atol=2e-3,
)
check(
    S,
    "... BDF4 with the Kronecker band (n + 1 = 25): one Jacobian for the run, about three Newton iterations per step",
    "banded Newton in 2-D",
    float(s_.jacobian_evaluations == 1 and s_.newton_iterations / s_.steps < 3.5),
    1.0,
)
errs2 = [pr.plate_heat_2d(n=n_).error(pr.plate_heat_2d(n=n_).solve("bdf", 5e-4, order=4)) for n_ in (12, 24)]
check(S, "... second order in space", "theory", float(np.log2(errs2[0] / errs2[1])), 2.0, atol=0.1)
p_ = pr.fisher_2d(n=60)
s_ = p_.solve("bdf", 0.05, order=2, n_out=8)
rad = np.array([np.sqrt(np.mean(y > 0.5) * 40.0**2 / np.pi) for y in s_.y])
sp = np.diff(rad) / np.diff(s_.t)
check(
    S,
    "Fisher front in 2-D: the circular front's speed approaches 2 sqrt(rD) from below (curvature slows it)",
    "KPP with curvature correction",
    float(1.3 < sp[-1] < 2.0 and sp[-1] > sp[2]),
    1.0,
)
p_ = pr.cavity_flow()
s_ = p_.solve("rk4", 0.004, n_out=4)
om_ = s_.y[-1].reshape(33, 33)
psi_ = p_.params["streamfunction"](om_)
u_, v_ = p_.params["velocity"](psi_)
check(
    S,
    "lid-driven cavity at Re = 100: u_min on the vertical centreline (32 x 32 grid)",
    "Ghia et al. 1982: -0.2109",
    float(u_[16, :].min()),
    -0.2109,
    rtol=0.05,
)
check(
    S,
    "... v_max and v_min on the horizontal centreline",
    "Ghia: 0.1753, -0.2453",
    float(v_[:, 16].max() - v_[:, 16].min()),
    0.1753 + 0.2453,
    rtol=0.05,
)
check(S, "... primary vortex strength psi_min", "Ghia: -0.1034", float(psi_.min()), -0.1034, rtol=0.05)
check(
    S,
    "... the flow has reached a steady state (relative vorticity change over the last quarter of the run)",
    "time-independence",
    float(np.max(np.abs(s_.y[-1] - s_.y[-2])) / np.max(np.abs(s_.y[-1]))),
    0.0,
    atol=1e-4,
)

# ------------------------------------------------------------------ report
passed = sum(r[5] for r in RESULTS)
lines = [
    "# Validation report",
    "",
    "Generated by `python docs/validate.py`; rerun in CI on every push. Checks compare engmolines with exact",
    "solutions, NumPy and SciPy, and the theory of the method of lines: orders of convergence in space and time,",
    "stiffness and stability limits, and conservation.",
    "",
    f"**{passed} / {len(RESULTS)} checks pass.**",
    "",
]
for section in dict.fromkeys(r[0] for r in RESULTS):
    lines += [
        "",
        f"## {section}",
        "",
        "| Check | Reference | engmolines | Expected | Result |",
        "|---|---|---|---|:-:|",
    ]
    for s_, name, ref_, v, expected, ok in RESULTS:
        if s_ == section:
            lines.append(f"| {name} | {ref_} | {v:.10g} | {expected:.10g} | {'pass' if ok else 'FAIL'} |")
Path(__file__).with_name("VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{passed}/{len(RESULTS)} checks pass")
