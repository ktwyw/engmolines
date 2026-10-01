"""A library of method-of-lines example problems, each a Problem with a grid, a semi-discrete right-hand side, an
initial state, its Jacobian band and, where one exists, the exact solution. Parabolic group: the heat equation
with every boundary type, a composite wall, cooling cylinders and spheres, a heated fin, Black-Scholes,
nonlinear conductivity, the porous-medium equation, Fisher's travelling wave, Allen-Cahn kinks, Cahn-Hilliard
phase separation, Gray-Scott patterns in 2-D, and Stefan melting.

>>> from engmolines import problems
>>> p = problems.heat_dirichlet(n=40)
>>> p.n, p.band
(41, (1, 1))
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray

from . import boundary as bc
from . import grids
from . import integrators as ti
from . import operators as op


@dataclass
class Problem:
    """A semi-discrete PDE: rhs(t, y) on grid, initial state y0, Jacobian band (lo, up) for the implicit solvers,
    an exact solution exact(x, t) when known, the suggested final time, and free-text notes."""

    name: str
    grid: object
    rhs: object
    y0: ndarray
    band: tuple | None = None
    exact: object = None
    t_end: float = 1.0
    notes: str = ""
    params: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.y0)

    @property
    def x(self):
        return self.grid.x

    def _solve(self, method, h, t_end, **kw) -> ti.Solution:
        T = self.t_end if t_end is None else t_end
        h = h if h is not None else T / 200
        if method == "bdf":
            return ti.bdf(self.rhs, self.y0, (0.0, T), h, kw.pop("order", 2), band=self.band, **kw)
        if method == "trapezoidal":
            return ti.trapezoidal(self.rhs, self.y0, (0.0, T), h, band=self.band, **kw)
        if method == "rk4":
            return ti.explicit_rk(self.rhs, self.y0, (0.0, T), h, **kw)
        if method == "dopri":
            return ti.dopri45(self.rhs, self.y0, (0.0, T), **kw)
        if method == "imex":
            return ti.imex(
                self.params["rhs_explicit"], self.params["rhs_implicit"], self.y0, (0.0, T), h, band=self.band, **kw
            )
        raise ValueError(method)

    def solve(
        self, method: str = "bdf", h: float | None = None, t_end: float | None = None, allow_failure: bool = False, **kw
    ) -> ti.Solution:
        """Integrate with 'bdf' (order kw, default 2), 'trapezoidal', 'rk4', 'dopri' or 'imex' (needs rhs_explicit
        and rhs_implicit in params). Raises if the integrator reports a failure, unless allow_failure."""
        sol = self._solve(method, h, t_end, **kw)
        if sol.message != "ok" and not allow_failure:
            raise RuntimeError(f"{self.name}: {sol.message}")
        return sol

    def error(self, sol: ti.Solution) -> float:
        """Max-norm error of the final state against the exact solution."""
        return float(np.max(np.abs(sol.y[-1] - self.exact(self.x, sol.t[-1]))))


# ------------------------------------------------------------------ linear heat conduction
def heat_dirichlet(n: int = 40, kappa: float = 1.0, L: float = 1.0) -> Problem:
    """u_t = kappa u_xx, u(0) = u(L) = 0, u0 = sin(pi x/L): exact sin(pi x/L) exp(-kappa (pi/L)^2 t)."""
    g = grids.uniform(0, L, n)
    return Problem(
        "heat, Dirichlet ends",
        g,
        lambda t, u: op.diffusion(u, g, kappa, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t),
        np.sin(np.pi * g.x / L),
        (1, 1),
        lambda x, t: np.sin(np.pi * x / L) * np.exp(-kappa * (np.pi / L) ** 2 * t),
        0.2,
        "the model problem: one Fourier mode decaying",
    )


def heat_neumann(n: int = 40, kappa: float = 1.0) -> Problem:
    """Insulated ends: u0 = cos(pi x) decays as exp(-kappa pi^2 t); the mean is conserved."""
    g = grids.uniform(0, 1, n)
    return Problem(
        "heat, insulated ends",
        g,
        lambda t, u: op.diffusion(u, g, kappa, bc.Neumann(0.0), bc.Neumann(0.0), t),
        np.cos(np.pi * g.x),
        (1, 1),
        lambda x, t: np.cos(np.pi * x) * np.exp(-kappa * np.pi**2 * t),
        0.2,
        "Neumann ghosts conserve the integral",
    )


def heat_robin(n: int = 40, kappa: float = 1.0, Bi: float = 2.0) -> Problem:
    """Convective ends: u_x = Bi u at x = 0 and -u_x = Bi u at x = 1 (symmetric slab, Biot number Bi). The exact
    first mode is cos(mu (x - 1/2)) exp(-kappa mu^2 t) with mu tan(mu/2) = Bi (mu from a root find)."""
    from scipy.optimize import brentq

    mu = brentq(lambda m: m * np.tan(m / 2) - Bi, 1e-6, np.pi - 1e-6)
    g = grids.uniform(0, 1, n)
    left, right = bc.Robin(-Bi, 1.0, 0.0), bc.Robin(Bi, 1.0, 0.0)  # u_x - Bi u = 0 at 0; u_x + Bi u = 0 at 1
    return Problem(
        "heat, convective (Robin) ends",
        g,
        lambda t, u: op.diffusion(u, g, kappa, left, right, t),
        np.cos(mu * (g.x - 0.5)),
        (1, 1),
        lambda x, t: np.cos(mu * (x - 0.5)) * np.exp(-kappa * mu**2 * t),
        0.2,
        "Biot number sets the eigenvalue",
        {"mu": mu, "Bi": Bi},
    )


def heat_time_dependent(n: int = 40, kappa: float = 1.0, omega: float = 2.0) -> Problem:
    """Semi-infinite slab with a boundary temperature cos(omega t) at x = 0 (periodic heating): the exact solution
    exp(-k x) cos(omega t - k x), k = sqrt(omega / 2 kappa), imposed at x = L as well so the domain is exact."""
    k = np.sqrt(omega / (2 * kappa))
    L = 4 / k
    g = grids.uniform(0, L, n)
    ex = lambda x, t: np.exp(-k * x) * np.cos(omega * t - k * x)  # noqa: E731
    left, right = bc.Dirichlet(lambda t: ex(0.0, t)), bc.Dirichlet(lambda t: ex(L, t))
    return Problem(
        "heat, periodically heated boundary",
        g,
        lambda t, u: op.diffusion(u, g, kappa, left, right, t),
        ex(g.x, 0.0),
        (1, 1),
        ex,
        2 * np.pi / omega,
        "penetration depth 1/k; the wave lags by k x",
        {"k": k},
    )


def composite_wall(n: int = 80, k1: float = 1.0, k2: float = 5.0, T_left: float = 1.0, T_right: float = 0.0) -> Problem:
    """Two layers of equal thickness with conductivities k1, k2 (interface on a face), Dirichlet ends: the steady
    profile is piecewise linear with a kink at the interface; the transient starts from T_right everywhere."""
    g = grids.cell_centred(0, 1, n)
    k = np.where(g.x < 0.5, k1, k2)
    R = 0.5 / k1 + 0.5 / k2
    q = (T_left - T_right) / R
    T_int = T_left - q * 0.5 / k1

    def steady(x, t=None):
        return np.where(x < 0.5, T_left - q * x / k1, T_int - q * (x - 0.5) / k2)

    return Problem(
        "composite wall",
        g,
        lambda t, u: op.diffusion(u, g, k, bc.Dirichlet(T_left), bc.Dirichlet(T_right), t),
        np.full(n, T_right),
        (1, 1),
        steady,
        3.0,
        "exact steady state; harmonic face averaging makes the interface flux continuous",
        {"q": q, "T_interface": T_int},
    )


def cooling_cylinder(n: int = 60, kappa: float = 1.0, R: float = 1.0) -> Problem:
    """Infinite cylinder, surface at 0, first Bessel mode J0(j01 r/R) decaying as exp(-kappa (j01/R)^2 t)."""
    from scipy.special import j0

    g = grids.radial(R, n, "cylindrical")
    j01 = 2.404825557695773
    return Problem(
        "cooling cylinder",
        g,
        lambda t, u: op.diffusion(u, g, kappa, None, bc.Dirichlet(0.0), t),
        j0(j01 * g.x / R),
        (1, 1),
        lambda r, t: j0(j01 * r / R) * np.exp(-kappa * (j01 / R) ** 2 * t),
        0.2,
        "the r = 0 singularity handled by the control volume",
        {"j01": j01},
    )


def cooling_sphere(n: int = 60, kappa: float = 1.0, R: float = 1.0) -> Problem:
    """Sphere, surface at 0, first mode sin(pi r/R)/(pi r/R) decaying as exp(-kappa (pi/R)^2 t)."""
    g = grids.radial(R, n, "spherical")
    return Problem(
        "cooling sphere",
        g,
        lambda t, u: op.diffusion(u, g, kappa, None, bc.Dirichlet(0.0), t),
        np.sinc(g.x / R),
        (1, 1),
        lambda r, t: np.sinc(r / R) * np.exp(-kappa * (np.pi / R) ** 2 * t),
        0.2,
        "spherical control volumes",
    )


def fin_transient(
    n: int = 50,
    k: float = 200.0,
    D: float = 0.006,
    L: float = 0.1,
    h: float = 25.0,
    rho_c: float = 2700 * 900,
    T_b: float = 80.0,
    T_inf: float = 20.0,
) -> Problem:
    """Pin fin: rho c T_t = k T_xx - (4 h / D)(T - T_inf), base at T_b, convective tip; starts at T_inf. The steady
    state is the classic cosh/sinh fin solution (exact(x, None))."""
    g = grids.uniform(0, L, n)
    alpha, beta = k / rho_c, 4 * h / (rho_c * D)
    m = np.sqrt(4 * h / (k * D))
    r = h / (m * k)

    def steady(x, t=None):
        return T_inf + (T_b - T_inf) * (np.cosh(m * (L - x)) + r * np.sinh(m * (L - x))) / (
            np.cosh(m * L) + r * np.sinh(m * L)
        )

    def rhs(t, T):
        return op.diffusion(
            T, g, alpha, bc.Dirichlet(T_b), bc.Robin(h, k, h * T_inf), t, source=lambda x, T, t: -beta * (T - T_inf)
        )

    y0 = np.full(n + 1, T_inf)
    y0[0] = T_b
    return Problem(
        "pin fin heating up",
        g,
        rhs,
        y0,
        (1, 1),
        steady,
        900.0,
        "stiff: eigenvalues -53 to -0.03 1/s; exact steady state",
        {"m": m, "time_constant": 1 / beta},
    )


# ------------------------------------------------------------------ finance: Black-Scholes as a heat equation
def black_scholes(
    n: int = 200, K: float = 100.0, r: float = 0.05, sigma: float = 0.2, T: float = 1.0, S_max: float = 400.0
) -> Problem:
    """European call: V_t + (1/2) sigma^2 S^2 V_SS + r S V_S - r V = 0 backwards from the payoff, integrated forward in
    time-to-expiry tau. Boundary V(0) = 0, V(S_max) = S_max - K e^{-r tau}; exact Black-Scholes formula."""
    from scipy.special import ndtr

    g = grids.uniform(0, S_max, n)
    S = g.x

    def exact(S_, tau):
        S_ = np.asarray(S_, float)
        if tau <= 0:
            return np.maximum(S_ - K, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = (np.log(S_ / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        return np.where(S_ > 0, S_ * ndtr(d1) - K * np.exp(-r * tau) * ndtr(d2), 0.0)

    def rhs(tau, V):
        Vp = bc.pad(V, g.dx, bc.Dirichlet(0.0), bc.Dirichlet(lambda t: S_max - K * np.exp(-r * t)), "node", tau)
        VS = (Vp[2:] - Vp[:-2]) / (2 * g.dx)
        VSS = (Vp[2:] - 2 * Vp[1:-1] + Vp[:-2]) / g.dx**2
        out = 0.5 * sigma**2 * S**2 * VSS + r * S * VS - r * V
        out[0] = 0.0
        out[-1] = r * K * np.exp(-r * tau)  # d/dtau of the far boundary value
        return out

    return Problem(
        "Black-Scholes European call",
        g,
        rhs,
        np.maximum(S - K, 0.0),
        (1, 1),
        exact,
        T,
        "a heat equation in disguise; the payoff kink is smoothed instantly",
        {"K": K, "r": r, "sigma": sigma},
    )


# ------------------------------------------------------------------ nonlinear diffusion
def nonlinear_conductivity(n: int = 60, b: float = 1.0) -> Problem:
    """u_t = (k(u) u_x)_x with k = 1 + b u, ends at 1 and 0: the steady state solves d/dx((1 + b u) u_x) = 0, i.e.
    u + b u^2/2 linear in x (exact(x, None))."""
    g = grids.uniform(0, 1, n)
    C = 1 + b / 2  # value of u + b u^2 / 2 at x = 0

    def steady(x, t=None):
        s = C * (1 - x)  # u + b u^2/2 = s
        return (-1 + np.sqrt(1 + 2 * b * s)) / b

    y0 = np.where(g.x == 0, 1.0, 0.0)
    return Problem(
        "nonlinear conductivity k = 1 + b u",
        g,
        lambda t, u: op.diffusion(u, g, lambda x, u: 1 + b * u, bc.Dirichlet(1.0), bc.Dirichlet(0.0), t),
        y0,
        (1, 1),
        steady,
        2.0,
        "Jacobian changes with u: Newton refreshes it as needed",
    )


def porous_medium(n: int = 400, m: int = 2, t0: float = 0.1) -> Problem:
    """u_t = (u^m)_xx = (m u^{m-1} u_x)_x with the Barenblatt similarity solution (compact support, sharp front
    moving as t^{1/(m+1)}); the exact solution is the Barenblatt profile started at t0."""
    g = grids.cell_centred(-5, 5, n)  # the support reaches |x| = 3.58 at t = 1
    alpha = 1 / (m + 1)
    beta = alpha
    kk = (m - 1) / (2 * m * (m + 1))

    def barenblatt(x, t):
        tt = t + t0
        return tt**-alpha * np.maximum(1 - kk * (x / tt**beta) ** 2, 0.0) ** (1 / (m - 1))

    def rhs(t, u):  # written as the linear Laplacian of w = u^m: smooth in u, so Newton converges at the front
        return op.diffusion(np.maximum(u, 0.0) ** m, g, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t)

    return Problem(
        "porous-medium equation (Barenblatt)",
        g,
        rhs,
        barenblatt(g.x, 0.0),
        (1, 1),
        barenblatt,
        1.0,
        "degenerate diffusion: zero diffusivity where u = 0 keeps the front sharp; u_t = (u^m)_xx form",
        {"m": m},
    )


# ------------------------------------------------------------------ reaction-diffusion
def fisher(n: int = 400, r: float = 1.0, D: float = 1.0, L: float = 60.0) -> Problem:
    """u_t = D u_xx + r u (1 - u): a front invading the unstable state u = 0 at the minimal speed 2 sqrt(r D)
    (Kolmogorov-Petrovsky-Piskunov). Started from a step; exact() returns the front position 2 sqrt(rD) t."""
    g = grids.uniform(0, L, n)
    c = 2 * np.sqrt(r * D)

    def rhs(t, u):
        return op.diffusion(u, g, D, bc.Neumann(0.0), bc.Neumann(0.0), t, source=lambda x, u, t: r * u * (1 - u))

    y0 = np.where(g.x < 5.0, 1.0, 0.0)
    return Problem(
        "Fisher-KPP travelling wave",
        g,
        rhs,
        y0,
        (1, 1),
        None,
        15.0,
        "front speed converges to 2 sqrt(rD) from above",
        {
            "speed": c,
            "rhs_explicit": lambda t, u: r * u * (1 - u),
            "rhs_implicit": lambda t, u: op.diffusion(u, g, D, bc.Neumann(0.0), bc.Neumann(0.0), t),
        },
    )


def front_position(x, u, level: float = 0.5) -> float:
    """Position where a decreasing profile crosses `level` (linear interpolation)."""
    i = int(np.argmax(u < level))
    if i == 0:
        return float(x[0])
    return float(x[i - 1] + (u[i - 1] - level) / (u[i - 1] - u[i]) * (x[i] - x[i - 1]))


def allen_cahn(n: int = 200, eps: float = 0.01, L: float = 2.0) -> Problem:
    """u_t = eps u_xx + u - u^3: the kink tanh(x / sqrt(2 eps)) is a stationary solution (exact for all t)."""
    g = grids.uniform(-L / 2, L / 2, n)
    kink = lambda x, t: np.tanh(x / np.sqrt(2 * eps))  # noqa: E731

    def rhs(t, u):
        return op.diffusion(u, g, eps, bc.Dirichlet(-1.0), bc.Dirichlet(1.0), t, source=lambda x, u, t: u - u**3)

    return Problem(
        "Allen-Cahn kink",
        g,
        rhs,
        kink(g.x, 0.0),
        (1, 1),
        kink,
        5.0,
        "a stationary front: the discretisation must not move it",
        {"eps": eps},
    )


def cahn_hilliard(n: int = 128, eps: float = 0.02, L: float = 1.0, seed: int = 3) -> Problem:
    """u_t = -(eps^2 u_xxxx) + (u^3 - u)_xx with no-flux ends: mass is conserved and the free energy decreases;
    a random mixture separates into domains of +-1. Fourth-order and very stiff (eigenvalues ~ -eps^2 16/dx^4)."""
    g = grids.uniform(0, L, n)
    rng = np.random.default_rng(seed)

    def rhs(t, u):
        mu = u**3 - u - eps**2 * op.diffusion(u, g, 1.0, bc.Neumann(0.0), bc.Neumann(0.0))  # chemical potential
        return op.diffusion(mu, g, 1.0, bc.Neumann(0.0), bc.Neumann(0.0))

    def energy(u):
        ux = np.gradient(u, g.x)
        return g.integrate(0.25 * (u**2 - 1) ** 2 + 0.5 * eps**2 * ux**2)

    return Problem(
        "Cahn-Hilliard phase separation",
        g,
        rhs,
        0.05 * rng.normal(size=n + 1),
        (2, 2),
        None,
        0.05,
        "mass conserved, energy monotone; band 2 from the biharmonic",
        {"energy": energy},
    )


def gray_scott(
    n: int = 64, F: float = 0.037, k: float = 0.06, Du: float = 2e-5, Dv: float = 1e-5, L: float = 1.0, seed: int = 0
) -> Problem:
    """Gray-Scott reaction-diffusion in 2-D (periodic): u_t = Du lap u - u v^2 + F (1 - u), v_t = Dv lap v + u v^2 -
    (F + k) v; from a perturbed square of v, spots and stripes grow. State: u and v flattened and concatenated."""
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    rng = np.random.default_rng(seed)
    u = np.ones((n, n))
    v = np.zeros((n, n))
    box = (np.abs(X - 0.5) < 0.1) & (np.abs(Y - 0.5) < 0.1)
    u[box], v[box] = 0.5, 0.25
    u += 0.02 * rng.random((n, n))
    v += 0.02 * rng.random((n, n))

    def lap(A):
        return (np.roll(A, 1, 0) + np.roll(A, -1, 0) + np.roll(A, 1, 1) + np.roll(A, -1, 1) - 4 * A) / dx**2

    def rhs(t, y):
        U, V = y[: n * n].reshape(n, n), y[n * n :].reshape(n, n)
        uvv = U * V * V
        return np.r_[(Du * lap(U) - uvv + F * (1 - U)).ravel(), (Dv * lap(V) + uvv - (F + k) * V).ravel()]

    g = grids.tensor(grids.cell_centred(0, L, n), grids.cell_centred(0, L, n))
    return Problem(
        "Gray-Scott patterns (2-D)",
        g,
        rhs,
        np.r_[u.ravel(), v.ravel()],
        None,
        None,
        2000.0,
        "explicit RK is competitive here: the reaction limits the step more than the small diffusivities",
        {"n": n, "F": F, "k": k},
    )


# ------------------------------------------------------------------ Stefan problem (moving boundary)
def stefan(n: int = 100, St: float = 1.0, t0: float = 0.05) -> Problem:
    """One-phase Stefan melting: liquid 0 < x < s(t) with T(0) = 1, T(s) = 0, front speed s' = -T_x(s) / St (latent
    heat: Stefan number St). Mapped to xi = x / s(t) in [0, 1] (Landau transformation), which turns the moving
    boundary into an extra ODE for s. Exact: s = 2 lam sqrt(t), lam from sqrt(pi) lam exp(lam^2) erf(lam) = 1/St."""
    from scipy.optimize import brentq
    from scipy.special import erf

    lam = brentq(lambda z: np.sqrt(np.pi) * z * np.exp(z**2) * erf(z) - 1 / St, 1e-6, 3.0)
    g = grids.uniform(0, 1, n)
    xi = g.x
    dxi = g.dx

    def exact(x, t):
        tt = t + t0
        s_ = 2 * lam * np.sqrt(tt)
        return 1 - erf(x * s_ / (2 * np.sqrt(tt))) / erf(lam)

    def rhs(t, y):
        T, s = y[:-1], y[-1]
        Tp = bc.pad(T, dxi, bc.Dirichlet(1.0), bc.Dirichlet(0.0), "node", t)
        Tx = (Tp[2:] - Tp[:-2]) / (2 * dxi)
        Txx = (Tp[2:] - 2 * Tp[1:-1] + Tp[:-2]) / dxi**2
        grad_front = (3 * T[-1] - 4 * T[-2] + T[-3]) / (2 * dxi) / s  # T_x at the front, one-sided second order
        s_dot = -grad_front / St
        out = Txx / s**2 + xi * s_dot / s * Tx  # T_t at fixed xi: T_xx/s^2 + (xi s'/s) T_xi
        out[0] = out[-1] = 0.0
        return np.r_[out, s_dot]

    s0 = 2 * lam * np.sqrt(t0)
    y0 = np.r_[1 - erf(xi * lam) / erf(lam), s0]  # the similarity profile is a function of xi alone
    return Problem(
        "Stefan melting front (Landau transformation)",
        g,
        rhs,
        y0,
        None,
        exact,
        1.0,
        "state: temperature on xi in [0, 1] plus the front position s(t)",
        {"lambda": lam, "t0": t0, "St": St},
    )


PARABOLIC = [
    heat_dirichlet,
    heat_neumann,
    heat_robin,
    heat_time_dependent,
    composite_wall,
    cooling_cylinder,
    cooling_sphere,
    fin_transient,
    black_scholes,
    nonlinear_conductivity,
    porous_medium,
    fisher,
    allen_cahn,
    cahn_hilliard,
    gray_scott,
    stefan,
]


# ================================================================== hyperbolic problems
def advection_periodic(n: int = 200, a: float = 1.0, scheme: str = "van-leer", L: float = 1.0) -> Problem:
    """u_t + a u_x = 0 on a periodic domain with a Gaussian pulse: exact translation. scheme 'centred' (second
    order, non-dissipative: needs an integrator whose stability region contains the imaginary axis), 'upwind',
    or a limiter."""
    g = grids.cell_centred(0, L, n)
    ex = lambda x, t: np.exp(-((((x - 0.5 * L - a * t) % L - 0.5 * L) / (0.05 * L)) ** 2))  # noqa: E731
    return Problem(
        f"linear advection, periodic ({scheme})",
        g,
        lambda t, u: op.advection(u, g, a, bc.Periodic(), bc.Periodic(), t, scheme),
        ex(g.x, 0.0),
        (2, 2),
        ex,
        L / a,
        "the exact solution is the initial pulse translated; every error is the scheme's",
        {"a": a, "scheme": scheme},
    )


def advection_inflow(n: int = 200, a: float = 1.0, omega: float = 4 * np.pi, scheme: str = "van-leer") -> Problem:
    """u_t + a u_x = 0 on [0, 1] with a time-dependent inflow u(0, t) = sin(omega t) and outflow (zero gradient) at
    x = 1: exact u = sin(omega (t - x/a)) behind the first characteristic, 0 ahead of it."""
    g = grids.cell_centred(0, 1, n)
    ex = lambda x, t: np.where(x < a * t, np.sin(omega * (t - x / a)), 0.0)  # noqa: E731
    left = bc.Dirichlet(lambda t: np.sin(omega * t))
    return Problem(
        "advection with a time-dependent inflow",
        g,
        lambda t, u: op.advection(u, g, a, left, bc.Neumann(0.0), t, scheme),
        np.zeros(n),
        (2, 2),
        ex,
        1.5,
        "one boundary condition only: at the inflow",
        {"a": a, "omega": omega},
    )


def burgers_riemann(n: int = 400, uL: float = 1.0, uR: float = 0.0) -> Problem:
    """Inviscid Burgers u_t + (u^2/2)_x = 0 with Riemann data: a shock at speed (uL + uR)/2 for uL > uR, a
    rarefaction fan u = x/t between uL t and uR t for uL < uR."""
    g = grids.cell_centred(-1, 1, n)

    def ex(x, t):
        if uL > uR:
            return np.where(x < 0.5 * (uL + uR) * t, uL, uR)
        return np.where(x < uL * t, uL, np.where(x > uR * t, uR, x / max(t, 1e-300)))

    rhs = lambda t, u: op.conservation_law(
        u, g, lambda w: 0.5 * w * w, lambda w: np.abs(w), "transmissive", "transmissive", t
    )  # noqa: E731
    return Problem(
        f"Burgers Riemann problem ({'shock' if uL > uR else 'rarefaction'})",
        g,
        rhs,
        np.where(g.x < 0, uL, uR),
        None,
        ex,
        0.8,
        "Rankine-Hugoniot and the entropy condition",
        {"uL": uL, "uR": uR},
    )


def burgers_sine(n: int = 400) -> Problem:
    """u_t + (u^2/2)_x = 0, periodic, u0 = sin(2 pi x): breaks at t = 1/(2 pi); before breaking the exact
    solution follows from characteristics u = sin(2 pi (x - u t)) (solved by fixed-point iteration)."""
    g = grids.cell_centred(0, 1, n)

    def ex(x, t):
        u = np.sin(2 * np.pi * x)
        for _ in range(200):
            u = np.sin(2 * np.pi * (x - u * t))
        return u

    rhs = lambda t, u: op.conservation_law(u, g, lambda w: 0.5 * w * w, lambda w: np.abs(w), "periodic", "periodic", t)  # noqa: E731
    return Problem(
        "Burgers wave steepening",
        g,
        rhs,
        np.sin(2 * np.pi * g.x),
        None,
        ex,
        0.12,
        "exact until the breaking time 0.159",
        {"t_break": 1 / (2 * np.pi)},
    )


def wave_string(n: int = 100, c: float = 1.0, L: float = 1.0, mode: int = 2) -> Problem:
    """u_tt = c^2 u_xx, fixed ends, standing wave sin(m pi x/L) cos(m pi c t/L). State [u, v] for the first-order
    system; params['accel'] is the acceleration for Stormer-Verlet."""
    g = grids.uniform(0, L, n)
    w = mode * np.pi / L
    accel = lambda t, u: op.diffusion(u, g, c**2, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t)  # noqa: E731

    def rhs(t, y):
        u, v = y[: n + 1], y[n + 1 :]
        return np.r_[v, accel(t, u)]

    ex = lambda x, t: np.sin(w * x) * np.cos(w * c * t)  # noqa: E731
    p = Problem(
        "vibrating string",
        g,
        rhs,
        np.r_[np.sin(w * g.x), np.zeros(n + 1)],
        None,
        ex,
        2 * L / (mode * c),
        "purely imaginary spectrum +-i c k: explicit RK4 or Verlet",
        {"accel": accel, "c": c},
    )
    p.error = lambda sol: float(np.max(np.abs(sol.y[-1][: n + 1] - ex(g.x, sol.t[-1]))))
    return p


def beam_vibration(n: int = 60, EI_rhoA: float = 1.0, L: float = 1.0, mode: int = 1) -> Problem:
    """Euler-Bernoulli beam w_tt = -(EI/rho A) w_xxxx, pinned ends: mode sin(m pi x) at omega = (m pi)^2 sqrt(EI/rho A).
    The biharmonic operator makes omega_max ~ 4 sqrt(EI/rhoA)/dx^2: far stiffer than the string."""
    g = grids.uniform(0, L, n)
    w = mode * np.pi / L
    omega = w**2 * np.sqrt(EI_rhoA)
    accel = lambda t, u: -EI_rhoA * op.biharmonic(u, g, "pinned", "pinned")  # noqa: E731

    def rhs(t, y):
        u, v = y[: n + 1], y[n + 1 :]
        return np.r_[v, accel(t, u)]

    ex = lambda x, t: np.sin(w * x) * np.cos(omega * t)  # noqa: E731
    p = Problem(
        "pinned beam vibration",
        g,
        rhs,
        np.r_[np.sin(w * g.x), np.zeros(n + 1)],
        None,
        ex,
        2 * np.pi / omega,
        "eigenvalues +-i omega up to 4/dx^2: Verlet needs h < dx^2/2",
        {"accel": accel, "omega": omega},
    )
    p.error = lambda sol: float(np.max(np.abs(sol.y[-1][: n + 1] - ex(g.x, sol.t[-1]))))
    return p


def telegraph(n: int = 100, c: float = 1.0, a: float = 0.5) -> Problem:
    """u_tt + 2 a u_t = c^2 u_xx (transmission line with loss), fixed ends: u = e^{-a t} sin(pi x) cos(omega t),
    omega = sqrt(c^2 pi^2 - a^2); first-order system, damped so every eigenvalue has real part -a."""
    g = grids.uniform(0, 1, n)
    omega = np.sqrt(c**2 * np.pi**2 - a**2)

    def rhs(t, y):
        u, v = y[: n + 1], y[n + 1 :]
        dv = op.diffusion(u, g, c**2, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t) - 2 * a * v
        dv[0] = dv[-1] = 0.0  # the ends are fixed: their velocity stays zero
        return np.r_[v, dv]

    ex = lambda x, t: np.exp(-a * t) * np.sin(np.pi * x) * np.cos(omega * t)  # noqa: E731
    v0 = -a * np.sin(np.pi * g.x)  # d/dt at t = 0
    p = Problem(
        "telegraph equation",
        g,
        rhs,
        np.r_[np.sin(np.pi * g.x), v0],
        None,
        ex,
        3.0,
        "damped wave: BDF is dissipative anyway, RK4 fine",
        {"omega": omega, "a": a},
    )
    p.error = lambda sol: float(np.max(np.abs(sol.y[-1][: n + 1] - ex(g.x, sol.t[-1]))))
    return p


def shallow_water_dam_break(n: int = 400, hL: float = 1.0, hR: float = 0.1, gr: float = 9.81) -> Problem:
    """Shallow-water equations h_t + (hu)_x = 0, (hu)_t + (hu^2 + g h^2/2)_x = 0; dam break with still water on both
    sides: Stoker's exact solution (rarefaction on the left, shock on the right). State rows [h, hu]."""
    from scipy.optimize import brentq

    g = grids.cell_centred(-1, 1, n)
    cL = np.sqrt(gr * hL)

    def f_hm(hm):  # Stoker: matching the rarefaction to the shock
        cm = np.sqrt(gr * hm)
        um = 2 * (cL - cm)
        s = um * hm / (hm - hR) if hm != hR else 0.0
        return um - (hm - hR) * np.sqrt(0.5 * gr * (hm + hR) / (hm * hR)) if hm > hR else 1.0, s

    hm = brentq(lambda h_: f_hm(h_)[0], hR * 1.0001, hL * 0.9999)
    cm = np.sqrt(gr * hm)
    um = 2 * (cL - cm)
    s_shock = um * hm / (hm - hR)

    def ex(x, t):
        xi = x / max(t, 1e-300)
        h_ = np.where(
            xi < -cL, hL, np.where(xi < um - cm, (2 * cL - xi) ** 2 / (9 * gr), np.where(xi < s_shock, hm, hR))
        )
        u_ = np.where(xi < -cL, 0.0, np.where(xi < um - cm, 2 * (cL + xi) / 3, np.where(xi < s_shock, um, 0.0)))
        return np.vstack([h_, h_ * u_])

    def flux(W):
        h_, q = W[0], W[1]
        u_ = q / h_
        return np.vstack([q, q * u_ + 0.5 * gr * h_ * h_])

    def speed(W):
        return np.abs(W[1] / W[0]) + np.sqrt(gr * W[0])

    rhs = lambda t, y: op.conservation_law(y.reshape(2, n), g, flux, speed, "transmissive", "transmissive", t).ravel()  # noqa: E731
    h0 = np.where(g.x < 0, hL, hR)
    p = Problem(
        "shallow-water dam break (Stoker)",
        g,
        rhs,
        np.r_[h0, np.zeros(n)],
        None,
        ex,
        0.2,
        "a 2 x 2 hyperbolic system: rarefaction, plateau, shock",
        {"h_m": hm, "u_m": um, "shock_speed": s_shock},
    )
    p.error = lambda sol: float(np.max(np.abs(sol.y[-1][:n] - ex(g.x, sol.t[-1])[0])))
    return p


def plug_flow_reactor(n: int = 100, Pe: float = 10.0, Da: float = 2.0) -> Problem:
    """Axially dispersed plug-flow reactor (dimensionless): c_t = -c_x + (1/Pe) c_xx - Da c with Danckwerts
    boundary conditions c - (1/Pe) c_x = 1 at the inlet, c_x = 0 at the outlet; started empty. The steady state is
    the classic Danckwerts solution, exact(x, None)."""
    g = grids.uniform(0, 1, n)
    q = np.sqrt(1 + 4 * Da / Pe)
    ap, am = Pe * (1 + q) / 2, Pe * (1 - q) / 2

    def steady(x, t=None):
        A = np.array([[1 - am / Pe, 1 - ap / Pe], [am * np.exp(am), ap * np.exp(ap)]])
        c1, c2 = np.linalg.solve(A, [1.0, 0.0])
        return c1 * np.exp(am * x) + c2 * np.exp(ap * x)

    def rhs(t, c):
        adv = op.advection(c, g, 1.0, bc.Robin(1.0, -1 / Pe, 1.0), bc.Neumann(0.0), t, "upwind")
        diff = op.diffusion(c, g, 1 / Pe, bc.Robin(1.0, -1 / Pe, 1.0), bc.Neumann(0.0), t)
        return adv + diff - Da * c

    return Problem(
        "axially dispersed plug-flow reactor",
        g,
        rhs,
        np.zeros(n + 1),
        (1, 1),
        steady,
        4.0,
        "Danckwerts boundary conditions as Robin/Neumann ghosts; exact steady conversion",
        {"Pe": Pe, "Da": Da, "conversion": 1 - steady(np.array([1.0]))[0]},
    )


HYPERBOLIC = [
    advection_periodic,
    advection_inflow,
    burgers_riemann,
    burgers_sine,
    wave_string,
    beam_vibration,
    telegraph,
    shallow_water_dam_break,
    plug_flow_reactor,
]


# ================================================================== dispersive problems (Fourier spectral)
def kdv_soliton(n: int = 256, L: float = 40.0, kappa: float = 0.8, x0: float = -10.0) -> Problem:
    """Korteweg-de Vries u_t + 6 u u_x + u_xxx = 0, periodic, Fourier spectral in space: the soliton
    2 kappa^2 sech^2(kappa (x - 4 kappa^2 t - x0)) travels at 4 kappa^2 unchanged."""
    from . import spectral

    x = spectral.grid(-L / 2, L / 2, n)
    g = grids.periodic(-L / 2, L / 2, n)
    ex = lambda x_, t: 2 * kappa**2 / np.cosh(kappa * (((x_ - 4 * kappa**2 * t - x0) + L / 2) % L - L / 2)) ** 2  # noqa: E731

    def rhs(t, u):
        return -6 * u * spectral.derivative(u, L, 1) - spectral.derivative(u, L, 3)

    return Problem(
        "KdV soliton (spectral)",
        g,
        rhs,
        ex(x, 0.0),
        None,
        ex,
        4.0,
        "eigenvalues up to i (pi n/L)^3 = i 265: dopri steps ~ 0.01",
        {"kappa": kappa, "speed": 4 * kappa**2},
    )


def kdv_two_solitons(n: int = 256, L: float = 40.0, k1: float = 1.0, k2: float = 0.6) -> Problem:
    """Two KdV solitons of different speeds set up to collide: they pass through each other unchanged except for a
    phase shift; mass integral u and momentum integral u^2 are conserved throughout."""
    from . import spectral

    x = spectral.grid(-L / 2, L / 2, n)
    g = grids.periodic(-L / 2, L / 2, n)
    u0 = 2 * k1**2 / np.cosh(k1 * (x + 12)) ** 2 + 2 * k2**2 / np.cosh(k2 * (x + 2)) ** 2

    def rhs(t, u):
        return -6 * u * spectral.derivative(u, L, 1) - spectral.derivative(u, L, 3)

    return Problem(
        "KdV two-soliton collision (spectral)",
        g,
        rhs,
        u0,
        None,
        None,
        8.0,
        "the taller soliton overtakes the shorter one",
        {"k1": k1, "k2": k2, "mass": g.integrate(u0), "momentum": g.integrate(u0**2)},
    )


def nls_soliton(n: int = 256, L: float = 40.0, v: float = 0.5) -> Problem:
    """Focusing nonlinear Schrodinger i psi_t + psi_xx + 2 |psi|^2 psi = 0: soliton sech(x - 2 v t) exp(i(v x +
    (1 - v^2) t)). State [Re psi, Im psi]; mass integral |psi|^2 and energy are conserved."""
    from . import spectral

    x = spectral.grid(-L / 2, L / 2, n)
    g = grids.periodic(-L / 2, L / 2, n)

    def ex(x_, t):
        xs = ((x_ - 2 * v * t) + L / 2) % L - L / 2
        return 1 / np.cosh(xs) * np.exp(1j * (v * x_ + (1 - v**2) * t))

    def rhs(t, y):
        psi = y[:n] + 1j * y[n:]
        dpsi = 1j * (spectral.derivative(psi, L, 2) + 2 * np.abs(psi) ** 2 * psi)
        return np.r_[dpsi.real, dpsi.imag]

    psi0 = ex(x, 0.0)
    p = Problem(
        "NLS soliton (spectral)",
        g,
        rhs,
        np.r_[psi0.real, psi0.imag],
        None,
        ex,
        4.0,
        "complex field split into real and imaginary parts",
        {"v": v, "mass": g.integrate(np.abs(psi0) ** 2)},
    )
    p.error = lambda sol: float(np.max(np.abs((sol.y[-1][:n] + 1j * sol.y[-1][n:]) - ex(x, sol.t[-1]))))
    return p


DISPERSIVE = [kdv_soliton, kdv_two_solitons, nls_soliton]


# ================================================================== coupled PDE-ODE and DAE problems
def packed_bed_adsorber(
    n: int = 100, u: float = 1.0, D: float = 1e-3, K: float = 5.0, k_ldf: float = 5.0, eps: float = 0.4, L: float = 1.0
) -> Problem:
    """Fixed-bed adsorption with a linear isotherm q* = K c and linear-driving-force uptake: fluid
    c_t = -u c_x + D c_xx - ((1 - eps)/eps) q_t, solid q_t = k (K c - q) at every position (a PDE coupled to an
    ODE at each point). Feed c = 1 from t = 0; Danckwerts inlet, zero gradient outlet. The breakthrough time of
    the front is the stoichiometric time L (1 + (1 - eps) K / eps) / u; total mass is conserved."""
    g = grids.uniform(0, L, n)
    ratio = (1 - eps) / eps
    left, right = bc.Robin(1.0, -D / u, 1.0), bc.Neumann(0.0)

    def rhs(t, y):
        c, q = y[: n + 1], y[n + 1 :]
        dq = k_ldf * (K * c - q)
        dc = op.advection(c, g, u, left, right, t, "upwind") + op.diffusion(c, g, D, left, right, t) - ratio * dq
        return np.r_[dc, dq]

    t_st = L * (1 + ratio * K) / u
    p = Problem(
        "packed-bed adsorber (linear isotherm, LDF)",
        g,
        rhs,
        np.zeros(2 * (n + 1)),
        (n + 2, n + 2),
        None,
        1.5 * t_st,
        "state [c, q]; the q rows couple to c at the same index: band n + 2",
        {"t_stoich": t_st, "ratio": ratio, "K": K, "u": u, "L": L},
    )
    return p


def catalyst_pellet(n: int = 60, phi: float = 3.0) -> Problem:
    """Spherical catalyst pellet with a first-order reaction (dimensionless): c_t = lap_r c - phi^2 c, c(1) = 1,
    symmetric centre. Steady state sinh(phi r)/(r sinh phi) and effectiveness factor
    eta = 3 (phi coth phi - 1)/phi^2 (Thiele)."""
    g = grids.radial(1.0, n, "spherical")

    def steady(x, t=None):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(x > 0, np.sinh(phi * x) / (np.maximum(x, 1e-300) * np.sinh(phi)), phi / np.sinh(phi))

    eta = 3 * (phi / np.tanh(phi) - 1) / phi**2
    y0 = np.zeros(n + 1)
    y0[-1] = 1.0
    return Problem(
        "catalyst pellet (Thiele modulus)",
        g,
        lambda t, c: op.diffusion(c, g, 1.0, None, bc.Dirichlet(1.0), t, source=lambda x, c, t: -(phi**2) * c),
        y0,
        (1, 1),
        steady,
        2.0,
        "effectiveness factor from the surface flux",
        {"phi": phi, "eta": eta},
    )


def counterflow_exchanger(n: int = 100, NTU: float = 2.0, Cr: float = 0.5) -> Problem:
    """Counterflow heat exchanger (dimensionless): hot stream T_h,t = -T_h,x - NTU (T_h - T_c) flowing right, cold
    stream T_c,t = +T_c,x - NTU Cr (T_h - T_c) flowing left; inlets T_h(0) = 1, T_c(1) = 0. Two coupled advection
    equations with opposite upwind directions; exact steady effectiveness
    eps = (1 - exp(-NTU (1 - Cr)))/(1 - Cr exp(-NTU (1 - Cr)))."""
    g = grids.uniform(0, 1, n)

    def rhs(t, y):
        Th, Tc = y[: n + 1], y[n + 1 :]
        ex_ = NTU * (Th - Tc)
        dTh = op.advection(Th, g, 1.0, bc.Dirichlet(1.0), bc.Neumann(0.0), t, "upwind") - ex_
        dTc = op.advection(Tc, g, -1.0, bc.Neumann(0.0), bc.Dirichlet(0.0), t, "upwind") + Cr * ex_
        dTh[0] = dTc[-1] = 0.0  # inlet temperatures are imposed: sources added after an operator must not move them
        return np.r_[dTh, dTc]

    eff = (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr)))
    y0 = np.zeros(2 * (n + 1))
    y0[0] = 1.0
    return Problem(
        "counterflow heat exchanger",
        g,
        rhs,
        y0,
        (n + 2, n + 2),
        None,
        6.0,
        "steady effectiveness = 1 - T_h(1) (exact NTU relation)",
        {"NTU": NTU, "Cr": Cr, "effectiveness": eff},
    )


def heat_as_dae(n: int = 40) -> Problem:
    """The heat equation with the Dirichlet condition written as an algebraic equation 0 = u_0 - g(t) instead of a
    ghost point: F(t, y, y') = [y'_0 - ... replaced by y_0 - g(t), y'_i - (u_xx)_i, y_n - 0]. Solved with the
    DAE-BDF; index 1. g(t) = sin(pi t) at the left, 0 at the right; reference: the ghost-point formulation."""
    g = grids.uniform(0, 1, n)
    dx = g.dx
    gl = lambda t: np.sin(np.pi * t)  # noqa: E731

    def F(t, y, yp):
        res = np.empty_like(y)
        res[0] = y[0] - gl(t)
        res[-1] = y[-1]
        res[1:-1] = yp[1:-1] - (y[2:] - 2 * y[1:-1] + y[:-2]) / dx**2
        return res

    ref_rhs = lambda t, u: op.diffusion(u, g, 1.0, bc.Dirichlet(gl), bc.Dirichlet(0.0), t)  # noqa: E731
    p = Problem(
        "heat equation as a DAE (boundary condition as an algebraic equation)",
        g,
        ref_rhs,
        np.zeros(n + 1),
        (1, 1),
        None,
        1.0,
        "params['F'] is the residual for dae_bdf",
        {"F": F, "band": (1, 1)},
    )
    return p


def string_moving_load(n: int = 200, c: float = 1.0, v: float = 0.5, F0: float = 1.0, width: float = 0.02) -> Problem:
    """String with a load F0 moving at speed v: w_tt = c^2 w_xx + F0 delta(x - v t) (delta as a narrow Gaussian),
    fixed ends. The energy of the string equals the work done by the load, integral F0 w_t(vt) dt, at all times."""
    g = grids.uniform(0, 1, n)
    x = g.x
    load = lambda t: F0 * np.exp(-(((x - v * t) / width) ** 2)) / (width * np.sqrt(np.pi))  # noqa: E731

    def rhs(t, y):
        w, wt = y[: n + 1], y[n + 1 :]
        a = op.diffusion(w, g, c**2, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t) + load(t)
        a[0] = a[-1] = 0.0
        return np.r_[wt, a]

    return Problem(
        "string under a moving load",
        g,
        rhs,
        np.zeros(2 * (n + 1)),
        None,
        None,
        1.0 / v,
        "energy = work done by the load",
        {"load": load, "c": c, "v": v},
    )


COUPLED = [packed_bed_adsorber, catalyst_pellet, counterflow_exchanger, heat_as_dae, string_moving_load]


# ================================================================== two-dimensional problems
def plate_heat_2d(n: int = 24, kappa: float = 1.0) -> Problem:
    """u_t = kappa (u_xx + u_yy) on the unit square, u = 0 on the boundary: sin(pi x) sin(2 pi y) decays as
    exp(-5 kappa pi^2 t). State: the (n+1)^2 field flattened; Jacobian band n + 1 (Kronecker structure)."""
    g2 = grids.tensor(grids.uniform(0, 1, n), grids.uniform(0, 1, n))
    bcs = {k: bc.Dirichlet(0.0) for k in ("left", "right", "bottom", "top")}
    shape = g2.shape
    ex = lambda X, Y, t: np.sin(np.pi * X) * np.sin(2 * np.pi * Y) * np.exp(-5 * kappa * np.pi**2 * t)  # noqa: E731
    p = Problem(
        "heat in a plate (2-D)",
        g2,
        lambda t, y: op.laplacian2d(y.reshape(shape), g2, bcs, t, kappa).ravel(),
        ex(g2.X, g2.Y, 0.0).ravel(),
        (n + 1, n + 1),
        None,
        0.05,
        "band n + 1: the banded solve costs n^2 (n+1)^2, far below the dense (n+1)^6",
        {"shape": shape, "exact2d": ex},
    )
    p.error = lambda sol: float(np.max(np.abs(sol.y[-1].reshape(shape) - ex(g2.X, g2.Y, sol.t[-1]))))
    return p


def fisher_2d(n: int = 80, r: float = 1.0, D: float = 1.0, L: float = 40.0) -> Problem:
    """Fisher-KPP in 2-D: a circular front expanding from a central seed at the asymptotic speed 2 sqrt(rD)
    (measured from the front radius growth); no-flux boundaries."""
    g2 = grids.tensor(grids.uniform(0, L, n), grids.uniform(0, L, n))
    bcs = {k: bc.Neumann(0.0) for k in ("left", "right", "bottom", "top")}
    shape = g2.shape
    R = np.hypot(g2.X - L / 2, g2.Y - L / 2)

    def rhs(t, y):
        U = y.reshape(shape)
        return (op.laplacian2d(U, g2, bcs, t, D) + r * U * (1 - U)).ravel()

    p = Problem(
        "Fisher-KPP front in 2-D",
        g2,
        rhs,
        np.where(R < 3.0, 1.0, 0.0).ravel(),
        (n + 1, n + 1),
        None,
        8.0,
        "radius grows at 2 sqrt(rD) minus a curvature correction",
        {"shape": shape, "R": R, "speed": 2 * np.sqrt(r * D)},
    )
    return p


def cavity_flow(n: int = 32, Re: float = 100.0) -> Problem:
    """Lid-driven cavity in vorticity-streamfunction form: omega_t = -(u omega_x + v omega_y) + (1/Re) lap omega,
    lap psi = -omega, u = psi_y, v = -psi_x, lid velocity 1. The streamfunction Poisson problem is solved at
    every evaluation with a factorised operator; wall vorticity from Thom's formula. Reference: Ghia et al. 1982
    centreline velocities at Re = 100 (u_min = -0.2109 on the vertical centreline)."""
    from . import stencils as st_

    g = grids.uniform(0, 1, n)
    dx = g.dx
    m = n - 1  # interior points per direction
    D2 = st_.diff_operator(g.x, 2, 2).to_dense()[1:-1, 1:-1]  # interior Laplacian (Dirichlet psi = 0)
    Lap = np.kron(D2, np.eye(m)) + np.kron(np.eye(m), D2)
    lu = np.linalg.inv(Lap)  # small dense inverse: applied as a matrix product each evaluation

    def streamfunction(omega):
        psi = np.zeros((n + 1, n + 1))
        psi[1:-1, 1:-1] = (lu @ (-omega[1:-1, 1:-1].ravel())).reshape(m, m)
        return psi

    def velocity(psi):
        u = np.zeros_like(psi)
        v = np.zeros_like(psi)
        u[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * dx)
        v[1:-1, :] = -(psi[2:, :] - psi[:-2, :]) / (2 * dx)
        u[:, -1] = 1.0  # the lid
        return u, v

    def rhs(t, y):
        omega = y.reshape(n + 1, n + 1)
        psi = streamfunction(omega)
        u, v = velocity(psi)
        w = omega.copy()  # wall vorticity by Thom's formula
        w[0, :] = -2 * psi[1, :] / dx**2
        w[-1, :] = -2 * psi[-2, :] / dx**2
        w[:, 0] = -2 * psi[:, 1] / dx**2
        w[:, -1] = -2 * psi[:, -2] / dx**2 - 2 / dx  # moving lid
        adv = np.zeros_like(w)
        adv[1:-1, 1:-1] = u[1:-1, 1:-1] * (w[2:, 1:-1] - w[:-2, 1:-1]) / (2 * dx) + v[1:-1, 1:-1] * (
            w[1:-1, 2:] - w[1:-1, :-2]
        ) / (2 * dx)
        lap = np.zeros_like(w)
        lap[1:-1, 1:-1] = (w[2:, 1:-1] + w[:-2, 1:-1] + w[1:-1, 2:] + w[1:-1, :-2] - 4 * w[1:-1, 1:-1]) / dx**2
        out = -adv + lap / Re
        out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = (
            0.0  # boundary vorticity is algebraic (Thom): recomputed, not integrated
        )
        return out.ravel()

    p = Problem(
        "lid-driven cavity (vorticity-streamfunction)",
        g,
        rhs,
        np.zeros((n + 1) ** 2),
        None,
        None,
        30.0,
        "explicit integration: the cavity is not stiff at Re = 100 on a 32 x 32 grid; steady by t = 30",
        {"streamfunction": streamfunction, "velocity": velocity, "Re": Re, "n": n, "ghia_umin": -0.2109},
    )
    return p


TWO_D = [plate_heat_2d, fisher_2d, cavity_flow]


# ================================================================== the project: a cooled tubular reactor
def tubular_reactor(
    n: int = 100,
    T_in: float = 320.0,
    k0: float = 5e7,
    E: float = 60e3,
    u: float = 0.2,
    L: float = 2.0,
    D: float = 1e-3,
    alpha: float = 1e-3,
    B: float = 120.0,
    hw: float = 0.1,
    T_w: float | None = None,
    c_in: float = 1.0,
) -> Problem:
    """Wall-cooled tubular reactor with a first-order exothermic reaction (plug flow with axial dispersion):
    c_t = -u c_x + D c_xx - k(T) c,  T_t = -u T_x + alpha T_xx + B k(T) c / c_in - hw (T - T_w),
    k(T) = k0 exp(-E/(R T)); Danckwerts inlets, zero-gradient outlets; the wall is at T_w (default: the inlet
    temperature). B is the adiabatic temperature rise. State [c, T]. Ignition (runaway) occurs between
    T_in = 330 and 340 K with the default parameters: the project fits k0 and E from sub-critical profiles and
    locates that boundary by parametric sensitivity."""
    R = 8.314
    T_w = T_in if T_w is None else T_w
    g = grids.uniform(0, L, n)
    left_c, left_T = bc.Robin(1.0, -D / u, c_in), bc.Robin(1.0, -alpha / u, T_in)
    right = bc.Neumann(0.0)

    def rhs(t, y):
        c, T = y[: n + 1], y[n + 1 :]
        k = k0 * np.exp(
            -E / (R * np.clip(T, 200.0, 2000.0))
        )  # the rate law evaluated in a physical range (Newton iterates may stray)
        dc = op.advection(c, g, u, left_c, right, t, "upwind") + op.diffusion(c, g, D, left_c, right, t) - k * c
        dT = (
            op.advection(T, g, u, left_T, right, t, "upwind")
            + op.diffusion(T, g, alpha, left_T, right, t)
            + B * k * c / c_in
            - hw * (T - T_w)
        )
        return np.r_[dc, dT]

    y0 = np.r_[np.zeros(n + 1), np.full(n + 1, T_w)]
    return Problem(
        "cooled tubular reactor",
        g,
        rhs,
        y0,
        (n + 2, n + 2),
        None,
        80.0,
        "state [c, T]; steady by t = 80 s (residence time 10 s)",
        {"k0": k0, "E": E, "T_in": T_in, "B": B, "hw": hw, "T_w": T_w, "u": u},
    )


PROJECT = [tubular_reactor]
ALL = PARABOLIC + HYPERBOLIC + DISPERSIVE + COUPLED + TWO_D + PROJECT
