"""Time integrators for semi-discrete systems y' = f(t, y): explicit Runge-Kutta (fixed step and the adaptive
Dormand-Prince 5(4) pair), implicit methods with Newton iterations on a banded Jacobian (backward Euler, the
trapezoidal rule, BDF of orders 1-5 with variable order start, Radau IIA), IMEX splitting for stiff diffusion
with explicit reaction or advection, exponential integration for linear diffusion, Stormer-Verlet for second-order
wave equations, and BDF for index-1 DAEs with a singular mass matrix.

>>> import numpy as np
>>> from engmolines import integrators as ti
>>> s = ti.bdf(lambda t, y: -1000 * (y - np.cos(t)) - np.sin(t), [1.0], (0.0, 1.0), 0.05, order=2)
>>> bool(abs(s.y[-1, 0] - np.cos(1.0)) < 1e-4)
True
"""

from __future__ import annotations

import numpy as np

from . import linalg as la
from ._common import Solution

# ------------------------------------------------------------------ explicit Runge-Kutta
RK4 = (
    np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0.0]]),
    np.array([1, 2, 2, 1.0]) / 6,
    np.array([0, 0.5, 0.5, 1.0]),
)
HEUN = (np.array([[0, 0], [1.0, 0]]), np.array([0.5, 0.5]), np.array([0, 1.0]))
EULER = (np.array([[0.0]]), np.array([1.0]), np.array([0.0]))


def _outputs(t_span, n_out):
    return np.linspace(t_span[0], t_span[1], n_out + 1) if n_out else None


def explicit_rk(f, y0, t_span, h: float, tableau=RK4, n_out: int = 10) -> Solution:
    """Fixed-step explicit Runge-Kutta with the given tableau (RK4 by default); stores n_out + 1 snapshots."""
    A, b, c = tableau
    s = len(b)
    t0, t1 = t_span
    if not np.isfinite(h) or h <= 0 or (t1 - t0) / h > 1e8:
        raise ValueError(f"explicit_rk: step {h!r} would need {(t1 - t0) / h:.1e} steps")
    n = int(round((t1 - t0) / h))
    h = (t1 - t0) / n
    y = np.array(y0, float)
    keep = np.unique(np.linspace(0, n, n_out + 1).astype(int)) if n_out else np.array([n])
    ts, ys = [t0] if 0 in keep else [], [y.copy()] if 0 in keep else []
    K = np.zeros((s, len(y)))
    for k in range(n):
        t = t0 + k * h
        for i in range(s):
            K[i] = f(t + c[i] * h, y + h * (A[i, :i] @ K[:i]))
        y = y + h * (b @ K)
        if k + 1 in keep:
            ts.append(t0 + (k + 1) * h)
            ys.append(y.copy())
    return Solution(np.array(ts), np.array(ys), steps=n, evaluations=n * s)


_DP_C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
_DP_A = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ]
)
_DP_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
_DP_B4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])


def dopri45(
    f, y0, t_span, rtol: float = 1e-6, atol: float = 1e-9, n_out: int = 10, max_steps: int = 500000, h0=None
) -> Solution:
    """Adaptive Dormand-Prince 5(4) with PI step control; output at n_out + 1 equally spaced times by cubic
    Hermite interpolation. The step-size history is returned in extra['h']."""
    t0, t1 = t_span
    y = np.array(y0, float)
    t = t0
    fy = np.asarray(f(t, y), float)
    evals = 1
    sc = atol + rtol * np.abs(y)
    if h0 is None:
        d0, d1 = np.sqrt(np.mean((y / sc) ** 2)), np.sqrt(np.mean((fy / sc) ** 2))
        h = 0.01 * d0 / d1 if d0 > 1e-5 and d1 > 1e-5 else 1e-6
        f1 = np.asarray(f(t + h, y + h * fy), float)
        evals += 1
        d2 = np.sqrt(np.mean(((f1 - fy) / sc) ** 2)) / h
        h = min(100 * h, (0.01 / max(d1, d2)) ** 0.2 if max(d1, d2) > 1e-15 else max(1e-6, h * 1e-3))
    else:
        h = h0
    out_t = _outputs(t_span, n_out)
    ts, ys = [t0], [y.copy()]
    next_out = 1
    accepted = rejected = 0
    err_prev = 1e-4
    hs = []
    while t < t1 and accepted + rejected < max_steps:
        h = min(h, t1 - t)
        K = np.zeros((7, len(y)))
        K[0] = fy
        for i in range(1, 7):
            K[i] = f(t + _DP_C[i] * h, y + h * (_DP_A[i, :i] @ K[:i]))
        evals += 6
        y5 = y + h * (_DP_B5 @ K)
        sc = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err = float(np.sqrt(np.mean(((h * ((_DP_B5 - _DP_B4) @ K)) / sc) ** 2)))
        if err <= 1.0:
            t_new, f_new = t + h, K[6]
            while out_t is not None and next_out < len(out_t) and out_t[next_out] <= t_new + 1e-12 * max(1, abs(t_new)):
                s_ = (out_t[next_out] - t) / h
                yi = (
                    (2 * s_**3 - 3 * s_**2 + 1) * y
                    + (s_**3 - 2 * s_**2 + s_) * h * fy
                    + (-2 * s_**3 + 3 * s_**2) * y5
                    + (s_**3 - s_**2) * h * f_new
                )
                ts.append(out_t[next_out])
                ys.append(yi)
                next_out += 1
            hs.append(h)
            t, y, fy = t_new, y5, f_new
            accepted += 1
            fac = 0.9 * err ** (-0.7 / 5) * err_prev ** (0.4 / 5) if err > 0 else 5.0
            h *= min(5.0, max(0.2, fac))
            err_prev = max(err, 1e-4)
        else:
            rejected += 1
            h *= max(0.2, 0.9 * err ** (-0.2))
    if out_t is None:
        ts, ys = [t0, t], [np.array(y0, float), y]
    elif t < t1:  # stopped early: append the last state reached so the caller can see where it got to
        ts.append(t)
        ys.append(y.copy())
    msg = "ok" if t >= t1 else "maximum steps reached"
    return Solution(np.array(ts), np.array(ys), accepted, rejected, evals, message=msg, extra={"h": np.array(hs)})


# ------------------------------------------------------------------ implicit methods with banded Newton
class _Newton:
    """Newton iterations for G(y) = y - h*beta*f(t, y) - rhs = 0 with a banded (or dense) Jacobian of f, refreshed
    only when convergence slows (a modified Newton: one factorisation serves many steps)."""

    def __init__(self, f, band=None, jac=None, tol=1e-9, maxiter=8):
        self.f, self.band, self.jac, self.tol, self.maxiter = f, band, jac, tol, maxiter
        self.fact = None
        self.h_beta = None
        self.jacobians = 0
        self.iterations = 0

    def refresh(self, t, y, h_beta):
        n = len(y)
        if self.jac is not None:
            J = self.jac(t, y)
            if self.band is not None and np.ndim(J) == 2 and J.shape[0] == n:
                J = la.to_banded(J, *self.band)
        elif self.band is not None:
            J = la.banded_jacobian(lambda u: self.f(t, u), y, *self.band)
        else:
            J = la.dense_jacobian(lambda u: self.f(t, u), y)
        self.jacobians += 1
        if self.band is not None:
            lo, up = self.band
            M = -h_beta * J
            M[up, :] += 1.0
            self.fact = ("banded", la.lu_banded(M, lo, up))
        else:
            self.fact = ("dense", np.eye(n) - h_beta * J)
        self.h_beta = h_beta

    def _solve(self, r):
        kind, F = self.fact
        return la.lu_banded_solve(F[0], F[1], r) if kind == "banded" else np.linalg.solve(F, r)

    def solve(self, t, y_guess, h_beta, rhs):
        """Solve y - h_beta f(t, y) = rhs from y_guess. Refreshes the Jacobian if none, if h_beta changed by
        more than 20 %, or if the iteration stalls."""
        if self.fact is None or abs(h_beta - self.h_beta) > 0.2 * abs(self.h_beta):
            self.refresh(t, y_guess, h_beta)
        y = y_guess.copy()
        for _attempt in range(5):
            last = None
            for _ in range(self.maxiter):
                G = y - h_beta * np.asarray(self.f(t, y), float) - rhs
                dy = self._solve(-G)
                y += dy
                self.iterations += 1
                nd = np.max(np.abs(dy)) / max(1.0, np.max(np.abs(y)))
                if not np.isfinite(nd):
                    return y_guess.copy(), False
                if nd <= self.tol:
                    return y, True
                if (
                    last is not None and nd > 0.5 * last
                ):  # linear convergence slower than 0.5: the frozen Jacobian is stale
                    break
                last = nd
            self.refresh(t, y, h_beta)
        return y, False


_BDF = {
    1: ([1.0], 1.0),
    2: ([4 / 3, -1 / 3], 2 / 3),
    3: ([18 / 11, -9 / 11, 2 / 11], 6 / 11),
    4: ([48 / 25, -36 / 25, 16 / 25, -3 / 25], 12 / 25),
    5: ([300 / 137, -300 / 137, 200 / 137, -75 / 137, 12 / 137], 60 / 137),
}

_R5A = np.array(
    [
        [(88 - 7 * np.sqrt(6)) / 360, (296 - 169 * np.sqrt(6)) / 1800, (-2 + 3 * np.sqrt(6)) / 225],
        [(296 + 169 * np.sqrt(6)) / 1800, (88 + 7 * np.sqrt(6)) / 360, (-2 - 3 * np.sqrt(6)) / 225],
        [(16 - np.sqrt(6)) / 36, (16 + np.sqrt(6)) / 36, 1 / 9],
    ]
)
_R5C = np.array([(4 - np.sqrt(6)) / 10, (4 + np.sqrt(6)) / 10, 1.0])


def _radau_step(f, t, y, h, band, jac):
    """One Radau IIA (order 5) step by full Newton on the stage equations, with a dense stage Jacobian built from
    the (banded or dense) Jacobian of f. For large systems (n > 400) the dense 3n x 3n stage system is too
    expensive, and the step is taken as eight backward-Euler substeps with the banded modified Newton instead."""
    n = len(y)
    if n > 400:
        return _euler_substeps(f, t, y, h, band, jac)
    if jac is not None:
        J = jac(t, y)
        if np.ndim(J) == 2 and J.shape[0] != n:
            J = la.from_banded(J, *band)
    elif band is not None:
        J = la.from_banded(la.banded_jacobian(lambda u: f(t, u), y, *band), *band)
    else:
        J = la.dense_jacobian(lambda u: f(t, u), y)
    M = np.eye(3 * n) - h * np.kron(_R5A, J)
    K = np.tile(np.asarray(f(t, y), float), (3, 1))
    its = 0
    with np.errstate(all="ignore"):
        for _ in range(50):
            Y = y + h * _R5A @ K
            G = K - np.array([f(t + _R5C[i] * h, Y[i]) for i in range(3)])
            dK = np.linalg.solve(M, -G.ravel()).reshape(3, n)
            K += dK
            its += 1
            if not np.all(np.isfinite(K)):
                break
            if np.max(np.abs(dK)) <= 1e-12 * max(1.0, np.max(np.abs(K))):
                return y + h * (_R5A[-1] @ K), its
    # Radau's undamped Newton failed (rough initial data, strongly nonlinear coefficients): fall back to
    # eight backward-Euler substeps, which are L-stable and use the safeguarded modified Newton
    yk, its2 = _euler_substeps(f, t, y, h, band, jac)
    return yk, its + its2


def _euler_substeps(f, t, y, h, band, jac, m: int = 8):
    """Backward-Euler substeps from t to t + h; a substep whose Newton iteration fails is halved (down to h/2^16)."""
    newton = _Newton(f, band, jac)
    yk, tk, hk, t_end = y.copy(), t, h / m, t + h
    while tk < t_end - 1e-14 * max(1.0, abs(t_end)):
        hk = min(hk, t_end - tk)
        y_new, ok = newton.solve(tk + hk, yk, hk, yk)
        if ok:
            yk, tk = y_new, tk + hk
            hk = min(2 * hk, h / m)
        else:
            hk /= 2
            newton.fact = None
            if hk < h / 2**16:
                raise RuntimeError(f"starting step failed at t = {tk + hk:g}")
    return yk, newton.iterations


def bdf(
    f, y0, t_span, h: float, order: int = 2, band=None, jac=None, n_out: int = 10, newton_tol: float = 1e-9
) -> Solution:
    """Fixed-step BDF of the given order (1-5), started with Radau IIA steps; each step a modified-Newton solve with
    a banded Jacobian (band = (lo, up) from the operator's stencil) or a dense one. The Jacobian is reused across
    steps until Newton slows, so the cost per step is usually one banded solve."""
    t0, t1 = t_span
    n = int(round((t1 - t0) / h))
    h = (t1 - t0) / n
    y = np.array(y0, float)
    ys, ts = [y.copy()], [t0]
    keep = set(np.unique(np.linspace(0, n, n_out + 1).astype(int)))
    out_t, out_y = [t0], [y.copy()]
    newton = _Newton(f, band, jac, newton_tol)
    radau_its = 0
    for k in range(n):
        t = t0 + (k + 1) * h
        if k + 1 < order:
            y, its = _radau_step(f, ts[-1], ys[-1], h, band, jac)
            radau_its += its
        else:
            a, beta = _BDF[order]
            rhs = sum(aj * ys[-1 - j] for j, aj in enumerate(a))
            guess = ys[-1] + (ys[-1] - ys[-2]) if len(ys) > 1 else ys[-1]
            y, ok = newton.solve(t, guess, h * beta, rhs)
            if not ok:
                return Solution(
                    np.array(out_t),
                    np.array(out_y),
                    k,
                    0,
                    0,
                    newton.iterations,
                    newton.jacobians,
                    message=f"Newton failed at t = {t:g}",
                )
        ys.append(y.copy())
        ts.append(t)
        if len(ys) > order:
            ys.pop(0)
            ts.pop(0)
        if k + 1 in keep:
            out_t.append(t)
            out_y.append(y.copy())
    return Solution(np.array(out_t), np.array(out_y), n, 0, 0, newton.iterations + radau_its, newton.jacobians)


def trapezoidal(f, y0, t_span, h: float, band=None, jac=None, n_out: int = 10) -> Solution:
    """Implicit trapezoidal rule (Crank-Nicolson in time): A-stable, second order, not L-stable."""
    t0, t1 = t_span
    n = int(round((t1 - t0) / h))
    h = (t1 - t0) / n
    y = np.array(y0, float)
    keep = set(np.unique(np.linspace(0, n, n_out + 1).astype(int)))
    out_t, out_y = [t0], [y.copy()]
    newton = _Newton(f, band, jac)
    for k in range(n):
        t = t0 + k * h
        fy = np.asarray(f(t, y), float)
        y, ok = newton.solve(t + h, y + h * fy, 0.5 * h, y + 0.5 * h * fy)
        if not ok:
            return Solution(
                np.array(out_t),
                np.array(out_y),
                k,
                0,
                0,
                newton.iterations,
                newton.jacobians,
                message=f"Newton failed at t = {t + h:g}",
            )
        if k + 1 in keep:
            out_t.append(t + h)
            out_y.append(y.copy())
    return Solution(np.array(out_t), np.array(out_y), n, 0, 0, newton.iterations, newton.jacobians)


def imex(
    f_explicit, f_implicit, y0, t_span, h: float, band=None, jac=None, n_out: int = 10, order: int = 2
) -> Solution:
    """Implicit-explicit splitting y' = f_explicit + f_implicit: order 1 (forward/backward Euler) or order 2
    (second-order Adams-Bashforth for the explicit part, Crank-Nicolson for the implicit part: CNAB2). The
    implicit part is typically linear diffusion, solved by Newton with a fixed banded Jacobian; the explicit part
    (reaction, advection) sets the stability limit instead of the diffusion."""
    t0, t1 = t_span
    n = int(round((t1 - t0) / h))
    h = (t1 - t0) / n
    y = np.array(y0, float)
    keep = set(np.unique(np.linspace(0, n, n_out + 1).astype(int)))
    out_t, out_y = [t0], [y.copy()]
    newton = _Newton(f_implicit, band, jac)
    fe_prev = None
    for k in range(n):
        t = t0 + k * h
        fe = np.asarray(f_explicit(t, y), float)
        if order == 1 or fe_prev is None:
            rhs = y + h * fe
            y, ok = newton.solve(t + h, y + h * fe, h, rhs)  # backward Euler on the implicit part
        else:
            fi = np.asarray(f_implicit(t, y), float)
            rhs = y + h * (1.5 * fe - 0.5 * fe_prev) + 0.5 * h * fi
            y, ok = newton.solve(t + h, y + h * fe, 0.5 * h, rhs)
        if not ok:
            return Solution(
                np.array(out_t),
                np.array(out_y),
                k,
                0,
                0,
                newton.iterations,
                newton.jacobians,
                message=f"Newton failed at t = {t + h:g}",
            )
        fe_prev = fe
        if k + 1 in keep:
            out_t.append(t + h)
            out_y.append(y.copy())
    return Solution(np.array(out_t), np.array(out_y), n, 0, 0, newton.iterations, newton.jacobians)


def exponential(A, y0, t_span, n_out: int = 10, forcing=None) -> Solution:
    """Exact-in-time solution of the linear system y' = A y (+ b, constant) by the eigendecomposition of the dense
    operator A: y(t) = V exp(Lambda t) V^-1 (y0 + A^-1 b) - A^-1 b. No time-step error at all - the reference
    against which time integrators are measured on linear problems."""
    A = np.asarray(A, float)
    lam, V = np.linalg.eig(A)
    Vinv = np.linalg.inv(V)
    y0 = np.array(y0, float)
    shift = np.zeros_like(y0) if forcing is None else np.linalg.solve(A, np.asarray(forcing, float))
    ts = np.linspace(t_span[0], t_span[1], n_out + 1)
    ys = np.array([(V @ (np.exp(lam * (t - ts[0])) * (Vinv @ (y0 + shift)))).real - shift for t in ts])
    return Solution(ts, ys, n_out, 0, 0)


def verlet(accel, u0, v0, t_span, h: float, n_out: int = 10) -> Solution:
    """Stormer-Verlet (velocity Verlet) for u'' = accel(t, u): second order, symplectic, energy-preserving over long
    times; stable for h * omega_max <= 2. Returns the displacements in y[:, :n] and velocities in y[:, n:]."""
    t0, t1 = t_span
    n = int(round((t1 - t0) / h))
    h = (t1 - t0) / n
    u, v = np.array(u0, float), np.array(v0, float)
    a = np.asarray(accel(t0, u), float)
    keep = set(np.unique(np.linspace(0, n, n_out + 1).astype(int)))
    out_t, out_y = [t0], [np.r_[u, v]]
    for k in range(n):
        t = t0 + k * h
        v_half = v + 0.5 * h * a
        u = u + h * v_half
        a = np.asarray(accel(t + h, u), float)
        v = v_half + 0.5 * h * a
        if k + 1 in keep:
            out_t.append(t + h)
            out_y.append(np.r_[u, v])
    return Solution(np.array(out_t), np.array(out_y), n, 0, n + 1)


def dae_bdf(F, y0, t_span, h: float, order: int = 2, band=None, n_out: int = 10, newton_tol: float = 1e-9) -> Solution:
    """Fully implicit BDF for F(t, y, y') = 0 (index-1 DAEs: algebraic rows are simply F = 0 with no y'). Each step
    solves F(t, y, (y - sum a_j y_{n-j}) / (h beta)) = 0 by Newton with a finite-difference Jacobian (banded if band
    is given). Started with BDF1."""
    t0, t1 = t_span
    n = int(round((t1 - t0) / h))
    h = (t1 - t0) / n
    y = np.array(y0, float)
    hist = [y.copy()]
    keep = set(np.unique(np.linspace(0, n, n_out + 1).astype(int)))
    out_t, out_y = [t0], [y.copy()]
    its = jacs = 0
    for k in range(n):
        t = t0 + (k + 1) * h
        q = min(order, k + 1)
        a, beta = _BDF[q]
        past = sum(aj * hist[-1 - j] for j, aj in enumerate(a))
        G = lambda z, t=t, past=past, beta=beta: np.asarray(F(t, z, (z - past) / (h * beta)), float)  # noqa: E731
        z = hist[-1].copy()
        J = la.banded_jacobian(G, z, *band) if band else la.dense_jacobian(G, z)
        jacs += 1
        fact = la.lu_banded(J, *band) if band else None
        ok = False
        for _ in range(12):
            g = G(z)
            dz = la.lu_banded_solve(fact[0], fact[1], -g) if band else np.linalg.solve(J, -g)
            z += dz
            its += 1
            if np.max(np.abs(dz)) <= newton_tol * max(1.0, np.max(np.abs(z))):
                ok = True
                break
        if not ok:
            return Solution(np.array(out_t), np.array(out_y), k, 0, 0, its, jacs, message=f"Newton failed at t = {t:g}")
        hist.append(z.copy())
        if len(hist) > order:
            hist.pop(0)
        if k + 1 in keep:
            out_t.append(t)
            out_y.append(z.copy())
    return Solution(np.array(out_t), np.array(out_y), n, 0, 0, its, jacs)
