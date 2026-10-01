"""Finite-difference stencils: Fornberg's algorithm for the weights of any derivative on any points, differentiation
matrices with matched-order one-sided closures at the boundaries, stencil (diagonal-offset) operators that apply
in O(n) without forming dense matrices, and conversions to banded storage for the implicit solvers.

>>> import numpy as np
>>> from engmolines import stencils
>>> stencils.fd_weights(0.0, [-1.0, 0.0, 1.0], 2)          # the classic second-derivative stencil
array([ 1., -2.,  1.])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def fd_weights(x0: float, xs, m: int) -> np.ndarray:
    """Weights w with sum_j w_j f(x_j) approximating f^(m)(x0), exact for polynomials of degree len(xs) - 1
    (Fornberg 1988)."""
    xs = np.asarray(xs, float)
    n = len(xs)
    C = np.zeros((n, m + 1))
    C[0, 0] = 1.0
    c1, c4 = 1.0, xs[0] - x0
    for i in range(1, n):
        mn = min(i, m)
        c2, c5, c4 = 1.0, c4, xs[i] - x0
        for j in range(i):
            c3 = xs[i] - xs[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    C[i, k] = c1 * (k * C[i - 1, k - 1] - c5 * C[i - 1, k]) / c2
                C[i, 0] = -c1 * c5 * C[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                C[j, k] = (c4 * C[j, k] - k * C[j, k - 1]) / c3
            C[j, 0] = c4 * C[j, 0] / c3
        c1 = c2
    return C[:, m]


@dataclass
class StencilOperator:
    """A linear operator stored by diagonals: (A u)_i = sum_k coeffs[k][i] * u[i + offsets[k]], entries outside the
    range dropped (or wrapped when periodic). Applies in O(n * stencil width); converts to dense or banded form."""

    offsets: np.ndarray
    coeffs: np.ndarray  # shape (len(offsets), n)
    periodic: bool = False

    @property
    def n(self) -> int:
        return self.coeffs.shape[1]

    @property
    def bandwidth(self) -> tuple:
        return int(-min(self.offsets.min(), 0)), int(max(self.offsets.max(), 0))

    def matvec(self, u) -> np.ndarray:
        u = np.asarray(u)
        n = self.n
        out = np.zeros_like(u, dtype=np.result_type(u, self.coeffs))
        shape = (n,) + (1,) * (u.ndim - 1)  # coefficients broadcast along the leading axis of 2-D fields
        for off, c in zip(self.offsets, self.coeffs):
            c = c.reshape(shape)
            if self.periodic:
                out += c * np.roll(u, -off, axis=0)
            elif off >= 0:
                out[: n - off] += c[: n - off] * u[off:]
            else:
                out[-off:] += c[-off:] * u[: n + off]
        return out

    __call__ = matvec

    def to_dense(self) -> np.ndarray:
        n = self.n
        A = np.zeros((n, n))
        for off, c in zip(self.offsets, self.coeffs):
            for i in range(n):
                j = i + off
                if self.periodic:
                    A[i, j % n] += c[i]
                elif 0 <= j < n:
                    A[i, j] += c[i]
        return A

    def to_banded(self) -> tuple:
        """LAPACK-style banded storage ab[l + u + i - j, j] = A[i, j] and the bandwidths (l, u); periodic wrap-around
        entries do not fit a band and raise."""
        if self.periodic:
            raise ValueError("A periodic operator is not banded; use to_dense or the FFT.")
        lo, up = self.bandwidth
        n = self.n
        ab = np.zeros((lo + up + 1, n))
        for off, c in zip(self.offsets, self.coeffs):
            for i in range(n):
                j = i + off
                if 0 <= j < n:
                    ab[up + i - j, j] += c[i]
        return ab, lo, up

    def __add__(self, other):
        if isinstance(other, StencilOperator):
            offs = np.union1d(self.offsets, other.offsets)
            co = np.zeros((len(offs), self.n))
            for op in (self, other):
                for off, c in zip(op.offsets, op.coeffs):
                    co[np.searchsorted(offs, off)] += c
            return StencilOperator(offs, co, self.periodic)
        return NotImplemented

    def __mul__(self, s):
        return StencilOperator(self.offsets, s * self.coeffs, self.periodic)

    __rmul__ = __mul__


def identity(n: int) -> StencilOperator:
    """The identity as a stencil operator (offset 0, unit coefficients)."""
    return StencilOperator(np.array([0]), np.ones((1, n)))


def diff_operator(x, deriv: int, acc: int = 2, periodic: bool = False) -> StencilOperator:
    """Differentiation operator of order `deriv` and accuracy `acc` on the points x: centred stencils of
    deriv + acc - 1 (odd) points inside, one-sided stencils of the same accuracy at the boundaries (on a
    periodic grid the centred stencil wraps). Works on non-uniform grids via Fornberg weights at each point."""
    x = np.asarray(x, float)
    n = len(x)
    width = deriv + acc - 1 if (deriv + acc) % 2 == 0 else deriv + acc
    half = width // 2
    if periodic:
        dx = x[1] - x[0]
        w = fd_weights(0.0, np.arange(-half, half + 1) * dx, deriv)
        return StencilOperator(np.arange(-half, half + 1), np.tile(w[:, None], (1, n)), periodic=True)
    npts = deriv + acc  # one-sided stencils need one more point than centred ones for the same accuracy
    rows = {}
    for i in range(n):
        if i - half < 0:
            idx = np.arange(0, min(npts, n))
        elif i + half >= n:
            idx = np.arange(max(n - npts, 0), n)
        else:
            idx = np.arange(i - half, i + half + 1)
        w = fd_weights(x[i], x[idx], deriv)
        for j, wj in zip(idx, w):
            rows.setdefault(j - i, np.zeros(n))[i] += wj
    offs = np.array(sorted(rows))
    return StencilOperator(offs, np.array([rows[o] for o in offs]))


def upwind_operator(x, velocity_sign: int = 1, acc: int = 1) -> StencilOperator:
    """First-derivative operator biased against the flow: for velocity_sign > 0 it uses points upstream (to the
    left). acc 1 gives the classic upwind stencil, acc 2 the second-order (Beam-Warming-type) biased stencil."""
    x = np.asarray(x, float)
    n = len(x)
    rows = {}
    for i in range(n):
        lo = i - acc if velocity_sign > 0 else i
        hi = i if velocity_sign > 0 else i + acc
        lo, hi = max(lo, 0), min(hi, n - 1)
        if hi - lo < acc:  # near the boundary opposite to the upwind side: fall back to whatever points exist
            lo, hi = max(i - acc, 0), min(i + acc, n - 1)
        idx = np.arange(lo, hi + 1)
        w = fd_weights(x[i], x[idx], 1)
        for j, wj in zip(idx, w):
            rows.setdefault(j - i, np.zeros(n))[i] += wj
    offs = np.array(sorted(rows))
    return StencilOperator(offs, np.array([rows[o] for o in offs]))


def kron_sum(Ax: StencilOperator, Ay: StencilOperator):
    """The 2-D operator Ax (x) I + I (x) Ay acting on fields of shape (nx, ny), returned as a function on the
    flattened vector together with its bandwidth (nx * ny unknowns, bandwidth ny * half-width of Ax)."""
    nx, ny = Ax.n, Ay.n

    def apply(u):
        U = np.asarray(u).reshape(nx, ny)
        return (Ax.matvec(U) + Ay.matvec(U.T).T).ravel()

    lo = max(Ax.bandwidth[0] * ny, Ay.bandwidth[0])
    up = max(Ax.bandwidth[1] * ny, Ay.bandwidth[1])
    return apply, (lo, up)
