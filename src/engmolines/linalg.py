"""Linear algebra for large semi-discrete systems: banded LU with partial pivoting (O(n b^2) instead of O(n^3)),
banded Jacobians by finite differences with column colouring (b + 1 evaluations instead of n), and dense
fallbacks.

>>> import numpy as np
>>> from engmolines import linalg as la
>>> A = np.diag(2.0 * np.ones(5)) - np.diag(np.ones(4), 1) - np.diag(np.ones(4), -1)
>>> ab = la.to_banded(A, 1, 1)
>>> np.allclose(la.solve_banded(ab, 1, 1, np.ones(5)), np.linalg.solve(A, np.ones(5)))
True
"""

from __future__ import annotations

import numpy as np


def to_banded(A, lo: int, up: int) -> np.ndarray:
    """Banded storage ab[up + i - j, j] = A[i, j] for |i - j| within the bandwidths."""
    A = np.asarray(A, float)
    n = len(A)
    ab = np.zeros((lo + up + 1, n))
    for i in range(n):
        for j in range(max(0, i - lo), min(n, i + up + 1)):
            ab[up + i - j, j] = A[i, j]
    return ab


def from_banded(ab, lo: int, up: int) -> np.ndarray:
    """Dense matrix from banded storage with bandwidths (lo, up)."""
    n = ab.shape[1]
    A = np.zeros((n, n))
    for j in range(n):
        for i in range(max(0, j - up), min(n, j + lo + 1)):
            A[i, j] = ab[up + i - j, j]
    return A


def lu_banded(ab, lo: int, up: int):
    """LU factorisation of a banded matrix with partial pivoting. Pivoting widens the upper band to lo + up, so the
    factors are stored in an array with lo extra rows (LAPACK's gbtrf layout). Returns (lu, piv)."""
    n = ab.shape[1]
    U = up + lo  # upper bandwidth after pivoting
    a = np.zeros((2 * lo + up + 1, n))
    a[lo:, :] = ab
    piv = np.arange(n)
    for k in range(n):
        m = min(lo, n - 1 - k)
        col = a[U : U + m + 1, k]  # rows k..k+m of column k
        p = int(np.argmax(np.abs(col)))
        if col[p] == 0.0:
            raise ZeroDivisionError(f"zero pivot at column {k}")
        piv[k] = k + p
        if p:  # swap rows k and k+p within the band
            jj = np.arange(k, min(n, k + U + 1))
            r1, r2 = U + k - jj, U + k + p - jj
            a[r1, jj], a[r2, jj] = a[r2, jj].copy(), a[r1, jj].copy()
        pivot = a[U, k]
        if m == 0:
            continue
        a[U + 1 : U + m + 1, k] /= pivot
        jj = np.arange(k + 1, min(n, k + U + 1))
        if len(jj):
            akj = a[U + k - jj, jj]  # row k of the trailing columns
            rows = (U + k - jj)[None, :] + np.arange(1, m + 1)[:, None]
            a[rows, jj] -= np.outer(a[U + 1 : U + m + 1, k], akj)
    return (a, lo, U), piv


def lu_banded_solve(fact, piv, b) -> np.ndarray:
    """Solve with the factors of lu_banded: forward substitution with the row swaps, then back substitution."""
    a, lo, U = fact
    n = a.shape[1]
    x = np.asarray(b, float).copy()
    for k in range(n):  # forward with the row swaps: L is unit lower with multipliers stored in a[U+1:, k]
        p = piv[k]
        if p != k:
            x[k], x[p] = x[p], x[k]
        m = min(lo, n - 1 - k)
        if m:
            x[k + 1 : k + 1 + m] -= a[U + 1 : U + 1 + m, k] * x[k]
    for k in range(n - 1, -1, -1):  # back substitution with U
        x[k] /= a[U, k]
        i0 = max(0, k - U)
        if i0 < k:
            x[i0:k] -= a[U + i0 - k : U, k] * x[k]
    return x


def solve_banded(ab, lo: int, up: int, b) -> np.ndarray:
    """Factorise and solve a banded system in one call (O(n b^2) + O(n b))."""
    fact, piv = lu_banded(ab, lo, up)
    return lu_banded_solve(fact, piv, b)


def banded_jacobian(f, u, lo: int, up: int, eps: float = 1e-7) -> np.ndarray:
    """Jacobian of f at u in banded storage, by forward differences with column colouring: columns j, j + w,
    j + 2w, ... (w = lo + up + 1) never share a row, so all n columns come from w evaluations."""
    u = np.asarray(u, float)
    n = len(u)
    f0 = np.asarray(f(u), float)
    w = lo + up + 1
    ab = np.zeros((w, n))
    for c in range(w):
        cols = np.arange(c, n, w)
        h = eps * np.maximum(1.0, np.abs(u[cols]))
        up_ = u.copy()
        up_[cols] += h
        df = np.asarray(f(up_), float) - f0
        for j, hj in zip(cols, h):
            for i in range(max(0, j - up), min(n, j + lo + 1)):
                ab[up + i - j, j] = df[i] / hj
    return ab


def dense_jacobian(f, u, eps: float = 1e-7) -> np.ndarray:
    """Dense Jacobian by forward differences, one evaluation per column (for small systems and checks)."""
    u = np.asarray(u, float)
    f0 = np.asarray(f(u), float)
    J = np.empty((len(f0), len(u)))
    for j in range(len(u)):
        h = eps * max(1.0, abs(u[j]))
        up_ = u.copy()
        up_[j] += h
        J[:, j] = (np.asarray(f(up_), float) - f0) / h
    return J
