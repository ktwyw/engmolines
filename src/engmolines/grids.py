"""Grids for the method of lines: uniform node-centred and cell-centred grids, stretched grids that cluster points
where the solution varies fastest, radial grids for cylindrical and spherical symmetry, and tensor-product 2-D grids.

>>> from engmolines import grids
>>> g = grids.uniform(0.0, 1.0, 4)
>>> g.x.tolist(), g.dx
([0.0, 0.25, 0.5, 0.75, 1.0], 0.25)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid1D:
    """Points x (nodes or cell centres), the spacing dx (uniform) or spacings (non-uniform), and the domain."""

    x: np.ndarray
    a: float
    b: float
    kind: str = "node"  # 'node': x includes the ends; 'cell': cell centres; 'periodic': x[0] = a, x[-1] = b - dx

    @property
    def n(self) -> int:
        return len(self.x)

    @property
    def dx(self):
        d = np.diff(self.x)
        return float(d[0]) if np.allclose(d, d[0]) else d

    @property
    def uniform(self) -> bool:
        return np.isscalar(self.dx)

    @property
    def faces(self) -> np.ndarray:
        """Cell faces: midpoints between nodes (plus the ends) for node grids, the cell boundaries for cell grids."""
        if self.kind == "cell":
            return np.r_[self.a, 0.5 * (self.x[:-1] + self.x[1:]), self.b]
        if self.kind == "periodic":
            dx = self.x[1] - self.x[0]
            return np.r_[self.x - dx / 2, self.x[-1] + dx / 2]
        return np.r_[self.x[0], 0.5 * (self.x[:-1] + self.x[1:]), self.x[-1]]

    @property
    def widths(self) -> np.ndarray:
        """Control-volume width of each point (used for conservation sums and integrals)."""
        f = self.faces
        return f[1:] - f[:-1]

    def integrate(self, u) -> float:
        """Integral of u over the domain by the midpoint/trapezoid rule matched to the grid kind."""
        return float(np.sum(np.asarray(u) * self.widths))


def uniform(a: float, b: float, n: int) -> Grid1D:
    """n intervals, n + 1 nodes including both ends."""
    return Grid1D(np.linspace(a, b, n + 1), a, b, "node")


def cell_centred(a: float, b: float, n: int) -> Grid1D:
    """n cells with centres at a + (i + 1/2) dx: the finite-volume grid (no point on the boundary)."""
    dx = (b - a) / n
    return Grid1D(a + (np.arange(n) + 0.5) * dx, a, b, "cell")


def stretched(a: float, b: float, n: int, cluster: float = 0.5, strength: float = 2.0) -> Grid1D:
    """Nodes clustered around the fraction `cluster` of the domain (densest there, coarsest at the ends) by an
    arctanh mapping; cluster 0 or 1 clusters towards that end (a boundary layer). strength 0 is uniform, 2 gives
    a spacing ratio of about 30."""
    if cluster <= 0 or cluster >= 1:  # one-sided: dense at one end, coarse at the other
        s = np.linspace(0, 1, n + 1)
        t = s if strength == 0 else np.arctanh(s * np.tanh(strength)) / strength
        x = a + (b - a) * (t if cluster <= 0 else 1 - t[::-1])
        x[0], x[-1] = a, b
        return Grid1D(x, a, b, "node")
    s = np.linspace(-1, 1, n + 1)
    if strength == 0:
        xi = s
    else:
        xi = np.arctanh(s * np.tanh(strength)) / strength
    # map [-1, 1] to [a, b] with the cluster point at a + cluster (b - a)
    t = 0.5 * (xi + 1)
    x = a + (b - a) * (t if cluster == 0.5 else _shift(t, cluster))
    x[0], x[-1] = a, b  # exact ends (the mapping rounds them by an ulp)
    return Grid1D(x, a, b, "node")


def _shift(t, c):
    """Monotone map of [0, 1] onto itself sending 1/2 to c (a quadratic rational)."""
    return c * t / ((2 * c - 1) * t + 1 - c) if c != 0.5 else t  # sends 0 -> 0, 1/2 -> c, 1 -> 1


def periodic(a: float, b: float, n: int) -> Grid1D:
    """n equispaced points on [a, b) for periodic (spectral or finite-difference) problems; every width is dx."""
    return Grid1D(a + (b - a) * np.arange(n) / n, a, b, "periodic")


def radial(R: float, n: int, geometry: str = "cylindrical") -> Grid1D:
    """Uniform radial grid with a node at r = 0; `geometry` is recorded for the operators' 1/r terms."""
    g = uniform(0.0, R, n)
    g.kind = geometry
    return g


@dataclass
class Grid2D:
    """Tensor product of two 1-D grids; fields are arrays of shape (nx, ny) with x along axis 0."""

    gx: Grid1D
    gy: Grid1D

    @property
    def shape(self):
        return (self.gx.n, self.gy.n)

    @property
    def X(self):
        return np.meshgrid(self.gx.x, self.gy.x, indexing="ij")[0]

    @property
    def Y(self):
        return np.meshgrid(self.gx.x, self.gy.x, indexing="ij")[1]

    def integrate(self, U) -> float:
        return float(np.sum(np.asarray(U) * np.outer(self.gx.widths, self.gy.widths)))


def tensor(gx: Grid1D, gy: Grid1D) -> Grid2D:
    """Tensor-product 2-D grid of two 1-D grids (fields have shape (nx, ny), x along axis 0)."""
    return Grid2D(gx, gy)
