"""engmolines: the numerical method of lines for engineering PDEs.

grids        node-centred, cell-centred, stretched, radial and 2-D tensor grids
stencils     Fornberg weights, differentiation and upwind operators in stencil form, banded conversion, Kronecker sums
boundary     Dirichlet, Neumann, Robin, periodic and beam end conditions by ghost points
operators    diffusion (variable coefficient, planar/cylindrical/spherical), advection (centred, upwind, limited),
             dispersion, biharmonic, 2-D Laplacian
linalg       banded LU with pivoting, coloured banded Jacobians
stiffness    Jacobian spectra, stiffness ratios, stability functions, explicit step limits, cost model
analysis     Fourier symbols, von Neumann amplification, modified equations, convergence studies, conservation
spectral     Fourier spectral derivatives, dealiasing, wavenumbers
datasets     course data files (a tubular reactor's temperature profiles and start-up)
datasets     the course data file (reactor temperature profiles)
problems     example PDEs as ready-made semi-discrete systems with exact or reference solutions
integrators  explicit RK, Dormand-Prince 5(4), BDF 1-5 and trapezoidal with banded Newton, Radau IIA, IMEX,
             exponential integration, Stormer-Verlet, DAE-BDF
"""

from . import (
    analysis,
    boundary,
    datasets,
    grids,
    integrators,
    linalg,
    operators,
    problems,
    spectral,
    stencils,
    stiffness,
)
from ._common import Solution

__version__ = "0.1.0"
__all__ = [
    "Solution",
    "__version__",
    "analysis",
    "boundary",
    "datasets",
    "grids",
    "integrators",
    "linalg",
    "operators",
    "problems",
    "spectral",
    "stencils",
    "stiffness",
]
