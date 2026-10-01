# Changelog

## 0.1.0 - 2026-10-01

First public release.

- **Library** (`engmolines`, 11 modules, NumPy only): grids, stencils with matched closures, ghost-point
  boundary conditions, conservative operators in planar and radial geometry, flux-limited advection and
  finite-volume conservation laws, Fourier spectral derivatives, banded linear algebra with coloured Jacobians,
  explicit, adaptive, implicit (BDF, trapezoidal, Radau start), IMEX, exponential, symplectic and DAE
  integrators, stiffness and von Neumann analysis, and 37 example problems with exact or benchmark solutions.
- **Course**: 16 executed notebooks with learning objectives, "Inside the algorithm" sections and exercises,
  and 16 notebooks of worked solutions (48 exercises), ending in a reactor-runaway project with its dataset.
- **Validation**: 174 checks against exact solutions, SciPy/LAPACK, stability theory, conservation laws and
  benchmarks, run with warnings as errors.
