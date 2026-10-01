"""Generate the course notebooks (single source of truth) and execute them.

python notebooks/build_notebooks.py            # write and execute all notebooks
python notebooks/build_notebooks.py 03 15      # only notebooks whose names start with 03 or 15
python notebooks/build_notebooks.py --no-run   # write without executing
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
NOTEBOOKS: dict[str, list] = {}
EXTRAS: dict[str, dict] = {}

COLAB = """try:                      # on Google Colab (or anywhere engmolines is missing): install it from GitHub
    import engmolines
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/ktwyw/engmolines"], check=True)"""

STYLE = '''plt.rcParams.update({"figure.figsize": (7, 4), "figure.dpi": 90, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

import inspect
from IPython.display import Code

def show_source(obj):
    """Display the source code of a library function or class."""
    return Code(inspect.getsource(obj), language="python")

def heading(text):
    print(f"\\n{text}\\n" + "=" * len(text))'''


def md(text):
    return new_markdown_cell(text.strip("\n"))


def code(text):
    return new_code_cell(text.strip("\n"))


def setup_cell(extra_imports=()):
    lines = ["import matplotlib.pyplot as plt", "import numpy as np", "import pandas as pd"]
    for imp in extra_imports:
        if imp not in lines:
            lines.append(imp)
    return code(COLAB + "\n" + "\n".join(lines) + "\n" + STYLE)


def read_data(name):
    """Code snippet reading a bundled CSV file (comment lines start with #)."""
    return f'pd.read_csv(datasets.path("{name}"), comment="#")'


# =====================================================================================================


NOTEBOOKS["00_what_is_the_method_of_lines"] = [
    md(r"""
# 00 · What is the method of lines?

A partial differential equation such as the heat equation u_t = κ u_xx has two kinds of variable. The method
of lines treats them differently: discretise the *space* derivatives on a grid, and keep *time* continuous. What
remains is a system of ordinary differential equations - one per grid point - and ODE solvers are mature,
adaptive and well understood. This notebook builds that idea in twenty lines, then shows the two consequences
that organise the whole course: the semi-discrete system has its own exact solution (so spatial and temporal
errors can be separated and measured), and it is *stiff*, which decides which time integrators can be used.
"""),
    setup_cell(
        [
            "from scipy import integrate as sint",
            "from engmolines import grids, boundary as bc, operators as op, integrators as ti, stencils as st, stiffness, analysis",
        ]
    ),
    md("## The method of lines in twenty lines"),
    code(r"""n, kappa = 40, 1.0
x = np.linspace(0, 1, n + 1); dx = x[1] - x[0]

def rhs(t, u):                               # du/dt for every grid point: the semi-discrete heat equation
    dudt = np.zeros_like(u)
    dudt[1:-1] = kappa * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2   # central difference for u_xx
    return dudt                              # the boundary values u[0], u[-1] stay fixed (du/dt = 0)

u0 = np.sin(np.pi * x)
sol = sint.solve_ivp(rhs, (0, 0.1), u0, method="BDF", rtol=1e-8, atol=1e-10)
exact = np.sin(np.pi * x) * np.exp(-kappa * np.pi**2 * 0.1)
print(f"max error against the PDE solution: {np.max(np.abs(sol.y[:, -1] - exact)):.2e}  ({len(sol.t)} time steps)")
plt.plot(x, u0, "k--", label="t = 0"); plt.plot(x, sol.y[:, -1], "o", ms=3, label="method of lines, t = 0.1"); plt.plot(x, exact, label="exact"); plt.legend(); plt.xlabel("x"); plt.show()"""),
    md(r"""
That is the whole method: a right-hand side that evaluates the spatial derivatives at every grid point, handed
to an ODE solver. The error of 2×10⁻⁴ is almost entirely the spatial discretisation (the second-order
difference on 40 intervals); the BDF solver's tolerance of 10⁻⁸ contributes nothing visible. Everything that
follows is about understanding and controlling the two parts of that error, and about doing this efficiently
when the systems are large and stiff.

## The same, with the library
"""),
    code(r"""g = grids.uniform(0, 1, 40)
f = lambda t, u: op.diffusion(u, g, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t)   # noqa: E731
sol = ti.bdf(f, np.sin(np.pi * g.x), (0, 0.1), 0.002, order=4, band=(1, 1))
print(sol)
print(f"max error: {np.max(np.abs(sol.y[-1] - exact)):.2e}")"""),
    md(r"""
`operators.diffusion` applies the stencil with the boundary conditions handled by ghost points;
`integrators.bdf` is a fourth-order backward-differentiation formula whose Newton iterations use the tridiagonal
structure of the Jacobian (`band=(1, 1)`): one factorisation served all 50 steps, at about two Newton
iterations each. The `Solution` object reports these statistics because, in the method of lines, the cost of a
time step matters as much as its accuracy.

## The semi-discrete system has its own exact solution
"""),
    code(r"""A = st.diff_operator(g.x, 2, 2).to_dense(); A[0, :] = A[-1, :] = 0.0     # the matrix of the semi-discrete system u' = A u
ref = ti.exponential(A, np.sin(np.pi * g.x), (0, 0.1))                    # u(t) = exp(A t) u0: exact in time
lam_discrete = 4 / g.dx**2 * np.sin(np.pi * g.dx / 2) ** 2                  # the discrete decay rate of the first mode
print(f"continuous decay rate pi^2 = {np.pi**2:.6f}; discrete decay rate (4/dx^2) sin^2(pi dx/2) = {lam_discrete:.6f}")
print(f"BDF4 (h = 0.002) vs the PDE:          {np.max(np.abs(sol.y[-1] - exact)):.2e}   <- spatial error")
print(f"BDF4 (h = 0.002) vs the semi-discrete: {np.max(np.abs(sol.y[-1] - ref.y[-1])):.2e}   <- time error")
for h in (0.01, 0.005, 0.0025):
    s = ti.bdf(f, np.sin(np.pi * g.x), (0, 0.1), h, order=4, band=(1, 1))
    print(f"   h = {h}: time error {np.max(np.abs(s.y[-1] - ref.y[-1])):.2e}")"""),
    md(r"""
Because the semi-discrete system is linear, u′ = A u, its solution exp(At) u₀ can be computed exactly (here
from the eigendecomposition of A). That reference splits the error cleanly: against it, BDF4's time error is
10⁻¹⁰ and falls sixteenfold per halving of the step; against the PDE, the error is the 2×10⁻⁴ of the spatial
stencil, whose decay rate (4/Δx²) sin²(πΔx/2) differs from π² in the fourth digit. Every convergence study in this
course separates the two errors this way: refine the time step against the exact semi-discrete solution, and
refine the grid against the PDE.

## The semi-discrete system is stiff
"""),
    code(r"""lam = stiffness.spectrum(f, np.sin(np.pi * g.x), band=(1, 1))
lam = lam[np.abs(lam) > 1e-9]                                             # drop the two boundary rows
print(f"eigenvalues from {lam[0].real:.1f} to {lam[-1].real:.3f}: stiffness ratio {stiffness.stiffness_ratio(lam):.0f} (~ (2n/pi)^2 = {(2 * 40 / np.pi)**2:.0f})")
for method in ("euler", "rk4"):
    print(f"largest stable step for {method}: {stiffness.explicit_step_limit(lam, method):.2e}  (the solution's own time scale is 1/pi^2 = {1 / np.pi**2:.2f})")
sd = ti.dopri45(f, np.sin(np.pi * g.x), (0, 0.1), rtol=1e-6, atol=1e-9)
print(f"adaptive Dormand-Prince: {sd.steps} steps, mean step {np.mean(sd.extra['h']):.2e} - pinned at the stability limit, not chosen for accuracy")
plt.plot(lam.real, lam.imag, "o", ms=3); plt.xlabel("Re lambda"); plt.ylabel("Im lambda"); plt.title("spectrum of the semi-discrete heat operator"); plt.show()"""),
    md(r"""
The eigenvalues of the discrete Laplacian run from −π² (the mode we care about) to −4/Δx² (the sawtooth the
grid can barely represent): a stiffness ratio that grows like n². An explicit method's step must satisfy the
stability condition of the *fastest* mode, h < 2.8/(4/Δx²) ≈ 4×10⁻⁴ for RK4, although the solution evolves on
the time scale 1/π² ≈ 0.1. The adaptive explicit solver confirms it: it takes hundreds of tiny steps, its step
pinned by stability. Implicit methods (BDF, the trapezoidal rule, Radau) have no such limit, and with the
banded Jacobian their steps are cheap. Stiffness is the central fact of the method of lines, and the subject of
notebook 02.
"""),
]

NOTEBOOKS["01_stencils_and_boundary_conditions"] = [
    md(r"""
# 01 · Stencils and boundary conditions

The spatial half of the method of lines is a choice of stencils: which neighbouring values approximate a
derivative, with what order of accuracy, and what happens at the ends of the grid. This notebook derives stencil
weights for any derivative on any points (Fornberg's algorithm), measures their orders including the one-sided
closures at the boundaries, and then treats the boundary conditions themselves by ghost points: Dirichlet,
Neumann, Robin and periodic, on node-centred and cell-centred grids, with the test that every ghost formula
must pass.
"""),
    setup_cell(
        ["from engmolines import grids, stencils as st, boundary as bc, operators as op, analysis, linalg as la"]
    ),
    md("## Fornberg's weights"),
    code(r"""print("u_xx, 3 points:        ", st.fd_weights(0.0, [-1, 0, 1], 2))
print("u_xx, 5 points (4th):   ", st.fd_weights(0.0, [-2, -1, 0, 1, 2], 2) * 12, "/ 12")
print("u_x one-sided (2nd):    ", st.fd_weights(0.0, [0, 1, 2], 1) * 2, "/ 2")
print("u_x, non-uniform points:", np.round(st.fd_weights(0.3, [0.0, 0.5, 1.2], 1), 4))
print("u_xxxx, 5 points:       ", st.fd_weights(0.0, [-2, -1, 0, 1, 2], 4))"""),
    md(r"""
Fornberg's recurrence gives the weights of the formula Σ w_j f(x_j) ≈ f^(m)(x₀) that is exact for polynomials
of the highest possible degree, on any set of points - centred or one-sided, uniform or not. Every stencil in
the library comes from it, including the biharmonic [1, −4, 6, −4, 1]/Δx⁴ for beams and the one-sided
formulas the boundaries need.

## Differentiation operators and their orders
"""),
    code(r"""f = lambda x: np.sin(3 * x)                                  # noqa: E731
exact = {1: lambda x: 3 * np.cos(3 * x), 2: lambda x: -9 * np.sin(3 * x), 4: lambda x: 81 * np.sin(3 * x)}
ns = [20, 40, 80, 160]
for deriv, acc in ((1, 2), (1, 4), (2, 2), (2, 4), (2, 6), (4, 2)):
    errs = [np.max(np.abs(st.diff_operator(np.linspace(0, 1, n + 1), deriv, acc)(f(np.linspace(0, 1, n + 1))) - exact[deriv](np.linspace(0, 1, n + 1)))) for n in ns]
    print(f"derivative {deriv}, accuracy {acc}: errors {['%.1e' % e for e in errs]}, observed orders {np.round(analysis.observed_order(errs), 2)}")
D = st.diff_operator(np.linspace(0, 1, 9), 2, 2)
print("\nthe 3-point u_xx operator on 9 points, as a dense matrix (times dx^2):\n", np.round(D.to_dense() * (1 / 8) ** 2, 3))"""),
    md(r"""
`diff_operator` builds a differentiation operator of any derivative and accuracy: centred stencils inside,
one-sided stencils of the *same* accuracy at the ends (one more point each, so the matrix's first and last
rows reach further in). The orders come out as designed - and sometimes better, because the one-sided closures
happen to be more accurate for this function; the sixth-order operator reaches round-off (10⁻¹⁰, the
subtraction of nearly equal values divided by Δx²) on the finest grid, where its order collapses. The
operators are stored by diagonals and applied in O(n); a dense matrix is only formed for display or for
eigenvalues.

## Ghost points: the method-of-lines way to impose boundary conditions
"""),
    code(r"""u = np.array([1.0, 1.4, 1.9, 2.5, 3.2]); dx = 0.1
for name, left, right, kind in (("Dirichlet / Neumann, node grid", bc.Dirichlet(0.7), bc.Neumann(-2.0), "node"),
                                ("Robin / Neumann, cell grid", bc.Robin(2.0, 0.5, 3.0), bc.Neumann(1.5), "cell"),
                                ("periodic", bc.Periodic(), bc.Periodic(), "cell")):
    p = bc.pad(u, dx, left, right, kind=kind, nghost=2)
    print(f"{name:<32}: {np.round(p, 3)}")
p = bc.pad(u, dx, bc.Robin(2.0, 0.5, 3.0), bc.Neumann(1.5), kind="cell")
print(f"\nRobin check at the left face: alpha u + beta u_x = {2.0 * (p[0] + p[1]) / 2 + 0.5 * (p[1] - p[0]) / dx:.3f} (should be 3)")
p = bc.pad(u, dx, bc.Dirichlet(0.7), bc.Neumann(-2.0), kind="node")
print(f"Neumann check at the right node: (u_ghost - u_inner)/(2 dx) = {(p[-1] - p[-3]) / (2 * dx):.3f} (should be -2)")"""),
    md(r"""
Rather than writing special stencils for the boundary rows, the state is extended by ghost values chosen so
that the boundary condition holds (to second order), and the interior stencil is applied everywhere. A
Dirichlet value is imposed by odd reflection about the boundary, a Neumann slope by even reflection plus the
slope, a Robin condition by solving the small equation that links boundary value and slope; periodic grids
simply wrap. On a node grid the boundary point is itself a grid point - its Dirichlet value is fixed and its
rate set to zero (or to dg/dt for a time-dependent value) - while a cell-centred grid has no point on the
boundary and imposes everything through the ghosts. The checks above are the tests every ghost formula must
pass: evaluate the condition with the ghosts in place and get the imposed value back.

## Consistency: what the operator does to a polynomial
"""),
    code(r"""g = grids.uniform(0, 1, 40)
print("u = x^2, Dirichlet ends: u_xx =", np.unique(np.round(op.diffusion(g.x**2, g, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(1.0))[1:-1], 10)))
gc = grids.cell_centred(0, 1, 40)
k = np.where(gc.x < 0.5, 1.0, 4.0)
u_layer = np.where(gc.x < 0.5, gc.x, 0.5 + 0.25 * (gc.x - 0.5))          # continuous flux k u_x = 1 across the interface at the face x = 0.5
print("composite wall, flux-continuous profile: max |residual| =", f"{np.max(np.abs(op.diffusion(u_layer, gc, k, bc.Dirichlet(0.0), bc.Dirichlet(0.625)))):.1e}")
gr = grids.radial(1.0, 40, "cylindrical")
print("cylindrical (1/r)(r u_r)_r of r^2 (should be 4, including r = 0):", np.unique(np.round(op.diffusion(gr.x**2, gr, 1.0, None, bc.Dirichlet(1.0))[:-1], 8)))"""),
    md(r"""
Three consistency checks that catch most discretisation errors: a quadratic must give the exact second
derivative; a piecewise-linear profile with continuous flux through a conductivity jump must be a steady state
(the operator averages conductivities harmonically at the faces - which is why material interfaces belong on
faces, so the composite wall lives on a cell-centred grid); and the r = 0 axis of a cylindrical grid must be
handled by a control volume, not by dividing by zero.

## Non-uniform grids
"""),
    code(r"""gs = grids.stretched(0, 1, 40, cluster=0.5, strength=2.0)
print(f"stretched grid: spacing from {np.diff(gs.x).min():.4f} to {np.diff(gs.x).max():.4f}; the differentiation operator uses Fornberg weights at every point")
f2 = lambda x: np.exp(-((x - 0.5) / 0.05) ** 2)                 # noqa: E731, a narrow bump at the cluster point
d2 = lambda x: f2(x) * (4 * (x - 0.5) ** 2 / 0.05**4 - 2 / 0.05**2)   # noqa: E731
gu = grids.uniform(0, 1, 40)
print(f"u_xx of the bump: error on the uniform grid {np.max(np.abs(st.diff_operator(gu.x, 2, 2)(f2(gu.x)) - d2(gu.x))):.2e}, on the stretched grid {np.max(np.abs(st.diff_operator(gs.x, 2, 2)(f2(gs.x)) - d2(gs.x))):.2e}")"""),
    md(r"""
The same 40 points, clustered where the solution varies, reduce the error of the second derivative of a narrow
bump by a factor of four (the bump is still under-resolved; the gain grows with the clustering). Stretched grids cost nothing in the method of lines - the stencil weights absorb
the spacing - and are the usual remedy for boundary layers and fronts of known position.
"""),
]

NOTEBOOKS["02_stiffness_and_choosing_the_integrator"] = [
    md(r"""
# 02 · Stiffness and choosing the integrator

Semi-discrete systems are stiff: their fastest modes are grid artefacts with rates −4κ/Δx², far faster than
anything in the solution. This notebook makes stiffness quantitative - spectra, stiffness ratios, the stability
regions of the integrators and the step each one can take - and then compares the integrators on the same
problem: explicit Runge-Kutta pinned at its stability limit, the adaptive solver wasting its steps, BDF and the
trapezoidal rule stepping at the solution's pace, and the difference between A-stability and L-stability that
shows up as ringing. It ends with a cost model that says when implicit methods win.
"""),
    setup_cell(
        [
            "from engmolines import grids, boundary as bc, operators as op, integrators as ti, stencils as st, stiffness, analysis"
        ]
    ),
    md("## Spectra and stiffness ratios"),
    code(r"""rows = []
for n in (10, 20, 40, 80):
    g = grids.uniform(0, 1, n)
    f = lambda t, u, g=g: op.diffusion(u, g, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t)
    lam = stiffness.spectrum(f, np.zeros(n + 1), band=(1, 1)); lam = lam[np.abs(lam) > 1e-9]
    rows.append((n, lam[0].real, -4 / g.dx**2, stiffness.stiffness_ratio(lam), stiffness.explicit_step_limit(lam, "rk4"), stiffness.explicit_step_limit(lam, "euler")))
print(pd.DataFrame(rows, columns=["n", "lambda_min", "-4/dx^2", "stiffness ratio", "RK4 step limit", "Euler step limit"]).to_string(index=False, float_format="%.4g"))"""),
    md(r"""
Refining the grid by two multiplies the extreme eigenvalue by four and the stiffness ratio by four, and divides
the explicit step by four: the explicit cost of the heat equation grows like n³ in 1-D (n points × n² steps) -
and like n⁴ in 2-D. The theoretical value −4/Δx² is the exact bound; the largest grid mode comes within a
fraction of a percent of it.

## Stability regions
"""),
    code(r"""X, Y = np.meshgrid(np.linspace(-5, 1, 300), np.linspace(-3.5, 3.5, 300)); Z = X + 1j * Y
fig, ax = plt.subplots(figsize=(6, 5))
for i, m in enumerate(("euler", "heun", "rk4", "dopri")):
    ax.contour(X, Y, np.abs(stiffness.stability_function(m, Z)), [1.0], colors=f"C{i}"); ax.plot([], [], color=f"C{i}", label=m)
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5); ax.set_aspect("equal"); ax.legend(); ax.set_title("|R(z)| = 1: stable inside"); plt.show()
th = np.linspace(-200, 200, 801)
print("A-stable (|R| <= 1 on the whole imaginary axis): trapezoidal", np.max(np.abs(stiffness.stability_function("trapezoidal", 1j * th))) <= 1 + 1e-12, "; BDF2", np.max(np.abs(stiffness.stability_function("bdf2", 1j * th))) <= 1 + 1e-12)
print(f"|R(-1000)|: trapezoidal {abs(stiffness.stability_function('trapezoidal', -1000.0)):.3f} (not L-stable), backward Euler {abs(stiffness.stability_function('backward_euler', -1000.0)):.3f}, BDF2 {abs(stiffness.stability_function('bdf2', -1000.0)):.3f}")"""),
    md(r"""
On y′ = λy a method multiplies the solution by R(hλ) per step; it is stable when |R| ≤ 1. The explicit methods'
regions are bounded - RK4 reaches −2.79 on the negative real axis, Euler only −2 - so the stiffest mode sets the
step. The trapezoidal rule and BDF2 are A-stable (the whole left half-plane), but only BDF2 (and backward
Euler) are L-stable: R → 0 as hλ → −∞, so the fastest modes are damped in one step, while the trapezoidal rule
multiplies them by nearly −1 and they ring.

## The integrators side by side
"""),
    code(r"""g = grids.uniform(0, 1, 40)
f = lambda t, u: op.diffusion(u, g, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t)   # noqa: E731
u0 = np.sin(np.pi * g.x)
A = st.diff_operator(g.x, 2, 2).to_dense(); A[0, :] = A[-1, :] = 0.0
ref = ti.exponential(A, u0, (0, 0.1)).y[-1]
lam = stiffness.spectrum(f, u0, band=(1, 1)); lam = lam[np.abs(lam) > 1e-9]
h_rk4 = stiffness.explicit_step_limit(lam, "rk4")
runs = {"RK4 at 0.95 of its limit": ti.explicit_rk(f, u0, (0, 0.1), 0.95 * h_rk4),
        "RK4 at 1.05 of its limit": ti.explicit_rk(f, u0, (0, 0.1), 1.05 * h_rk4),
        "Dormand-Prince rtol 1e-6": ti.dopri45(f, u0, (0, 0.1), 1e-6, 1e-9),
        "backward Euler h = 0.005": ti.bdf(f, u0, (0, 0.1), 0.005, 1, band=(1, 1)),
        "BDF2 h = 0.005": ti.bdf(f, u0, (0, 0.1), 0.005, 2, band=(1, 1)),
        "BDF4 h = 0.005": ti.bdf(f, u0, (0, 0.1), 0.005, 4, band=(1, 1)),
        "trapezoidal h = 0.005": ti.trapezoidal(f, u0, (0, 0.1), 0.005, band=(1, 1))}
for name, s in runs.items():
    err = np.max(np.abs(s.y[-1] - ref))
    print(f"{name:<26}: {s.steps:5d} steps, time error {err:9.2e}" + (f", Newton {s.newton_iterations}, Jacobians {s.jacobian_evaluations}" if s.newton_iterations else "") + ("   <- UNSTABLE" if err > 1 else ""))"""),
    md(r"""
Against the exact semi-discrete solution: RK4 is accurate to 10⁻¹¹ just below its limit and blows up just above
it; the adaptive solver survives by taking hundreds of steps; the implicit methods take twenty steps of 0.005 -
twelve times the explicit limit - at one Jacobian factorisation and two Newton iterations per step. Their errors
are set by order: 10⁻³ for backward Euler, 10⁻⁵ for BDF2 and the trapezoidal rule, 10⁻⁸ for BDF4.

## Ringing: why L-stability matters
"""),
    code(r"""u_rough = np.where((g.x > 0.4) & (g.x < 0.6), 1.0, 0.0)                     # a discontinuous initial state excites the fastest modes
h = 0.02
tz = ti.trapezoidal(f, u_rough, (0, 0.1), h, band=(1, 1), n_out=5)
be = ti.bdf(f, u_rough, (0, 0.1), h, 1, band=(1, 1), n_out=5)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
for ax, s, title in ((axes[0], tz, "trapezoidal: the band's edges ring"), (axes[1], be, "backward Euler: damped at once")):
    for t, y in zip(s.t[1:], s.y[1:]):
        ax.plot(g.x, y, label=f"t = {t:.2f}")
    ax.set_title(title); ax.legend(fontsize=7)
plt.show()
print(f"h lambda_max = {h * lam[0].real:.0f}: R = {stiffness.stability_function('trapezoidal', h * lam[0].real).real:+.3f} for the trapezoidal rule, {stiffness.stability_function('backward_euler', h * lam[0].real).real:+.4f} for backward Euler")"""),
    md(r"""
With hλ_max ≈ −130, the trapezoidal rule's amplification factor for the sawtooth modes is −0.97: the edges of the
hot band flip sign every step and the solution oscillates for many steps although it is "stable". Backward
Euler's factor is 0.008 and the profile is smooth after one step. For rough data - and PDE initial data is
often rough - L-stability, not just A-stability, is required; BDF2 and Radau have it, the trapezoidal rule
(Crank-Nicolson) does not, which is why it is often started with a few backward-Euler steps.

## When do implicit methods win? A cost model
"""),
    code(r"""rows = []
for n in (20, 40, 80, 160, 320):
    g_ = grids.uniform(0, 1, n)
    lam_ = stiffness.spectrum(lambda t, u, g_=g_: op.diffusion(u, g_, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t), np.zeros(n + 1), band=(1, 1))
    lam_ = lam_[np.abs(lam_) > 1e-9]
    c = stiffness.cost_model(n + 1, lam_, 0.1, 1, explicit="rk4", implicit_h=0.1 / 100)
    rows.append((n, c["explicit_steps"], c["explicit_flops"], c["implicit_steps"], c["implicit_flops"], c["ratio"]))
print(pd.DataFrame(rows, columns=["n", "RK4 steps", "RK4 flops", "BDF steps", "BDF flops", "RK4/BDF"]).to_string(index=False, float_format="%.3g"))"""),
    md(r"""
Explicit work grows like n³ (steps ∝ n², work per step ∝ n); implicit work with a banded Jacobian grows like n
(a fixed number of steps for the accuracy, each a tridiagonal solve). The crossover is at a few dozen points;
by n = 320 the implicit method is over a hundred times cheaper, and the ratio keeps growing like n². In two dimensions, with the band of width n, the
implicit advantage is smaller but still decisive for diffusion-dominated problems; for problems whose
stiffness comes from advection or reaction alone, explicit methods remain competitive - the subject of the
later notebooks.
"""),
]

NOTEBOOKS["03_banded_newton_and_the_cost_of_implicit_steps"] = [
    md(r"""
# 03 · Banded Newton and the cost of implicit steps

An implicit step solves a nonlinear system the size of the grid. Done carelessly - a dense Jacobian by n
function evaluations, a dense LU of n³/3 flops, every step - it costs more than the explicit method it was
meant to replace. Done with the structure of the method of lines it costs almost nothing: the Jacobian has the
band of the stencil, a banded LU costs n b², a finite-difference Jacobian needs b + 1 evaluations by colouring
the columns, and one Jacobian serves many steps. This notebook shows each of these ideas working, and what
the Newton iteration does when the Jacobian goes stale.
"""),
    setup_cell(
        [
            "from scipy import linalg as sla",
            "from engmolines import grids, boundary as bc, operators as op, integrators as ti, stencils as st, linalg as la, problems as pr",
        ]
    ),
    md("## The Jacobian has the band of the stencil"),
    code(r"""g = grids.uniform(0, 1, 30)
f = lambda u: op.diffusion(u, g, lambda x, u: 1 + u, bc.Dirichlet(1.0), bc.Dirichlet(0.0))    # noqa: E731, nonlinear conductivity
u = np.linspace(1, 0, 31)
J = la.dense_jacobian(f, u)
nz = np.abs(J) > 1e-9
print(f"Jacobian {J.shape}: nonzeros only on the diagonals {sorted({int(j - i) for i, j in zip(*np.where(nz))})}")
plt.spy(nz, markersize=3); plt.title("sparsity of the Jacobian: tridiagonal"); plt.show()"""),
    md(r"""
Each grid point's rate depends only on itself and its stencil neighbours, so the Jacobian is banded with the
stencil's half-width: tridiagonal for the three-point Laplacian, pentadiagonal for five-point stencils, and
of band n for a 2-D grid of n × n points (notebook 12). A dense Jacobian of a 10⁴-point grid holds 10⁸ numbers,
all but 3×10⁴ of them zero.

## Colouring: b + 1 evaluations instead of n
"""),
    code(r"""Jb = la.banded_jacobian(f, u, 1, 1)
print(f"banded Jacobian storage {Jb.shape} (3 rows x n) from 3 function evaluations; max |difference| from the dense one: {np.max(np.abs(la.from_banded(Jb, 1, 1) - J)):.1e}")
D4 = st.diff_operator(g.x, 2, 4)
print(f"5-point stencil: band {D4.bandwidth} (the one-sided closures reach 3 points in) -> {sum(D4.bandwidth) + 1} evaluations")"""),
    md(r"""
Columns j, j + w, j + 2w, ... of a banded Jacobian (w = bandwidth) never share a row, so perturbing them
together in one function evaluation recovers all of them: w evaluations for the whole Jacobian, whatever n is
(Curtis, Powell and Reid, 1974). For a 2-D problem with band n this is n evaluations instead of n² - still
cheap.

## Banded LU: n b² flops instead of n³/3
"""),
    code(r"""for n in (100, 1000, 10000):
    for b in (1, 2, 10):
        print(f"n = {n:6d}, band {b:2d}: banded LU ~ {2 * n * b**2:.1e} flops, dense LU ~ {2 * n**3 / 3:.1e} flops ({2 * n**3 / 3 / (2 * n * b**2):.0e}x)")
ab = la.to_banded(np.eye(31) - 0.01 * J, 1, 1)
x = la.solve_banded(ab, 1, 1, np.ones(31))
print(f"\nour banded solve vs scipy.linalg.solve_banded: {np.max(np.abs(x - sla.solve_banded((1, 1), ab, np.ones(31)))):.1e}")"""),
    md(r"""
Elimination inside a band never creates entries outside it (partial pivoting can widen the upper band to
lo + up, which the factorisation allows for), so the factorisation costs about 2 n b² flops and each solve
2 n (2b + 1): linear in n. The library's banded LU reproduces SciPy's to rounding.

## One Jacobian, many steps: the modified Newton iteration
"""),
    code(r"""p = pr.nonlinear_conductivity()
for h, label in ((0.005, "moderate steps"), (0.02, "large steps")):
    s = p.solve("bdf", h, order=2)
    print(f"{label} (h = {h}): {s.steps} steps, {s.newton_iterations} Newton iterations ({s.newton_iterations / s.steps:.1f} per step), {s.jacobian_evaluations} Jacobian evaluations; steady-state error {np.max(np.abs(s.y[-1] - p.exact(p.x))):.1e}")"""),
    md(r"""
The Newton iteration keeps the factorised Jacobian from a previous step (a *modified* Newton method), which
converges linearly rather than quadratically but costs only one banded solve per iteration. The iteration
monitors its own convergence rate and refreshes the Jacobian - a new coloured evaluation and factorisation -
only when it slows below a threshold. On this nonlinear problem a handful of Jacobians serve hundreds of steps.

The large-step run is a warning. Its Newton iterations converged at every step and the integrator reported
success - yet the final "steady state" is wrong by 2: part of the solution went to u = −1, where the
conductivity 1 + u vanishes, a non-physical equilibrium of the discrete equations that a step of 0.02 from the
discontinuous start lands in. (At h = 0.05 Newton fails outright at the second step and the fixed-step
integrator says so.) An implicit step too large for the nonlinearity can converge to a spurious solution;
convergence of Newton is not evidence of accuracy, and a check against physics - here, that the conductivity
stays positive - is part of the method. The next cell shows what the convergence monitor sees.
"""),
    code(r"""newton = ti._Newton(lambda t, u: p.rhs(t, u), (1, 1))
y = p.y0.copy(); h = 0.005
newton.refresh(0.0, y, h)
z = y.copy(); rhs_ = y.copy()
print("modified Newton with the Jacobian frozen at the initial (discontinuous) state - step sizes:")
for k in range(8):
    G = z - h * p.rhs(h, z) - rhs_
    dz = newton._solve(-G); z += dz
    print(f"   iteration {k + 1}: |dz| = {np.max(np.abs(dz)):.2e}")
z = y.copy()
print("full Newton (Jacobian refreshed every iteration):")
for k in range(5):
    newton.refresh(h, z, h)
    G = z - h * p.rhs(h, z) - rhs_
    dz = newton._solve(-G); z += dz
    print(f"   iteration {k + 1}: |dz| = {np.max(np.abs(dz)):.2e}")"""),
    md(r"""
From the initial state - a hot end next to a cold body, where the conductivity doubles in the first step - the
Jacobian frozen at that state is useless: the iteration wanders and diverges, while full Newton converges
quadratically in five iterations. This is why the library's Newton monitors its rate and refreshes the
Jacobian as soon as convergence slows (or the iterate stops being finite): a coloured Jacobian costs a few
evaluations, far less than a failed step. After the transient the solution changes little per step and one
Jacobian lasts for hundreds of steps.
"""),
]

NOTEBOOKS["04_the_heat_equation_in_every_form"] = [
    md(r"""
# 04 · The heat equation in every form

The heat equation is the model parabolic problem, and its variants cover most of what engineering conduction
asks: fixed, insulated or convective surfaces, periodic heating, layered materials, cylinders and spheres, and
fins. Each has an exact solution, which makes it the ideal place to practise the discipline of the course:
separate the spatial and temporal errors, measure the orders, check conservation. This notebook runs through
the library's heat problems with those checks.
"""),
    setup_cell(["from engmolines import problems as pr, integrators as ti, analysis, stiffness"]),
    md("## Boundary conditions: Dirichlet, Neumann, Robin, time-dependent"),
    code(r"""for make in (pr.heat_dirichlet, pr.heat_neumann, pr.heat_robin, pr.heat_time_dependent):
    p = make()
    s = p.solve("bdf", 1e-3 if make is not pr.heat_time_dependent else 5e-3, order=4)
    print(f"{p.name:<36}: error vs exact {p.error(s):.2e}; {p.notes}")
p = pr.heat_neumann(); s = p.solve("bdf", 1e-3, order=2, n_out=5)
print(f"\ninsulated ends: the integral of u changes by {np.ptp(analysis.conservation(p.grid, s.y)):.1e} - the Neumann ghosts are conservative")
p = pr.heat_robin()
print(f"convective ends with Bi = {p.params['Bi']}: first eigenvalue mu = {p.params['mu']:.5f} from mu tan(mu/2) = Bi")"""),
    md(r"""
All four solve to the accuracy of the spatial grid (10⁻⁴), with the time error below it at BDF4 and h = 10⁻³.
The Neumann problem conserves the heat content because the ghost formula makes the boundary flux exactly
zero; the Robin problem's decay rate comes from the transcendental equation μ tan(μ/2) = Bi that couples the
Biot number to the eigenvalue - a result the numerical solution reproduces without knowing it.

## Separating the errors: a space-time convergence table
"""),
    code(r"""def run(n, h):
    p = pr.heat_dirichlet(n=n)
    s = p.solve("bdf", h, order=2)
    return p.x, s.y[-1]
p = pr.heat_dirichlet()
study = analysis.convergence_study(run, [10, 20, 40, 80], [0.02, 0.01, 0.005, 0.0025], lambda x: p.exact(x, 0.2))
print(pd.DataFrame(study["errors"], index=[f"n = {n}" for n in (10, 20, 40, 80)], columns=[f"h = {h}" for h in (0.02, 0.01, 0.005, 0.0025)]).to_string(float_format="%.2e"))
print("orders along the finest time step (space):", np.round(study["space_orders"], 2), "  along the finest grid (time):", np.round(study["time_orders"], 2))"""),
    md(
        "Isolating each error: BDF4 at a tiny step for the spatial order, and the exact semi-discrete solution for the time order."
    ),
    code(r"""from engmolines import stencils as st
e_space = [pr.heat_dirichlet(n=n).error(pr.heat_dirichlet(n=n).solve("bdf", 2e-4, order=4)) for n in (10, 20, 40, 80)]
p40 = pr.heat_dirichlet(n=40)
A = st.diff_operator(p40.x, 2, 2).to_dense(); A[0, :] = A[-1, :] = 0.0
ref = ti.exponential(A, p40.y0, (0, 0.2)).y[-1]
e_time = [np.max(np.abs(p40.solve("bdf", h, order=2).y[-1] - ref)) for h in (0.02, 0.01, 0.005, 0.0025)]
print("spatial orders (time error negligible):", np.round(analysis.observed_order(e_space), 2))
print("temporal orders (against the exact semi-discrete solution):", np.round(analysis.observed_order(e_time), 2))"""),
    md(r"""
Along a row the time error falls fourfold per halving of h (BDF2) until it meets the spatial floor; down a
column the spatial error falls fourfold per doubling of n until the time error takes over - and where the two
are comparable they can *cancel*: in the n = 10 row the error grows as h shrinks, because the time error had
been compensating part of the spatial one. Mixed tables are therefore read only where one error dominates;
the clean measurement of each order isolates it, as the next cell does.

## Layers, cylinders, spheres
"""),
    code(r"""p = pr.composite_wall(); s = p.solve("bdf", 0.0075, order=2, n_out=6)
for t, y in zip(s.t, s.y):
    plt.plot(p.x, y, label=f"t = {t:.1f}")
plt.plot(p.x, p.exact(p.x), "k--", label="exact steady state"); plt.axvline(0.5, color="0.7"); plt.legend(fontsize=7); plt.title("composite wall: k = 1 | k = 5"); plt.show()
print(f"steady-state error {np.max(np.abs(s.y[-1] - p.exact(p.x))):.1e}; heat flux q = {p.params['q']:.4f} = (T_L - T_R) / (L1/k1 + L2/k2)")
for make in (pr.cooling_cylinder, pr.cooling_sphere):
    p = make(); s = p.solve("bdf", 1e-3, order=4)
    print(f"{p.name:<18}: first-mode decay exact to {p.error(s):.1e} (the r = 0 control volume: no singularity)")"""),
    md(r"""
The composite wall reaches the series-resistance steady state exactly because harmonic face averaging makes
the flux continuous across the interface. The cylinder's and sphere's first modes - a Bessel function and
sin(πr)/(πr) - decay at their exact rates; at r = 0 the operator uses the control-volume form (2(p + 1)κ(u₁ −
u₀)/Δr² at the axis) rather than the singular 1/r.

## A fin heating up: stiff, with a known steady state
"""),
    code(r"""p = pr.fin_transient()
lam = stiffness.spectrum(p.rhs, p.y0, band=(1, 1)); lam = lam[np.abs(lam) > 1e-9]
print(f"eigenvalues from {lam[0].real:.1f} to {lam[-1].real:.4f} 1/s: ratio {stiffness.stiffness_ratio(lam):.0f}; RK4 limit {stiffness.explicit_step_limit(lam, 'rk4'):.3f} s")
s = p.solve("bdf", 2.0, order=2, n_out=9)
for t, y in zip(s.t, s.y):
    plt.plot(p.x * 1000, y, color=plt.cm.viridis(t / s.t[-1]))
plt.plot(p.x * 1000, p.exact(p.x), "k--", label="steady state (cosh/sinh)"); plt.xlabel("x (mm)"); plt.ylabel("T (degC)"); plt.legend(); plt.show()
print(f"BDF2 with 2 s steps ({s.steps} steps, {s.jacobian_evaluations} Jacobian): steady-state error {np.max(np.abs(s.y[-1] - p.exact(p.x))):.1e} K; time constant 1/beta = {p.params['time_constant']:.0f} s")"""),
    md(r"""
The fin's semi-discrete system has eigenvalues from −82 s⁻¹ (the grid modes) to −0.03 s⁻¹ (the fin's own
time constant of about 30 s): explicit RK4 could take steps of 0.034 s, BDF2 takes 2 s - sixty times larger -
and reaches the classical cosh/sinh steady state. The problem is the project of the companion course
`engnumeric`, solved there by hand; here it is one line.
"""),
]

NOTEBOOKS["05_nonlinear_diffusion"] = [
    md(r"""
# 05 · Nonlinear diffusion

When the diffusivity depends on the solution, the semi-discrete system is nonlinear and the implicit solvers
earn their Newton iterations. Four classic problems show four behaviours: a temperature-dependent conductivity
(a smooth nonlinearity, Newton converges easily), the porous-medium equation (degenerate: the diffusivity
vanishes where u = 0, the solution has a sharp front with compact support, and the right formulation matters),
the Allen-Cahn equation (a bistable reaction whose kink must not move), and the Cahn-Hilliard equation (fourth
order, with conserved mass and a decreasing free energy).
"""),
    setup_cell(["from engmolines import problems as pr, integrators as ti, analysis, stiffness"]),
    md("## Temperature-dependent conductivity"),
    code(r"""p = pr.nonlinear_conductivity(b=1.0)
s = p.solve("bdf", 0.005, order=2, n_out=5)
for t, y in zip(s.t, s.y):
    plt.plot(p.x, y, label=f"t = {t:.1f}")
plt.plot(p.x, p.exact(p.x), "k--", label="steady: u + u^2/2 linear"); plt.legend(fontsize=8); plt.show()
print(f"steady-state error {np.max(np.abs(s.y[-1] - p.exact(p.x))):.1e}; Newton {s.newton_iterations / s.steps:.1f} per step, {s.jacobian_evaluations} Jacobians over {s.steps} steps")"""),
    md(r"""
With k = 1 + u the steady flux k u_x is constant, so u + u²/2 is linear in x: the solution reproduces it to
10⁻⁵. The Jacobian changes with the solution, and the modified Newton refreshed it four times - during the
early transient when the profile changes fastest - and reused it for the remaining hundreds of steps.

## The porous-medium equation: a degenerate problem
"""),
    code(r"""p = pr.porous_medium(m=2)
s = p.solve("bdf", 2e-3, order=2, n_out=5)
for t, y in zip(s.t, s.y):
    plt.plot(p.x, y, label=f"t = {t:.1f}"); plt.plot(p.x, p.exact(p.x, t), "k:", lw=0.8)
plt.legend(fontsize=8); plt.title("Barenblatt solution (dotted: exact)"); plt.xlim(-4, 4); plt.show()
print(f"error vs Barenblatt {p.error(s):.1e}; mass drift {p.grid.integrate(s.y[-1]) - p.grid.integrate(p.y0):+.1e}; front at {p.x[s.y[-1] > 1e-6][-1]:.3f} (exact {np.sqrt(12 * 1.1**(2/3)):.3f})")
print(f"Newton {s.newton_iterations / s.steps:.1f} per step - the front costs iterations")"""),
    md(r"""
The Barenblatt solution spreads with compact support: the diffusivity m uᵐ⁻¹ is zero ahead of the front, and
the front moves at a finite speed, t^(1/3) for m = 2. Two choices made this work. The equation is discretised
as u_t = (uᵐ)_xx - the linear Laplacian applied to uᵐ - whose Jacobian m uᵐ⁻¹ is continuous, rather than as
∇·(m uᵐ⁻¹ ∇u) with a harmonic mean of diffusivities that is 0/0 at the front; and the domain is wide enough
to contain the support until the end (an earlier version of this problem let the front hit the wall and lost
a quarter of its mass). Even so, Newton needs more iterations than for a smooth problem.

## Allen-Cahn: a stationary kink
"""),
    code(r"""p = pr.allen_cahn(eps=0.01)
s = p.solve("bdf", 0.01, order=4, n_out=1)
print(f"after t = {p.t_end}: the kink moved by {np.max(np.abs(s.y[-1] - p.y0)):.1e} (it is an exact stationary solution)")
lam = stiffness.spectrum(p.rhs, p.y0, band=(1, 1)); lam = lam[np.abs(lam) > 1e-9]
print(f"spectrum: from {lam[0].real:.0f} (diffusion) to {lam[-1].real:+.4f}: the slowest mode is the kink's translation, almost neutral")
s_rk = ti.explicit_rk(p.rhs, p.y0, (0, 1.0), 0.95 * stiffness.explicit_step_limit(lam, "rk4"), n_out=1)
print(f"RK4 needs {s_rk.steps} steps for t = 1 where BDF4 took {int(1.0 / 0.01)}")"""),
    md(r"""
u_t = ε u_xx + u − u³ has two stable states ±1 joined by the kink tanh(x/√(2ε)), a stationary solution: a good
discretisation must not move it, and this one changes it by 3×10⁻⁴ in five time units (the spatial error of
the kink's profile on a grid of 200 points; it does not grow). The spectrum shows the
structure of bistable problems: fast diffusive modes, a reaction time scale of order one, and a nearly neutral
translation mode of the front - the mode that makes fronts drift under numerical perturbations if the scheme
is not careful.

## Cahn-Hilliard: fourth order, conserved mass, decreasing energy
"""),
    code(r"""p = pr.cahn_hilliard(eps=0.02)
s = p.solve("bdf", 1e-4, order=2, n_out=10)
E = [p.params["energy"](y) for y in s.y]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for t, y in zip(s.t[::3], s.y[::3]):
    axes[0].plot(p.x, y, label=f"t = {t:.3f}")
axes[0].legend(fontsize=7); axes[0].set_title("phase separation from a random mixture")
axes[1].plot(s.t, E, "o-"); axes[1].set_xlabel("t"); axes[1].set_ylabel("free energy"); axes[1].set_title("the energy never increases")
plt.show()
lam = stiffness.spectrum(p.rhs, s.y[-1], band=(2, 2))
print(f"mass drift {np.ptp(analysis.conservation(p.grid, s.y)):.1e}; energy monotone: {bool(np.all(np.diff(E) <= 1e-9))}")
print(f"extreme eigenvalue {lam[0].real:.3g} ~ -eps^2 16/dx^4 = {-0.02**2 * 16 / p.grid.dx**4:.3g}: an explicit step would be {2.8 / abs(lam[0].real):.1e}")"""),
    md(r"""
The Cahn-Hilliard equation is the gradient flow of a free energy under the constraint of conserved mass; a
random mixture separates into domains of the two phases that then coarsen. The discretisation conserves mass
to rounding (no-flux ends on a conservative operator) and the energy decreases at every step. The fourth-order
operator makes the system extremely stiff - eigenvalues of order ε²·16/Δx⁴ ≈ −2×10⁶ - so an explicit method
would need steps of 10⁻⁶ where BDF2 takes 10⁻⁴; the Jacobian is pentadiagonal (band 2) and Newton with it is
still cheap.
"""),
]

NOTEBOOKS["06_reaction_diffusion_fronts_and_patterns"] = [
    md(r"""
# 06 · Reaction-diffusion: fronts and patterns

Add a reaction term to diffusion and the solution can organise itself: Fisher's equation produces a front that
invades at a speed fixed by the equation, and the Gray-Scott system produces spots and stripes from noise. These
problems also raise the method-of-lines question of *splitting*: diffusion is stiff and linear, reaction is
nonlinear but not stiff - so treat diffusion implicitly and reaction explicitly (IMEX), and the step is limited
by the reaction alone. This notebook measures the KPP front speed in one and two dimensions, compares BDF with
IMEX, and grows Gray-Scott patterns.
"""),
    setup_cell(["from engmolines import problems as pr, integrators as ti, analysis, stiffness"]),
    md("## Fisher's travelling wave"),
    code(r"""p = pr.fisher()
s = p.solve("bdf", 0.02, order=2, n_out=15)
pos = np.array([pr.front_position(p.x, y) for y in s.y])
speed = np.diff(pos) / np.diff(s.t)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for t, y in zip(s.t[::3], s.y[::3]):
    axes[0].plot(p.x, y, label=f"t = {t:.0f}")
axes[0].legend(fontsize=7); axes[0].set_title("the front invades u = 0")
axes[1].plot(0.5 * (s.t[1:] + s.t[:-1]), speed, "o-"); axes[1].axhline(p.params["speed"], color="k", ls="--", label="2 sqrt(rD)"); axes[1].legend(); axes[1].set_ylim(1.5, 2.1); axes[1].set_title("front speed")
plt.show()
print(f"speed at the end {speed[-1]:.3f}, approaching 2 sqrt(rD) = {p.params['speed']} from below (Bramson: 2 - 3/(2t) + ...)")"""),
    md(r"""
u_t = D u_xx + r u(1 − u) started from a step develops a front moving at the minimal speed 2√(rD), the
Kolmogorov-Petrovsky-Piskunov result: diffusion spreads the leading edge, growth amplifies it, and the balance
selects the speed. The convergence to 2 is slow - Bramson's correction is −3/(2t) - which the measured speeds
reproduce; a quick look at the first few values would report 1.85 and call it an error.

## IMEX: implicit diffusion, explicit reaction
"""),
    code(r"""lam = stiffness.spectrum(p.rhs, s.y[-1], band=(1, 1)); lam = lam[np.abs(lam) > 1e-9]
print(f"spectrum of the full system: {lam[0].real:.0f} to {lam[-1].real:+.3f}; RK4 limit {stiffness.explicit_step_limit(lam, 'rk4'):.4f}")
runs = {"BDF2 (fully implicit)": p.solve("bdf", 0.02, order=2, n_out=15),
        "IMEX CNAB2 (diffusion implicit, reaction explicit)": p.solve("imex", 0.02, n_out=15),
        "IMEX Euler": p.solve("imex", 0.02, n_out=15, order=1)}
for name, r in runs.items():
    print(f"{name:<52}: front at {pr.front_position(p.x, r.y[-1]):.3f}, Newton {r.newton_iterations}, Jacobians {r.jacobian_evaluations}")
big = p.solve("imex", 0.5, n_out=15, order=1, allow_failure=True)
print(f"IMEX Euler at h = 0.5 (the reaction's limit is about 2/r = 2): stable, but first order - the front lags at {pr.front_position(p.x, big.y[-1]):.2f}")"""),
    md(r"""
The IMEX methods treat the linear diffusion implicitly - one factorisation of the tridiagonal Jacobian, and a
Newton iteration that converges in a single step because the implicit part is linear - and the reaction
explicitly, so their step is limited by the reaction rate r, not by 4D/Δx²: even h = 0.5 is stable (though
first-order IMEX Euler is then inaccurate). For problems whose stiffness is linear and whose nonlinearity is
cheap, IMEX removes the nonlinear solve from the stiff part; its second-order variant (Crank-Nicolson for
diffusion, Adams-Bashforth for the reaction) matches BDF2 at half the Newton work.

## A circular front in two dimensions
"""),
    code(r"""p2 = pr.fisher_2d(n=60)
s2 = p2.solve("bdf", 0.05, order=2, n_out=8)
rad = np.array([np.sqrt(np.mean(y > 0.5) * 40.0**2 / np.pi) for y in s2.y])
print("front radius:", np.round(rad, 2)); print("radial speed:", np.round(np.diff(rad) / np.diff(s2.t), 3), " (planar speed 2; a front of radius R moves at 2 - D/R)")
plt.imshow(s2.y[-1].reshape(p2.params["shape"]).T, origin="lower", extent=(0, 40, 0, 40), cmap="viridis"); plt.colorbar(); plt.title(f"Fisher front at t = {s2.t[-1]}"); plt.show()"""),
    md(r"""
In two dimensions the Jacobian has band n (the Kronecker structure of notebook 12) and BDF2 still needs one
Jacobian for the whole run. The circular front moves at the planar speed minus a curvature correction D/R,
so it accelerates towards 2 as it grows - the two-dimensional analogue of the slow approach seen in 1-D.

## Gray-Scott: patterns from noise
"""),
    code(r"""p = pr.gray_scott(n=64, F=0.037, k=0.06)
s = p.solve("rk4", 1.0, t_end=3000.0, n_out=3)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for ax, t, y in zip(axes, s.t, s.y):
    ax.imshow(y[64 * 64 :].reshape(64, 64).T, origin="lower", cmap="magma"); ax.set_title(f"v at t = {t:.0f}"); ax.axis("off")
plt.show()
print(f"explicit RK4 with h = 1: {s.steps} steps; the reaction limits the step (diffusivities 2e-5 and 1e-5 give 4D/dx^2 = {4 * 2e-5 / (1 / 64)**2:.2f}, so diffusion is NOT stiff here)")"""),
    md(r"""
Gray-Scott's two species have tiny diffusivities and comparable reaction rates; the diffusive eigenvalues
(4D/Δx² ≈ 0.3) are no larger than the reaction's, so the system is not stiff and explicit RK4 with h = 1 is the
right tool - 3000 cheap steps rather than 3000 banded solves. Choosing the integrator starts with the spectrum,
not with habit. The spots that emerge from the perturbed seed are the classic Turing-type pattern of this
parameter pair.
"""),
]

NOTEBOOKS["07_advection_the_hard_case"] = [
    md(r"""
# 07 · Advection: the hard case

Advection u_t + a u_x = 0 moves a profile without changing it, and that is exactly what discretisations find
hard: the centred stencil conserves the shape but is unstable with any explicit integrator whose stability
region excludes the imaginary axis; the upwind stencil is stable but smears everything it carries; higher-order
stencils oscillate at jumps. This notebook measures these behaviours, derives them from modified equations,
and shows the way out - flux limiters - before applying it to a reactor with inflow and outflow boundaries.
"""),
    setup_cell(
        [
            "from engmolines import problems as pr, integrators as ti, analysis, stiffness, grids, boundary as bc, operators as op"
        ]
    ),
    md("## Four schemes, one pulse"),
    code(r"""fig, ax = plt.subplots(figsize=(10, 4))
for sch in ("upwind", "centred", "van-leer", "superbee"):
    p = pr.advection_periodic(scheme=sch)
    s = p.solve("dopri", rtol=1e-7, atol=1e-9)
    ax.plot(p.x, s.y[-1], label=f"{sch}: error {p.error(s):.2f}, {s.steps} steps")
ax.plot(p.x, p.exact(p.x, p.t_end), "k--", lw=1, label="exact"); ax.legend(fontsize=8); ax.set_title("a Gaussian pulse after one period"); plt.show()"""),
    md(r"""
After one period the exact solution is the initial pulse. Upwind has flattened it (error 0.55); centred keeps
the height but trails wiggles (0.13); the limited schemes keep both height and shape (0.07-0.10). The adaptive
integrator's step counts also differ: the limited schemes are nonlinear and cost more evaluations.

## Why centred differences fail with explicit Euler, and why upwind smears
"""),
    code(r"""p = pr.advection_periodic(scheme="centred")
lam = stiffness.spectrum(p.rhs, p.y0)
print(f"centred: eigenvalues are purely imaginary (max |Re| = {np.max(np.abs(lam.real)):.1e}), up to +-i a/dx = {1 / p.grid.dx:.0f}")
print(f"   explicit Euler step limit: {stiffness.explicit_step_limit(lam, 'euler'):.1e}; RK4: {stiffness.explicit_step_limit(lam, 'rk4'):.4f} (= 2.83 dx/a)")
pu = pr.advection_periodic(scheme="upwind")
lamu = stiffness.spectrum(pu.rhs, pu.y0)
print(f"upwind: eigenvalues on a circle of radius a/dx centred at -a/dx: real parts from {lamu.real.min():.0f} to {lamu.real.max():.1f}; Euler limit {stiffness.explicit_step_limit(lamu, 'euler'):.4f} (= dx/a, the CFL condition)")
me = analysis.modified_equation("upwind", 1.0, p.grid.dx)
print(f"\nmodified equation of upwind: u_t + a u_x = (a dx/2) u_xx + ...: numerical diffusion {me['diffusion']:.4f}")
s = pu.solve("dopri", rtol=1e-9, atol=1e-11)
sig = np.sqrt(0.05**2 + 4 * me['diffusion'] * 1.0)
print(f"predicted peak after one period 0.05/sigma = {0.05 / sig:.3f}; computed {s.y[-1].max():.3f}")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(lam.real, lam.imag, "."); axes[0].set_title("centred: imaginary axis"); axes[1].plot(lamu.real, lamu.imag, "."); axes[1].set_title("upwind: a circle in the left half-plane")
for ax in axes: ax.set_aspect("equal"); ax.axvline(0, color="k", lw=0.5)
plt.show()"""),
    md(r"""
The centred operator is antisymmetric: its eigenvalues lie on the imaginary axis, where explicit Euler's
stability region has no interior - every mode grows - while RK4's region reaches ±2.83i, so RK4 and
Dormand-Prince are stable up to a Courant number of 2.8 (in the method of lines; the fully discrete CFL limit
is a property of the pair). Upwind's eigenvalues lie on a circle inside the left half-plane: stable with Euler up
to Courant number 1, but every mode decays - the modified equation u_t + a u_x = (aΔx/2) u_xx says upwind
solves the advection-diffusion equation, and the predicted spreading of the Gaussian matches to three digits.

## Flux limiters: second order where it helps, monotone where it must
"""),
    code(r"""g = grids.cell_centred(0, 1, 200)
sq = np.where((g.x > 0.1) & (g.x < 0.3), 1.0, 0.0) + np.exp(-((g.x - 0.65) / 0.05) ** 2)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(g.x, sq, "k--", lw=1, label="exact after one period")
for sch in ("centred", "upwind", "minmod", "van-leer", "superbee"):
    f = lambda t, u, sch=sch: op.advection(u, g, 1.0, bc.Periodic(), bc.Periodic(), t, sch)
    s = ti.dopri45(f, sq, (0, 1.0), 1e-7, 1e-9, n_out=1)
    tv = np.sum(np.abs(np.diff(np.r_[s.y[-1], s.y[-1][0]])))
    ax.plot(g.x, s.y[-1], label=f"{sch}: total variation {tv:.2f} (initial 4.00)")
ax.legend(fontsize=8); plt.show()"""),
    md(r"""
On a square wave plus a Gaussian the centred scheme oscillates (its total variation grows from 4 to over 5),
upwind smears, and the limited schemes are *total-variation diminishing*: they switch to upwind at the jumps and
to second order in the smooth parts. Minmod is the most cautious, superbee the sharpest (it keeps the square
wave almost intact but squares off the Gaussian's peak), van Leer a balance. Godunov's theorem says no linear
scheme can be both second order and monotone; the limiters escape it by being nonlinear.

## Boundaries: an inflow, and a reactor with Danckwerts conditions
"""),
    code(r"""p = pr.advection_inflow()
s = p.solve("dopri", rtol=1e-7, atol=1e-9, n_out=3)
for t, y in zip(s.t[1:], s.y[1:]):
    plt.plot(p.x, y, label=f"t = {t:.1f}"); plt.plot(p.x, p.exact(p.x, t), "k:", lw=0.8)
plt.legend(); plt.title("a sinusoidal inflow enters at x = 0 (dotted: exact)"); plt.show()
print(f"inflow problem: max error {p.error(s):.3f} (one boundary condition only: at the inflow; the outflow needs none)")
p = pr.plug_flow_reactor(Pe=10.0, Da=2.0)
s = p.solve("bdf", 0.01, order=2, n_out=6)
for t, y in zip(s.t, s.y):
    plt.plot(p.x, y, label=f"t = {t:.1f}")
plt.plot(p.x, p.exact(p.x), "k--", label="Danckwerts steady state"); plt.legend(fontsize=7); plt.show()
print(f"outlet conversion {1 - s.y[-1][-1]:.4f} (exact {p.params['conversion']:.4f}); the inlet jump c(0+) < 1 is the Danckwerts condition c - c_x/Pe = 1")"""),
    md(r"""
A hyperbolic equation takes its boundary condition where the characteristics enter: at the inflow only; the
outflow is left alone (a zero-gradient ghost), and imposing a value there would be inconsistent. The dispersed
plug-flow reactor adds a little diffusion and the Danckwerts conditions - a Robin condition at the inlet,
whose signature is the jump of the concentration below the feed value, and zero gradient at the outlet - and
reaches the classic closed-form steady conversion.
"""),
]

NOTEBOOKS["08_shocks_and_conservation_laws"] = [
    md(r"""
# 08 · Shocks and conservation laws

Nonlinear hyperbolic equations - Burgers, shallow water, gas dynamics - steepen smooth waves into shocks, after
which the PDE holds only in its integral (conservation) form and the solution must be selected by an entropy
condition. The method of lines handles them with finite-volume right-hand sides: fluxes through cell faces from
an approximate Riemann solver, reconstruction with limiters, and any time integrator. This notebook follows
Burgers' equation through breaking, checks shock speeds and rarefactions against the exact Riemann solutions,
and solves Stoker's dam break for the shallow-water system.
"""),
    setup_cell(["from engmolines import problems as pr, integrators as ti, analysis"]),
    md("## Burgers' equation: steepening and breaking"),
    code(r"""p = pr.burgers_sine()
s = p.solve("dopri", rtol=1e-8, atol=1e-10, t_end=0.5, n_out=5)
for t, y in zip(s.t, s.y):
    plt.plot(p.x, y, label=f"t = {t:.2f}" + (" (before breaking)" if t < p.params["t_break"] else ""))
plt.legend(fontsize=7); plt.title("u_t + (u^2/2)_x = 0: breaking at t = 1/(2 pi) = 0.159"); plt.show()
s_pre = p.solve("dopri", rtol=1e-8, atol=1e-10, t_end=0.12)
print(f"before breaking, vs the characteristic solution: error {p.error(s_pre):.1e}; after breaking the integral of u is conserved to {p.grid.integrate(s.y[-1]) - p.grid.integrate(p.y0):+.1e}")"""),
    md(r"""
Each value of u travels at speed u, so the crests overtake the troughs and the profile steepens until the
slope becomes infinite at t = 1/(2π); after that a shock forms and moves at the Rankine-Hugoniot speed. Before
breaking the finite-volume solution matches the exact characteristic solution to 10⁻³; after it the
conservation form keeps the integral of u exact.

## Riemann problems: shock and rarefaction
"""),
    code(r"""fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for ax, (uL, uR) in zip(axes, ((1.0, 0.0), (0.0, 1.0))):
    p = pr.burgers_riemann(uL=uL, uR=uR)
    s = p.solve("dopri", rtol=1e-7, atol=1e-9, n_out=4)
    for t, y in zip(s.t[1:], s.y[1:]):
        ax.plot(p.x, y, label=f"t = {t:.1f}"); ax.plot(p.x, p.exact(p.x, t), "k:", lw=0.8)
    ax.set_title(p.name); ax.legend(fontsize=7)
plt.show()
p = pr.burgers_riemann(uL=1.0, uR=0.0); s = p.solve("dopri", rtol=1e-7, atol=1e-9)
print(f"shock at {p.x[np.argmin(np.abs(s.y[-1] - 0.5))]:.4f} vs (uL + uR)/2 * t = {0.5 * 0.8:.4f}; overshoot {s.y[-1].max() - 1:.1e}")"""),
    md(r"""
A shock (uL > uR) travels at the Rankine-Hugoniot speed (uL + uR)/2, here within a cell width and without
overshoot; a rarefaction (uL < uR) fans out as u = x/t. The non-entropy "expansion shock" would also satisfy
the conservation law, and a careless scheme produces it; the Rusanov flux's numerical dissipation selects
the physical fan.

## Inside the finite-volume right-hand side
"""),
    code(r"""from engmolines import operators as op, grids
g = grids.cell_centred(-1, 1, 10)
u = np.where(g.x < 0, 1.0, 0.0)
rate = op.conservation_law(u, g, lambda w: 0.5 * w * w, lambda w: np.abs(w), "transmissive", "transmissive")
print("state     :", u)
print("rate      :", np.round(rate, 3), "  <- only the cell behind the jump changes: the flux difference (F_L - F_R)/dx = (0.5 - 0)/0.2")
print("sum of rates * dx:", f"{np.sum(rate) * g.dx:.2f}", "= flux in through the left boundary F(1) = 0.5 minus flux out F(0) = 0: the total changes only by what crosses the boundaries")"""),
    md(r"""
The rate of each cell is the difference of the fluxes through its two faces divided by its width; each face
flux is computed once from the states on either side, so whatever leaves one cell enters the next and the
total changes only by the fluxes through the domain's boundaries (exactly zero on a periodic grid). The library's Rusanov flux F* = (F(L) + F(R))/2 − s(R − L)/2 uses the largest
local wave speed s as dissipation; MUSCL reconstruction with a limiter raises the order to two away from jumps.

## The shallow-water dam break
"""),
    code(r"""p = pr.shallow_water_dam_break(hL=1.0, hR=0.1)
s = p.solve("dopri", rtol=1e-7, atol=1e-9, n_out=4)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for t, y in zip(s.t[1:], s.y[1:]):
    h, q = y[:400], y[400:]
    axes[0].plot(p.x, h, label=f"t = {t:.2f}"); axes[0].plot(p.x, p.exact(p.x, t)[0], "k:", lw=0.8)
    axes[1].plot(p.x, q / h)
axes[0].set_title("depth h (dotted: Stoker)"); axes[0].legend(fontsize=7); axes[1].set_title("velocity u"); plt.show()
h = s.y[-1][:400]
print(f"plateau depth {np.median(h[(p.x > 0.05) & (p.x < 0.3)]):.4f} (Stoker {p.params['h_m']:.4f}); shock speed {p.params['shock_speed']:.3f}; rarefaction head at -sqrt(g hL) = {-np.sqrt(9.81):.3f}")
print(f"mass conserved to {p.grid.integrate(h) - p.grid.integrate(p.y0[:400]):+.1e}")"""),
    md(r"""
The shallow-water equations are a 2 × 2 system; the same finite-volume operator handles it with vector states,
component-wise reconstruction and the wave speed |u| + √(gh). Removing the dam sends a rarefaction into the
deep water and a shock (a bore) into the shallow, with a plateau between them whose depth Stoker's exact
solution gives; the computed plateau agrees to 10⁻³ and the fronts sit within a cell of their exact positions.
"""),
]

NOTEBOOKS["09_waves_strings_and_beams"] = [
    md(r"""
# 09 · Waves, strings and beams

Second-order-in-time equations - the wave equation, vibrating strings, bending beams - are either rewritten as
first-order systems [u, v] and handed to any integrator, or integrated directly by Stormer-Verlet, which
preserves energy over long times. Their spectra are purely imaginary (undamped) or shifted left by damping;
their stiffness comes from the spatial operator's largest frequency: 2c/Δx for a string, 4/Δx² for a beam.
This notebook checks standing waves against exact solutions, watches energy under different integrators, and
handles damping and a moving load.
"""),
    setup_cell(["from engmolines import problems as pr, integrators as ti, analysis, stiffness"]),
    md("## The vibrating string: RK4 versus Stormer-Verlet"),
    code(r"""p = pr.wave_string(mode=2)
n = 101
T = p.t_end
runs = {"RK4 (first-order system)": ti.explicit_rk(p.rhs, p.y0, (0, 20 * T), 0.005, n_out=40),
        "Stormer-Verlet": ti.verlet(p.params["accel"], p.y0[:n], p.y0[n:], (0, 20 * T), 0.005, n_out=40),
        "BDF2 (dissipative)": ti.bdf(p.rhs, p.y0, (0, 20 * T), 0.005, 2, band=(n + 1, n + 1), n_out=40)}
for name, s in runs.items():
    E = np.array([analysis.wave_energy(y[:n], y[n:], 1.0, p.grid) for y in s.y])
    plt.plot(s.t / T, E / E[0], label=name)
    print(f"{name:<26}: energy change after 20 periods {E[-1] / E[0] - 1:+.2e}; position error {np.max(np.abs(s.y[-1][:n] - p.exact(p.x, s.t[-1]))):.1e}")
plt.xlabel("periods"); plt.ylabel("energy / initial"); plt.legend(); plt.show()"""),
    md(r"""
Over twenty periods RK4's numerical damping is invisible at this step - |R(ihω)| deviates from 1 only at
order (hω)⁶, and hω = 0.03 for the mode present - while BDF2, built for stiff dissipative problems, damps the
wave by 0.2 % and lags it. At this step neither RK4 nor Verlet changes the energy measurably (the 10⁻⁷ is the
discretisation error of the energy sum itself). The difference between them appears over very long runs or at
larger steps: Verlet is symplectic and keeps the energy bounded for any length of run, while RK4's small
dissipation accumulates (the exercises test this). For undamped waves, the right integrators are the ones
made for them.

## Spectra: strings, beams and stiffness
"""),
    code(r"""ps = pr.wave_string(); lam_s = stiffness.spectrum(ps.rhs, ps.y0)
pb = pr.beam_vibration(); lam_b = stiffness.spectrum(pb.rhs, pb.y0)
print(f"string: |lambda|max = {np.max(np.abs(lam_s.imag)):.0f} = 2c/dx = {2 / ps.grid.dx:.0f} (period of the fastest grid mode / period of the solution = {np.max(np.abs(lam_s.imag)) / (2 * np.pi / ps.t_end):.0f})")
print(f"beam:   |lambda|max = {np.max(np.abs(lam_b.imag)):.0f} = 4 sqrt(EI/rhoA)/dx^2 = {4 / pb.grid.dx**2:.0f} (ratio {np.max(np.abs(lam_b.imag)) / pb.params['omega']:.0f}): the beam is far stiffer")
sv = ti.verlet(pb.params["accel"], pb.y0[:61], pb.y0[61:], (0, pb.t_end), 0.4 * pb.grid.dx**2, n_out=1)
print(f"Verlet on the beam at h = 0.4 dx^2 (limit dx^2/2): error after one period {np.max(np.abs(sv.y[-1][:61] - pb.exact(pb.x, pb.t_end))):.1e}; steps {sv.steps}")"""),
    md(r"""
The string's grid modes oscillate 60 times faster than its solution; the beam's 1400 times, because the
biharmonic operator's largest eigenvalue scales as Δx⁻⁴ and frequencies as its square root. Verlet's limit
h ω_max ≤ 2 then forces h ≤ Δx²/2 - 6000 steps for one period of the first mode. Implicit methods exist for
such problems (Newmark's family, the trapezoidal rule with its ringing), but for an accurate undamped
long-time solution the explicit symplectic method with its small step is often the honest choice.

## Damping: the telegraph equation
"""),
    code(r"""p = pr.telegraph(a=0.5)
lam = stiffness.spectrum(p.rhs, p.y0); lam = lam[np.abs(lam) > 1e-9]
print(f"every dynamic eigenvalue has real part {np.unique(np.round(lam.real, 6))} = -a")
s = p.solve("rk4", 0.005, n_out=1)
print(f"RK4: error vs e^(-at) sin(pi x) cos(omega t) = {p.error(s):.1e}")
plt.plot(lam.real, lam.imag, "."); plt.axvline(0, color="k", lw=0.5); plt.title("telegraph: the spectrum shifted left by a"); plt.show()"""),
    md(r"""
Damping shifts the whole spectrum left by a, so every mode decays at the same rate and the oscillation
frequencies become √(c²k² − a²). With dissipation present the choice of integrator is less delicate: RK4 is
accurate, and a dissipative implicit method would add only a little to the physical damping.

## A moving load
"""),
    code(r"""p = pr.string_moving_load(v=0.5)
s = p.solve("rk4", 0.001, n_out=100)
n = 201
E = np.array([analysis.wave_energy(y[:n], y[n:], 1.0, p.grid) for y in s.y])
power = np.array([p.grid.integrate(p.params["load"](t) * y[n:]) for t, y in zip(s.t, s.y)])
work = np.r_[0.0, np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(s.t))]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for i in (25, 50, 75, 100):
    axes[0].plot(p.x, s.y[i][:n], label=f"t = {s.t[i]:.2f}")
axes[0].legend(fontsize=7); axes[0].set_title("deflection under a load moving at v = c/2")
axes[1].plot(s.t, E, label="string energy"); axes[1].plot(s.t, work, "--", label="work done by the load"); axes[1].legend(); plt.show()
print(f"energy balance: max |E - W| / max E = {np.max(np.abs(E - work)) / E.max():.3f}")"""),
    md(r"""
A load crossing a string at half the wave speed leaves waves behind it; the energy the string gains equals the
work the load does, integral F w_t dt at the load's position - a check that the forcing, the energy
bookkeeping and the integrator are all consistent (to 1 % here, the error of the trapezoid rule on the power
history).
"""),
]

NOTEBOOKS["10_dispersive_waves_and_spectral_methods"] = [
    md(r"""
# 10 · Dispersive waves and spectral methods

Dispersive equations - Korteweg-de Vries, nonlinear Schrodinger - have solutions that stay smooth and
periodic domains that make Fourier spectral differentiation natural: multiply the Fourier coefficients by
(ik)ᵐ and the spatial error is exponentially small. The price is stiffness of a new kind: the spectral u_xxx has
eigenvalues up to (πn/L)³ on the imaginary axis, so explicit steps shrink like Δx³. This notebook propagates
solitons with spectral accuracy, collides two of them, checks the invariants of KdV and NLS, and shows what
aliasing does.
"""),
    setup_cell(["from engmolines import problems as pr, integrators as ti, analysis, stiffness, spectral"]),
    md("## Spectral differentiation"),
    code(r"""x = spectral.grid(0, 2 * np.pi, 32)
u = np.exp(np.sin(x))
for order in (1, 2, 3):
    exact = {1: np.cos(x) * u, 2: (np.cos(x) ** 2 - np.sin(x)) * u, 3: (np.cos(x) ** 3 - 3 * np.sin(x) * np.cos(x) - np.cos(x)) * u}[order]
    print(f"derivative {order} of exp(sin x) with 32 points: max error {np.max(np.abs(spectral.derivative(u, 2 * np.pi, order) - exact)):.1e}")
for n in (64, 128, 256):
    print(f"n = {n}, L = 40: spectral u_xxx has |lambda|max = {spectral.max_eigenvalue(n, 40, 3):.0f} -> RK4 step {2.8 / spectral.max_eigenvalue(n, 40, 3):.1e}")"""),
    md(r"""
Thirteen digits from 32 points - against two or three for finite differences - is why dispersive problems
are solved spectrally. The eigenvalues of the spectral third derivative grow like n³: doubling the resolution
divides the explicit step by eight. (Implicit treatment of the linear u_xxx term, an "exponential integrator"
or integrating factor, is the standard remedy; here the adaptive explicit solver is kept for clarity.)

## A KdV soliton
"""),
    code(r"""p = pr.kdv_soliton(kappa=0.8)
s = p.solve("dopri", rtol=1e-8, atol=1e-10, n_out=4)
for t, y in zip(s.t, s.y):
    plt.plot(p.x, y, label=f"t = {t:.0f}")
plt.legend(); plt.title("u_t + 6 u u_x + u_xxx = 0: a soliton"); plt.show()
print(f"after t = 4: error vs 2 kappa^2 sech^2(kappa(x - 4 kappa^2 t - x0)) = {p.error(s):.1e}; {s.steps} steps (mean h {np.mean(s.extra['h']):.1e})")"""),
    md(r"""
The soliton 2κ² sech²(κ(x − 4κ²t)) travels at 4κ² without changing shape: nonlinear steepening and dispersive
spreading balance exactly. The method of lines with spectral space and Dormand-Prince time reproduces it to
10⁻⁸ after four time units. Its mean step of 10⁻⁴ is below even the stability limit of 3×10⁻⁴, because the
tolerance of 10⁻⁸ on a problem with frequencies up to 8000 demands it: here accuracy, not stability, sets the
step.

## Two solitons collide
"""),
    code(r"""p = pr.kdv_two_solitons(k1=1.0, k2=0.6)
s = p.solve("dopri", rtol=1e-8, atol=1e-10, n_out=8)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
for ax, i in zip(axes, (0, 4, 8)):
    ax.plot(p.x, s.y[i]); ax.set_title(f"t = {s.t[i]:.0f}"); ax.set_ylim(-0.1, 2.3)
plt.show()
mass = analysis.conservation(p.grid, s.y); mom = np.array([p.grid.integrate(y**2) for y in s.y])
print(f"mass drift {np.ptp(mass):.1e}, momentum drift {np.ptp(mom):.1e}; peaks after the collision {np.sort(s.y[-1])[-1]:.4f} (2 k1^2 = 2.0)")"""),
    md(r"""
The taller, faster soliton overtakes the shorter one; during the overlap the profile is not a sum of the two,
yet both emerge unchanged except for a phase shift - the defining property of solitons. The integrals of u and
u² (two of KdV's infinitely many invariants) are conserved to 10⁻¹⁴: the spectral operator is exactly
antisymmetric, and the adaptive integrator's error is far below the invariants' sensitivity.

## Nonlinear Schrodinger: a complex field
"""),
    code(r"""p = pr.nls_soliton(v=0.5)
s = p.solve("dopri", rtol=1e-8, atol=1e-10, n_out=4)
n = 256
for t, y in zip(s.t, s.y):
    plt.plot(p.x, np.abs(y[:n] + 1j * y[n:]), label=f"t = {t:.0f}")
plt.legend(); plt.title("|psi| of the NLS soliton"); plt.show()
psi = s.y[-1][:n] + 1j * s.y[-1][n:]
print(f"error vs the exact soliton {p.error(s):.1e}; mass integral |psi|^2 drift {p.grid.integrate(np.abs(psi)**2) - p.params['mass']:+.1e}")"""),
    md(r"""
The complex field is split into real and imaginary parts to form a real ODE system; the exact soliton
sech(x − 2vt) e^{i(vx + (1 − v²)t)} is reproduced to 10⁻⁷, and the mass ∫|ψ|² is conserved.

## Aliasing
"""),
    code(r"""x = spectral.grid(0, 2 * np.pi, 32)
u = np.cos(12 * x)
prod = u * u                                                   # contains the mode 24 > n/2 = 16: it aliases to mode 8
coeffs = np.abs(np.fft.fft(prod)) / 32
print(f"u = cos(12x) on 32 points; u^2 should have modes 0 and 24, but the FFT of the product shows mode {np.argmax(coeffs[1:16]) + 1} with amplitude {coeffs[8]:.2f} (aliased from 24)")
ud = spectral.dealias(u)
print(f"after the 2/3 rule (modes above {32 // 3} removed) the product of dealiased fields cannot alias into the kept range: cos(12x) itself is removed ({np.max(np.abs(ud)):.1e})")"""),
    md(r"""
A product of two resolved modes can produce a wavenumber the grid cannot represent, which folds back onto a
low mode - aliasing - and in a nonlinear simulation feeds energy into the wrong scales. The 2/3 rule keeps only
modes below n/3, so that no product of kept modes aliases into the kept range. The KdV runs above were
resolved well enough (the solitons' spectra decay exponentially) that aliasing was invisible; for turbulent or
under-resolved fields it is not.
"""),
]

NOTEBOOKS["11_coupled_systems_and_daes"] = [
    md(r"""
# 11 · Coupled systems and DAEs

Engineering models rarely come as a single PDE: a fluid phase exchanges with a solid phase at every point, two
streams flow in opposite directions, a reaction couples species. The method of lines stacks all the unknowns
into one state vector; the only new question is the band structure of the Jacobian. And when some equations
have no time derivative - boundary conditions, equilibrium relations, algebraic constraints - the system is a
differential-algebraic equation, which the BDF family handles directly. This notebook works through an
adsorption column, a catalyst pellet, a counterflow exchanger, and the heat equation as a DAE.
"""),
    setup_cell(["from engmolines import problems as pr, integrators as ti, analysis, linalg as la"]),
    md("## A packed-bed adsorber: PDE + ODE at every point"),
    code(r"""p = pr.packed_bed_adsorber(K=5.0)
s = p.solve("bdf", 0.02, order=2, n_out=60)
n = 101
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for i in range(0, 61, 12):
    axes[0].plot(p.x, s.y[i][:n], label=f"t = {s.t[i]:.1f}")
axes[0].set_title("fluid concentration c(x, t)"); axes[0].legend(fontsize=7)
axes[1].plot(s.t, s.y[:, n - 1]); axes[1].axvline(p.params["t_stoich"], color="k", ls="--", label="stoichiometric time"); axes[1].set_title("breakthrough curve c_out(t)"); axes[1].legend()
plt.show()
fed = p.params["u"] * s.t[-1]; out = np.sum(0.5 * (s.y[1:, n - 1] + s.y[:-1, n - 1]) * p.params["u"] * np.diff(s.t))
held = p.grid.integrate(s.y[-1][:n]) + p.params["ratio"] * p.grid.integrate(s.y[-1][n:])
print(f"breakthrough at c_out = 0.5: t = {np.interp(0.5, s.y[:, n - 1], s.t):.2f} (stoichiometric {p.params['t_stoich']:.2f}); mass balance fed {fed:.3f} = out {out:.3f} + held {held:.3f} (residual {fed - out - held:+.1e})")
print(f"state size {2 * n}, Jacobian band {p.band[0]} (q_i couples to c_i at index i + n): {s.jacobian_evaluations} Jacobian for {s.steps} steps")"""),
    md(r"""
The column's state is [c, q]: the fluid concentration obeys an advection-diffusion PDE with a sink, the solid
loading an ODE at each point. Stacked, the Jacobian has a band of n + 2 - the coupling between c_i and q_i sits
n positions off the diagonal - still cheap. The breakthrough front arrives at the stoichiometric time
L(1 + (1 − ε)K/ε)/u that a mass balance predicts, and the overall balance (fed = out + held) closes.

## A catalyst pellet: steady effectiveness from a transient
"""),
    code(r"""p = pr.catalyst_pellet(phi=3.0)
s = p.solve("bdf", 0.005, order=2, n_out=5)
for t, y in zip(s.t, s.y):
    plt.plot(p.x, y, label=f"t = {t:.1f}")
plt.plot(p.x, p.exact(p.x), "k--", label="sinh(phi r)/(r sinh phi)"); plt.legend(fontsize=7); plt.xlabel("r"); plt.show()
c = s.y[-1]; flux = (3 * c[-1] - 4 * c[-2] + c[-3]) / (2 * p.grid.dx)
print(f"effectiveness factor from the surface flux: {3 * flux / p.params['phi']**2:.4f} vs 3 (phi coth phi - 1)/phi^2 = {p.params['eta']:.4f}")"""),
    md(r"""
Diffusion and first-order reaction in a sphere: the steady profile sinh(φr)/(r sinh φ) is reached from an
empty start, and the effectiveness factor - reaction rate relative to a pellet at surface concentration -
follows from the surface flux (a second-order one-sided difference) to 0.1 %.

## Counterflow: two streams, two upwind directions
"""),
    code(r"""p = pr.counterflow_exchanger(n=400, NTU=2.0, Cr=0.5)
s = p.solve("bdf", 0.01, order=2)
n = 401
plt.plot(p.x, s.y[-1][:n], label="hot (flows right)"); plt.plot(p.x, s.y[-1][n:], label="cold (flows left)"); plt.legend(); plt.title("steady temperature profiles"); plt.show()
print(f"effectiveness 1 - T_h(1) = {1 - s.y[-1][n - 1]:.4f} vs the NTU relation {p.params['effectiveness']:.4f}")"""),
    md(r"""
Each stream is an advection equation with its own upwind direction and its own inlet; the exchange term
couples them. A detail worth remembering: a source term added *after* an operator has imposed a Dirichlet
inlet must not move the inlet - the problem re-pins the inlet rows - or the steady state is wrong (the
effectiveness came out at 0.92 instead of 0.77 before that line was added).

## The heat equation as a DAE
"""),
    code(r"""p = pr.heat_as_dae()
sd = ti.dae_bdf(p.params["F"], p.y0, (0, 1.0), 0.005, order=2, band=(1, 1))
sr = ti.bdf(p.rhs, p.y0, (0, 1.0), 0.005, 2, band=(1, 1))
print(f"DAE formulation vs ghost-point ODE formulation: max difference {np.max(np.abs(sd.y[-1] - sr.y[-1])):.1e}")
print(f"the algebraic equation u_0 - sin(pi t) = 0 holds at every output to {np.max(np.abs(sd.y[:, 0] - np.sin(np.pi * sd.t))):.1e}")
print(sd, "\n(this simple DAE solver refreshes its Jacobian at every step)")"""),
    md(r"""
Writing the boundary condition as an algebraic equation 0 = u₀ − g(t) and solving the whole system
F(t, y, y′) = 0 with a fully implicit BDF is the DAE view of the method of lines: boundary conditions, algebraic
constraints and equilibrium relations all become rows without derivatives. For an index-1 system (the
algebraic variables are determined by the differential ones) BDF converges at its usual order, and the
constraint is satisfied exactly at every step - the formulation to reach for when a boundary condition is
implicit, nonlinear, or couples several unknowns.
"""),
]

NOTEBOOKS["12_two_dimensions"] = [
    md(r"""
# 12 · Two dimensions

Nothing in the method of lines changes in two dimensions except the size of the system and the shape of its
Jacobian: an n × n grid gives n² unknowns, and the five-point Laplacian couples each point to its neighbours
in both directions, so the Jacobian has a band of width n (the Kronecker structure). Banded elimination still
costs only n² · n² = n⁴ flops per factorisation - far below the n⁶ of a dense one - and the implicit methods
remain practical. This notebook solves heat in a plate with the Kronecker band, grows a circular reaction
front, and finishes with the lid-driven cavity, a genuine fluid-dynamics computation whose steady state is
compared with the classic benchmark.
"""),
    setup_cell(["from engmolines import problems as pr, integrators as ti, stencils as st, stiffness, grids"]),
    md("## The Kronecker structure"),
    code(r"""Ax = st.diff_operator(np.linspace(0, 1, 6), 2, 2)
Ay = st.diff_operator(np.linspace(0, 1, 5), 2, 2)
apply2, band = st.kron_sum(Ax, Ay)
K = np.kron(Ax.to_dense(), np.eye(5)) + np.kron(np.eye(6), Ay.to_dense())
plt.spy(np.abs(K) > 1e-12, markersize=4); plt.title(f"2-D Laplacian on a 6 x 5 grid: band {band}"); plt.show()
print("the x-stencil couples point (i, j) to (i +- 1, j), which sits ny = 5 positions away in the flattened vector: band = ny * (half-width of Ax including its closures)")"""),
    md(r"""
Flattening the field row by row, a neighbour in y is one position away and a neighbour in x is n_y positions
away, so the 2-D operator A_x ⊗ I + I ⊗ A_y has a band of n_y times the x-stencil's half-width. `kron_sum`
applies it without forming the matrix (two 1-D operator applications on the reshaped field) and reports the
band for the implicit solvers.

## Heat in a plate
"""),
    code(r"""p = pr.plate_heat_2d(n=24)
s = p.solve("bdf", 1e-3, order=4, n_out=1)
print(f"{p.name}: {p.n} unknowns, band {p.band}; error vs sin(pi x) sin(2 pi y) exp(-5 pi^2 t) = {p.error(s):.1e}; {s.jacobian_evaluations} Jacobian, {s.newton_iterations / s.steps:.1f} Newton per step")
errs = [pr.plate_heat_2d(n=n).error(pr.plate_heat_2d(n=n).solve("bdf", 5e-4, order=4)) for n in (12, 24)]
print(f"spatial order {np.log2(errs[0] / errs[1]):.2f}")
sd = p.solve("dopri", rtol=1e-8, atol=1e-10, n_out=1)
print(f"Dormand-Prince for comparison: {sd.steps} steps (stability-limited near 3.3 dx^2/8 = {3.3 * p.grid.gx.dx**2 / 8:.1e}), error {p.error(sd):.1e}")
plt.imshow(s.y[-1].reshape(p.params["shape"]).T, origin="lower", cmap="coolwarm"); plt.title("u at t = 0.05"); plt.colorbar(); plt.show()"""),
    md(r"""
The 625-point plate is solved by BDF4 with one banded Jacobian of width 25 for the whole run, about two
Newton iterations per step, and second-order spatial accuracy. In 2-D the explicit limit is about 3.3Δx²/(8κ)
- the two-directional Laplacian has eigenvalues down to −8κ/Δx² - and the adaptive explicit solver takes
more steps than BDF4 for a run of 0.05; the gap widens as the grid is refined.

## A circular reaction front
"""),
    code(r"""p = pr.fisher_2d(n=60)
s = p.solve("bdf", 0.05, order=2, n_out=4)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for ax, t, y in zip(axes, s.t[1:], s.y[1:]):
    ax.imshow(y.reshape(p.params["shape"]).T, origin="lower", extent=(0, 40, 0, 40), cmap="viridis", vmin=0, vmax=1); ax.set_title(f"t = {t:.0f}"); ax.axis("off")
plt.show()
rad = np.array([np.sqrt(np.mean(y > 0.5) * 40.0**2 / np.pi) for y in s.y])
print(f"front radius {np.round(rad, 2)}; speeds {np.round(np.diff(rad) / np.diff(s.t), 2)} -> approaching the planar 2 sqrt(rD) = 2 as the curvature correction D/R fades")
print(f"{p.n} unknowns, band {p.band[0]}: {s.jacobian_evaluations} Jacobian, {s.newton_iterations / s.steps:.1f} Newton per step")"""),
    md(r"""
The 3721-unknown system has a band of 61 and BDF2 needs several Newton iterations per step (the nonlinear
reaction) with one Jacobian for the run. The front's speed increases with its radius as 2 − D/R: curvature
costs speed.

## The lid-driven cavity
"""),
    code(r"""p = pr.cavity_flow(n=32, Re=100.0)
s = p.solve("rk4", 0.004, n_out=6)
n = 33
omega = s.y[-1].reshape(n, n)
psi = p.params["streamfunction"](omega)
u, v = p.params["velocity"](psi)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].contour(p.x, p.x, psi.T, 20, cmap="RdBu"); axes[0].set_aspect("equal"); axes[0].set_title("streamlines")
axes[1].plot(u[16, :], p.x, label="u on the vertical centreline (computed)")
ghia_y = [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0]
ghia_u = [0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.0]
axes[1].plot(ghia_u, ghia_y, "ko", ms=4, label="Ghia et al. 1982"); axes[1].legend(fontsize=8); axes[1].set_xlabel("u"); axes[1].set_ylabel("y")
axes[2].semilogy(s.t[1:], [np.max(np.abs(s.y[i] - s.y[i - 1])) / np.max(np.abs(s.y[-1])) for i in range(1, len(s.t))], "o-"); axes[2].set_title("approach to steady state"); axes[2].set_xlabel("t")
plt.show()
print(f"u_min on the vertical centreline {u[16, :].min():.4f} at y = {p.x[np.argmin(u[16, :])]:.3f} (Ghia: -0.2109 at 0.4531); primary vortex psi_min {psi.min():.4f} (Ghia: -0.1034)")"""),
    md(r"""
In vorticity-streamfunction form the Navier-Stokes equations become one transport equation for the vorticity
(advected by the velocity, diffused by 1/Re) and a Poisson equation for the streamfunction solved at every
right-hand-side evaluation (its operator is factorised once); the velocity follows from the streamfunction and
the wall vorticity from Thom's formula. At Re = 100 on a 32 × 32 grid, explicit RK4 reaches the steady state
in a few thousand cheap steps - the viscous time scale Δx²Re/4 ≈ 0.016 limits the step, but the Poisson solve
makes implicit treatment unattractive - and the centreline velocity and vortex strength agree with Ghia,
Ghia and Shin's benchmark within the few percent expected of this grid.
"""),
]

NOTEBOOKS["13_diagnosing_failures"] = [
    md(r"""
# 13 · Diagnosing failures

Method-of-lines computations fail in recognisable ways, and each failure has a test that identifies it and a
remedy. This notebook stages eight of them deliberately - instability from a non-dissipative stencil with the
wrong integrator, a boundary closure that puts eigenvalues in the right half-plane, ringing of a non-L-stable
method on rough data, negative concentrations from an oscillatory scheme feeding a nonlinear rate, a domain
too small for the solution, aliasing in an under-resolved spectral run, a source added after the operator
pinned a boundary, and a Newton iteration that converged to a wrong answer - and shows the symptom, the
diagnosis and the fix for each.
"""),
    setup_cell(
        [
            "from engmolines import grids, boundary as bc, operators as op, integrators as ti, stencils as st, stiffness, analysis, problems as pr, spectral"
        ]
    ),
    md("## 1. Growth from the imaginary axis: centred advection + explicit Euler"),
    code(r"""p = pr.advection_periodic(scheme="centred")
s = ti.explicit_rk(p.rhs, p.y0, (0, 0.5), p.grid.dx, ti.EULER, n_out=5)
print("max |u| per output:", np.round([np.max(np.abs(y)) for y in s.y], 2), " <- every mode grows, the sawtooth fastest")
lam = stiffness.spectrum(p.rhs, p.y0)
print(f"diagnosis: the spectrum is purely imaginary (max |Re| = {np.max(np.abs(lam.real)):.0e}) and Euler's stability region has no interior on the imaginary axis -> use RK4/Dormand-Prince (CFL 2.8), or an upwind/limited stencil")"""),
    md(r"""
The symptom is a sawtooth that grows without bound; the test is the spectrum of the right-hand side (purely
imaginary) against the integrator's stability region (which must enclose a piece of the imaginary axis). RK4
does, Euler and Heun do not; or change the stencil to one with dissipation.

## 2. A boundary closure with eigenvalues in the right half-plane
"""),
    code(r"""g = grids.cell_centred(0, 1, 100)
def rhs_bad(t, u):                                      # centred advection with an extrapolated ghost at the *inflow*: the wrong closure for a > 0
    up = np.r_[2 * u[0] - u[1], u, u[-1]]               # left ghost by extrapolation, right ghost zero-gradient
    return -(up[2:] - up[:-2]) / (2 * g.dx)
lam = stiffness.spectrum(rhs_bad, np.zeros(100))
print(f"largest real part of the spectrum: {lam.real.max():+.2f}  <- a growing boundary mode")
s = ti.dopri45(rhs_bad, np.exp(-((g.x - 0.3) / 0.05) ** 2), (0, 2.0), 1e-6, 1e-9, n_out=1, max_steps=20000)
print(f"after t = 2 (the pulse has long left the domain): max |u| = {np.max(np.abs(s.y[-1])):.1e}")
good = lambda t, u: op.advection(u, g, 1.0, bc.Dirichlet(0.0), bc.Neumann(0.0), t, "centred")   # noqa: E731
lam_g = stiffness.spectrum(good, np.zeros(100))
print(f"with the inflow value imposed instead: largest real part {lam_g.real.max():+.2e}")"""),
    md(r"""
A boundary treatment inconsistent with the direction of information flow puts an eigenvalue in the right
half-plane: the solution looks fine while the pulse is inside and then grows from the boundary, exponentially
and at any step size. The diagnosis is again the spectrum - a positive real part is never a time-step problem -
and the fix is the correct inflow condition.

## 3. Ringing: rough data and a method that is not L-stable
"""),
    code(r"""g = grids.uniform(0, 1, 40)
f = lambda t, u: op.diffusion(u, g, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t)   # noqa: E731
u0 = np.where((g.x > 0.4) & (g.x < 0.6), 1.0, 0.0)
tz = ti.trapezoidal(f, u0, (0, 0.05), 0.01, band=(1, 1), n_out=1)
be = ti.bdf(f, u0, (0, 0.05), 0.01, 1, band=(1, 1), n_out=1)
print(f"trapezoidal: min value {tz.y[-1].min():+.3f} (negative temperatures from a non-negative start); backward Euler: min {be.y[-1].min():+.4f}")
print(f"diagnosis: h lambda_max = {0.01 * (-4 / g.dx**2):.0f}, |R| = {abs((1 - 32) / (1 + 32)):.2f} for the trapezoidal rule: the sawtooth is not damped -> L-stable method, or two backward-Euler steps first (Rannacher)")"""),
    md(r"""
Negative values from a non-negative start, oscillating from step to step, are the signature of a non-L-stable
method on data with high-frequency content. The test is |R(hλ_max)| near 1; the remedy is BDF, Radau, or a
Rannacher start.

## 4. Negative concentrations feeding a nonlinear rate
"""),
    code(r"""g = grids.cell_centred(0, 1, 200)
front = np.where(g.x < 0.3, 1.0, 0.0)
for sch in ("centred", "van-leer"):
    f = lambda t, c, sch=sch: op.advection(c, g, 1.0, bc.Periodic(), bc.Periodic(), t, sch) - 50.0 * c * (c - 1) * (c + 0.1)   # a cubic rate with a root at -0.1
    s = ti.dopri45(f, front, (0, 0.3), 1e-6, 1e-9, n_out=1)
    print(f"{sch:<9}: min c = {s.y[-1].min():+.3f}, max c = {s.y[-1].max():.3f}")"""),
    md(r"""
A centred scheme's undershoot takes the concentration below zero; a rate law evaluated there can become a
source (here the cubic flips sign below −0.1) and the "negative mass" grows. A monotone scheme (upwind or a
limiter) keeps the state in its physical range, which nonlinear rates require.

## 5. A domain too small for the solution
"""),
    code(r"""p = pr.porous_medium(n=200)
g_small = grids.cell_centred(-3, 3, 120)
f_small = lambda t, u: op.diffusion(np.maximum(u, 0) ** 2, g_small, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t)   # noqa: E731
s = ti.bdf(f_small, p.exact(g_small.x, 0.0), (0, 1.0), 2e-3, 2, band=(1, 1), n_out=1)
print(f"domain [-3, 3]: mass changed by {g_small.integrate(s.y[-1]) - g_small.integrate(p.exact(g_small.x, 0.0)):+.3f} (the exact support reaches |x| = {np.sqrt(12 * 1.1**(2/3)):.2f}); on [-5, 5] the drift is 1e-13")
print("diagnosis: track the integral of u (or the value at the boundary) over time; a conserved quantity that changes is leaving through the boundary")"""),
    md(r"""
When the solution reaches a boundary that was meant to be far away, the Dirichlet wall absorbs it and a
conserved integral drifts. Monitor conservation and the boundary values; enlarge the domain, or use a
transparent condition.

## 6. Aliasing in an under-resolved spectral run
"""),
    code(r"""for n in (64, 80, 256):
    p = pr.kdv_two_solitons(n=n)
    with np.errstate(all="ignore"):                       # the under-resolved run is allowed to blow up
        s = p.solve("dopri", rtol=1e-8, atol=1e-10, n_out=1, allow_failure=True)
    mom = p.grid.integrate(s.y[-1] ** 2) - p.params["momentum"]
    print(f"n = {n:3d}: reached t = {s.t[-1]:.1f} ({s.message}); momentum invariant changed by {mom:+.2e}; max |u| {np.max(np.abs(s.y[-1])):.2g} (solitons' peak 2.0)")"""),
    md(r"""
With too few points the product u u_x aliases energy into the resolved modes: at n = 80 the momentum
invariant drifts by 4×10⁻³ and the taller soliton's peak is wrong, at n = 64 the aliased energy feeds back
until the run blows up, and at n = 256 the invariants are conserved to 10⁻¹⁴. The test is the invariants (or the tail of the spectrum,
which must decay to rounding); the fix is resolution or the 2/3 rule.

## 7. A source added after the operator pinned a boundary
"""),
    code(r"""g = grids.uniform(0, 1, 50)
def rhs_wrong(t, T):
    return op.advection(T, g, 1.0, bc.Dirichlet(1.0), bc.Neumann(0.0), t, "upwind") - 2.0 * T     # the sink also acts on the inlet node
def rhs_right(t, T):
    out = op.advection(T, g, 1.0, bc.Dirichlet(1.0), bc.Neumann(0.0), t, "upwind") - 2.0 * T
    out[0] = 0.0
    return out
for name, f in (("source after operator", rhs_wrong), ("inlet re-pinned", rhs_right)):
    s = ti.bdf(f, np.r_[1.0, np.zeros(50)], (0, 5.0), 0.01, 2, band=(1, 1), n_out=1)
    print(f"{name:<22}: inlet value at the end {s.y[-1][0]:.4f} (imposed 1), outlet {s.y[-1][-1]:.4f} (exact e^-2 = {np.exp(-2):.4f})")"""),
    md(r"""
The operator sets the inlet node's rate to zero; a source added afterwards moves it anyway, and the whole
profile shifts (the exchanger of notebook 11 came out at 0.92 instead of 0.77 for this reason). The test is to
print the boundary values over time; the fix is to re-pin them after assembling the right-hand side - or to
pass the source into the operator.

## 8. Newton converged - to the wrong answer
"""),
    code(r"""p = pr.nonlinear_conductivity()
s = p.solve("bdf", 0.02, order=2)
print(f"BDF2 at h = 0.02: Newton converged at every step; final state min {s.y[-1].min():.3f} (the conductivity 1 + u is {1 + s.y[-1].min():.1e} there), steady-state error {np.max(np.abs(s.y[-1] - p.exact(p.x))):.2f}")
s2 = p.solve("bdf", 0.005, order=2)
print(f"BDF2 at h = 0.005: steady-state error {np.max(np.abs(s2.y[-1] - p.exact(p.x))):.1e}")"""),
    md(r"""
A nonlinear implicit equation can have several roots, and a large step from rough data can put Newton in the
basin of a non-physical one - here a state where the conductivity vanishes, which the discrete equations
accept as an equilibrium. Convergence proves nothing by itself; a physical check (positivity, bounds,
conservation) and a step-size refinement study do. The library's fixed-step integrators report Newton
failures, but a converged wrong root they cannot see.
"""),
]

NOTEBOOKS["14_moving_boundaries_and_transformed_domains"] = [
    md(r"""
# 14 · Moving boundaries and transformed domains

Some problems do not live on a fixed grid: a melting front moves, an option's price is natural in the
logarithm of the asset, a boundary layer needs points where the solution turns. The method of lines absorbs
these through changes of variable - the moving boundary becomes an extra ODE on a fixed computational domain
(Landau's transformation), a transformed coordinate turns a variable-coefficient equation into a constant
one, and a stretched grid puts resolution where it is needed. This notebook works through Stefan melting,
Black-Scholes, and a boundary layer on uniform versus stretched grids.
"""),
    setup_cell(
        [
            "from engmolines import problems as pr, integrators as ti, grids, boundary as bc, operators as op, stencils as st"
        ]
    ),
    md("## Stefan melting: the Landau transformation"),
    code(r"""p = pr.stefan(St=1.0)
s = p.solve("dopri", rtol=1e-8, atol=1e-10, n_out=5)
lam, t0 = p.params["lambda"], p.params["t0"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for t, y in zip(s.t, s.y):
    s_front = y[-1]
    axes[0].plot(p.x * s_front, y[:-1], label=f"t = {t + t0:.2f}, s = {s_front:.3f}")
axes[0].legend(fontsize=7); axes[0].set_xlabel("physical x"); axes[0].set_title("temperature in the liquid")
tt = np.linspace(t0, 1 + t0, 100)
axes[1].plot(s.t + t0, s.y[:, -1], "o", label="computed front"); axes[1].plot(tt, 2 * lam * np.sqrt(tt), label="2 lambda sqrt(t)"); axes[1].legend(); axes[1].set_xlabel("t"); axes[1].set_title("front position")
plt.show()
print(f"front error after t = 1: {abs(s.y[-1, -1] - 2 * lam * np.sqrt(1 + t0)):.1e}; lambda from sqrt(pi) lambda e^(lambda^2) erf(lambda) = 1/St: {lam:.5f}")"""),
    md(r"""
The liquid occupies 0 < x < s(t) with the front moving at the rate the heat flux into it dictates (the Stefan
condition). Mapping x to ξ = x/s(t) ∈ [0, 1] fixes the domain and adds the convective term ξ(ṡ/s) T_ξ and one
ODE for s - a standard method-of-lines system whose solution matches Neumann's similarity solution 2λ√t to
10⁻⁶. The front position is a state variable like any other, and the adaptive integrator treats it as such.

## Black-Scholes: a heat equation in disguise
"""),
    code(r"""p = pr.black_scholes(n=400, K=100.0, r=0.05, sigma=0.2, T=1.0)
s = p.solve("bdf", 1e-3, order=2, n_out=4)
for tau, y in zip(s.t, s.y):
    plt.plot(p.x, y, label=f"tau = {tau:.2f}")
plt.xlim(50, 160); plt.ylim(-2, 70); plt.xlabel("asset price S"); plt.ylabel("call value"); plt.legend(); plt.show()
print(f"max error vs the Black-Scholes formula {p.error(s):.2e}; at the money C(100) = {np.interp(100.0, p.x, s.y[-1]):.4f} (exact {p.exact(np.array([100.0]), 1.0)[0]:.4f})")
print("the far boundary follows S - K e^(-r tau): a time-dependent Dirichlet value, its rate dg/dtau supplied to the boundary node")"""),
    md(r"""
Written in time-to-expiry, the Black-Scholes equation is a parabolic equation with variable coefficients
½σ²S² and rS - a heat equation in the variable log S. The payoff's kink at S = K is smoothed instantly, as
rough data in a diffusion problem always are, and the method of lines prices the option to a few 10⁻³ of its
value with the plain finite-difference operator. (A log-price grid or a stretched grid around the strike would
do better per point; the exercise asks for it.)

## A boundary layer: uniform versus stretched grids
"""),
    code(r"""Pe = 200.0                                               # advection-diffusion u_t = -u_x + (1/Pe) u_xx, u(0) = 0, u(1) = 1: steady boundary layer of width 1/Pe at x = 1
exact = lambda x: (np.exp(Pe * (x - 1)) - np.exp(-Pe)) / (1 - np.exp(-Pe))   # noqa: E731
for name, g in (("uniform, 60 points", grids.uniform(0, 1, 60)), ("stretched towards x = 1, 60 points", grids.stretched(0, 1, 60, cluster=0.97, strength=2.5))):
    D1, D2 = st.diff_operator(g.x, 1, 2), st.diff_operator(g.x, 2, 2)
    def f(t, u, D1=D1, D2=D2):
        out = -D1(u) + D2(u) / Pe
        out[0] = out[-1] = 0.0
        return out
    u0 = np.zeros(g.n); u0[-1] = 1.0
    s = ti.bdf(f, u0, (0, 3.0), 0.01, 2, band=(3, 3), n_out=1)
    print(f"{name:<36}: smallest spacing {np.diff(g.x).min():.4f} (layer width 1/Pe = {1 / Pe}); steady error {np.max(np.abs(s.y[-1] - exact(g.x))):.2e}; min value {s.y[-1].min():+.3f}")
    plt.plot(g.x, s.y[-1], ".-", ms=4, label=name)
xx = np.linspace(0, 1, 500); plt.plot(xx, exact(xx), "k--", lw=1, label="exact"); plt.xlim(0.9, 1); plt.legend(); plt.show()"""),
    md(r"""
With Pe = 200 the steady boundary layer at x = 1 is 0.005 wide; sixty uniform points (spacing 0.017) cannot
resolve it, and the centred stencil answers with the familiar oscillation (cell Peclet number above 2, a
negative minimum). Sixty points stretched towards the wall put several inside the layer and resolve it. The
operators take the non-uniform spacing in their stride - `diff_operator` builds Fornberg weights at each point
- and the integrator never knows.
"""),
]

NOTEBOOKS["15_project_runaway_in_a_tubular_reactor"] = [
    md(r"""
# 15 · Project: runaway in a tubular reactor

**Problem.** A wall-cooled tubular reactor runs a first-order exothermic reaction. Three steady temperature
profiles were measured along it, at inlet (and wall) temperatures of 310, 320 and 330 K, with eleven
thermocouples - one of them dead in the second run. The rate constants k₀ and E of the Arrhenius law are not
known well enough to say how hot the reactor can be run. The task: build the method-of-lines model, fit k₀
and E from the three profiles with the model inside the least-squares loop, quantify how well they are
determined, locate the ignition (runaway) boundary by parametric sensitivity, show the transient of a runaway,
and report - with the stiffness analysis that says why the model is integrated the way it is.
"""),
    setup_cell(
        [
            "from scipy import optimize as sopt",
            "from engmolines import problems as pr, integrators as ti, stiffness, datasets",
        ]
    ),
    md("## 1. The data"),
    code(r"""data = pd.read_csv(datasets.path("reactor_profiles.csv"), comment="#")
dead = data.T_K < 0
print(f"{len(data)} readings; dropped {dead.sum()} dead thermocouple (run {data.T_in_K[dead].iloc[0]:.0f} K, thermocouple {data.thermocouple[dead].iloc[0]})")
data = data[~dead].reset_index(drop=True)
for T_in, grp in data.groupby("T_in_K"):
    plt.plot(grp.x_m, grp.T_K, "o-", label=f"T_in = {T_in:.0f} K")
plt.xlabel("x (m)"); plt.ylabel("T (K)"); plt.legend(); plt.title("measured steady profiles"); plt.show()"""),
    md("## 2. The model, and its steady state as a transient to t = 80 s"),
    code(r"""p = pr.tubular_reactor(T_in=330.0)
s = p.solve("bdf", 0.5, order=2, n_out=8)
n = p.n // 2
for t, y in zip(s.t, s.y):
    plt.plot(p.x, y[n:], color=plt.cm.viridis(t / s.t[-1]))
plt.xlabel("x (m)"); plt.ylabel("T (K)"); plt.title("approach to the steady state at T_in = 330 K (true parameters)"); plt.show()
print(s)
print(f"residence time L/u = {2.0 / 0.2:.0f} s; the states at t = 60 and t = 80 differ by {np.max(np.abs(s.y[-1] - s.y[-3])):.1e} K: steady")"""),
    md(r"""
The model is two coupled advection-diffusion-reaction equations (notebook 11's structure) with Danckwerts
inlets. Its steady state is reached by integrating the transient - the natural method-of-lines route to a
steady state, and the one that also yields the dynamics - with BDF2 at 0.5 s steps, far above the explicit
limit, using one banded Jacobian of width n + 2.

## 3. Why implicit: the spectrum
"""),
    code(r"""lam = stiffness.spectrum(p.rhs, s.y[-1], band=p.band); lam = lam[np.abs(lam) > 1e-9]
print(f"eigenvalues at the steady state: real parts from {lam.real.min():.0f} to {lam.real.max():+.3f} 1/s; stiffness ratio {stiffness.stiffness_ratio(lam):.0f}")
print(f"explicit RK4 limit {stiffness.explicit_step_limit(lam, 'rk4'):.4f} s against BDF's 0.5 s; the slowest mode has a time constant of {-1 / lam.real.max():.2f} s, so the approach to steady state is set by the residence time, not by an eigenvalue")
plt.plot(lam.real, lam.imag, "."); plt.xlabel("Re"); plt.ylabel("Im"); plt.title("reactor spectrum: diffusion (far left), upwind advection (the complex arc)"); plt.show()"""),
    md(r"""
Diffusion on the 2 cm grid gives the fastest modes, upwind advection the complex arc, and nothing is slower
than a few tenths of a second - the reactor flushes in a residence time. The stiffness ratio is a modest 50
or so (its slowest eigenvalue belongs to a non-normal, numerically differentiated Jacobian and is
ill-conditioned: eigenvalue solvers on different machines report 45 to 50), so an explicit method would need
about twenty times more steps than BDF2 at 0.5 s: implicit integration is worthwhile here, not indispensable.
(The grid, not the physics, makes it stiff: halving Δx doubles the ratio.)

## 4. Fitting k₀ and E with the model inside the loop
"""),
    code(r"""x_tc = np.linspace(0, 2, 11)
T_ref = 320.0                                                   # reparametrise: k_ref = k(T_ref) and E are far less correlated than k0 and E
R = 8.314
def profiles(theta):
    ln_kref, E = theta
    k0 = np.exp(ln_kref + E / (R * T_ref))
    out = []
    for T_in in (310.0, 320.0, 330.0):
        pm = pr.tubular_reactor(n=60, T_in=T_in, k0=k0, E=E)
        sm = pm.solve("bdf", 0.5, order=2, n_out=1)
        out.append(np.interp(x_tc, pm.x, sm.y[-1][pm.n // 2 :]))
    return out
def residuals(theta):
    model = profiles(theta)
    res = []
    for (T_in, grp), prof in zip(data.groupby("T_in_K"), model):
        res.append(np.interp(grp.x_m, x_tc, prof) - grp.T_K.to_numpy())
    return np.concatenate(res)
theta0 = [np.log(5e7 * np.exp(-60e3 / (R * T_ref)) * 1.5), 55e3]      # a poor start: 50 % off in k_ref, E off by 5 kJ/mol
fit = sopt.least_squares(residuals, theta0, diff_step=[1e-3, 1e-3], xtol=1e-8)
ln_kref, E_fit = fit.x
k0_fit = np.exp(ln_kref + E_fit / (R * T_ref))
J = fit.jac; r_ = fit.fun
cov = (r_ @ r_) / (len(r_) - 2) * np.linalg.inv(J.T @ J)
se = np.sqrt(np.diag(cov))
print(f"k(320 K) = {np.exp(ln_kref):.4e} 1/s (+/- {100 * se[0]:.1f} %), E = {E_fit / 1e3:.2f} +/- {se[1] / 1e3:.2f} kJ/mol  (generated with k0 = 5e7, E = 60: k(320) = {5e7 * np.exp(-60e3 / (R * 320)):.4e})")
print(f"-> k0 = {k0_fit:.3e} 1/s; correlation of the fitted pair {cov[0, 1] / (se[0] * se[1]):+.3f}; residual s.d. {np.std(r_):.2f} K (noise 0.5 K); {fit.nfev} model evaluations")"""),
    md(r"""
Each residual evaluation integrates the reactor to steady state three times; the least-squares solver
differentiates it by finite differences, and converges in seven evaluations to parameters within 1 % of the
generating ones, with a residual of 0.42 K against 0.5 K of noise. Two decisions make the fit work. The
parameters are k(T_ref) and E rather than k₀ and E: k₀ and E are tied by the "compensation effect" (a larger E
is compensated by an exponentially larger k₀), and in (k₀, E) a starting guess 5 kJ/mol off in E changes the
rate at 320 K tenfold - the model reactor ignites and the solver fails before the fit begins (shown below).
With k(T_ref) as the parameter, the same guess keeps the rate in the data's range and every step of the fit is
physically sensible. The estimates still correlate at −0.97, because the three runs span only 20 K and mostly
constrain one combination; a wider temperature range, not a cleverer parametrisation, would reduce that. And
the model inside the loop uses a coarser grid (60 intervals) than the final analysis, which costs a bias far
below the measurement noise and halves the fitting time.

## 5. The runaway boundary by parametric sensitivity
"""),
    code(r"""T_ins = np.arange(300.0, 346.0, 2.5)
rises = []
for T_in in T_ins:
    pm = pr.tubular_reactor(n=60, T_in=T_in, k0=k0_fit, E=E_fit)
    rises.append(pm.solve("bdf", 0.5, order=2, n_out=1).y[-1][pm.n // 2 :].max() - T_in)
rises = np.array(rises)
sens = np.gradient(rises, T_ins)
T_crit = T_ins[np.argmax(sens)]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(T_ins, rises, "o-"); axes[0].axvline(T_crit, color="C3", ls="--"); axes[0].set_xlabel("inlet/wall temperature (K)"); axes[0].set_ylabel("hot-spot rise (K)")
axes[1].plot(T_ins, sens, "o-"); axes[1].axvline(T_crit, color="C3", ls="--", label=f"max sensitivity at {T_crit:.1f} K"); axes[1].set_ylabel("d(rise)/dT_in"); axes[1].legend(); plt.show()
print(f"with the fitted parameters: ignition near T_in = {T_crit:.1f} K (the rise jumps from {rises[T_ins == T_crit - 2.5][0]:.0f} to {rises[T_ins == T_crit + 2.5][0]:.0f} K across it)")"""),
    md(r"""
Below the ignition temperature the hot spot rises gently with the inlet temperature; near it the rise becomes
extremely sensitive (the parametric-sensitivity criterion of Bilous and Amundson) and above it the reaction
goes to completion near the inlet with most of the adiabatic rise. The data, all taken below ignition,
determine the Arrhenius parameters well enough to place the boundary within the scan's resolution - which is
what the plant needs to know.

## 6. The transient of a runaway
"""),
    code(r"""pm = pr.tubular_reactor(n=100, T_in=330.0, k0=k0_fit, E=E_fit)
steady = pm.solve("bdf", 0.5, order=2, n_out=1).y[-1]
T_step = T_crit + 3.0
pr_hot = pr.tubular_reactor(n=100, T_in=T_step, k0=k0_fit, E=E_fit)
s_run = ti.bdf(pr_hot.rhs, steady, (0, 120.0), 0.25, 2, band=pr_hot.band, n_out=8)
n = pr_hot.n // 2
for t, y in zip(s_run.t, s_run.y):
    plt.plot(pr_hot.x, y[n:], color=plt.cm.inferno(t / s_run.t[-1]), label=f"t = {t:.0f} s" if t in (0, 30, 60, 120) else None)
plt.xlabel("x (m)"); plt.ylabel("T (K)"); plt.legend(); plt.title(f"inlet stepped from 330 K to {T_step:.1f} K: the hot spot moves upstream and grows"); plt.show()
print(f"hot spot: {steady[n:].max():.0f} K before the step, {s_run.y[-1][n:].max():.0f} K after 120 s; {s_run.jacobian_evaluations} Jacobians, {s_run.newton_iterations / s_run.steps:.1f} Newton per step")"""),
    md(r"""
Stepping the inlet and wall temperature a few degrees above the ignition value launches the runaway: the hot
spot grows and moves upstream as the reaction front settles near the inlet. During the transient the Newton
iteration refreshes its Jacobian several times - the state changes fast - and then settles to one or two
iterations per step.

## 7. The report
"""),
    code(r"""report = f'''# Runaway boundary of the cooled tubular reactor

Data: 3 steady profiles (310, 320, 330 K), 32 of 33 thermocouples (one dead reading dropped).
Model: plug flow with axial dispersion, first-order Arrhenius reaction, wall cooling; method of lines, upwind advection + central diffusion,
       steady states by BDF2 (0.5 s steps, one banded Jacobian; stiffness ratio ~{stiffness.stiffness_ratio(lam):.0f}).
Fit (k at 320 K and E, to avoid the k0-E compensation valley): k(320 K) = {np.exp(ln_kref):.3e} 1/s (+/- {100 * se[0]:.1f} %), E = {E_fit / 1e3:.1f} +/- {se[1] / 1e3:.1f} kJ/mol;
       residual s.d. {np.std(r_):.2f} K against 0.5 K thermocouple noise.
Ignition: maximum parametric sensitivity at T_in = T_wall = {T_crit:.1f} K; hot-spot rise {rises[T_ins == T_crit - 2.5][0]:.0f} K below it, {rises[T_ins == T_crit + 2.5][0]:.0f} K above.
Recommendation: keep inlet and wall below {T_crit - 5:.0f} K (a 5 K margin against the parameter uncertainty and the scan's resolution of 2.5 K).
Caveats: single first-order reaction, constant properties, uniform wall temperature; the fit used only sub-critical data, so the model is extrapolated at ignition.
'''
print(report)"""),
    md(r"""
The report traces every number to its data and method and states the model's limits. The method-of-lines
lessons are in it as well: the steady state came from a stiff transient integrated implicitly with a banded
Jacobian; the parameters were fitted through the model, in a parametrisation chosen from the physics; and the
answer - the ignition temperature - is a sensitivity, which a model yields and a few measurements never could.
"""),
]


# =====================================================================================================
# Teaching layer
def _x(objectives, prereq, time, inside, exercises=(), implement=""):
    return dict(
        objectives=objectives, prereq=prereq, time=time, inside=inside, exercises=list(exercises), implement=implement
    )


EXTRAS["00_what_is_the_method_of_lines"] = _x(
    [
        "write a method-of-lines right-hand side for a PDE",
        "separate spatial and temporal errors with the exact semi-discrete solution",
        "recognise stiffness in a semi-discrete spectrum",
        "read the statistics a time integrator reports",
    ],
    "an introductory ODE course and basic NumPy",
    "45 min",
    [
        md(r"""
## Inside the algorithm: the semi-discrete matrix and its spectrum by hand
"""),
        code(r"""A_hand = (np.diag(-2 * np.ones(39)) + np.diag(np.ones(38), 1) + np.diag(np.ones(38), -1)) / g.dx**2   # interior points only
ev = np.sort(np.linalg.eigvalsh(A_hand))
print("most negative eigenvalue by hand:", ev[0], " theory -4/dx^2 sin^2(39 pi/80):", -4 / g.dx**2 * np.sin(39 * np.pi / 80) ** 2)"""),
    ],
    [
        "Replace the central difference in the twenty-line solver by the fourth-order 5-point stencil (interior points only; keep the 3-point stencil next to the boundaries). How does the spatial error change on 40 intervals? And the extreme eigenvalue?",
        "Run the twenty-line solver with `method='RK45'` instead of BDF and compare the number of time steps for n = 20, 40, 80. Fit the growth law.",
    ],
    "**Implement it yourself:** write backward Euler for the semi-discrete heat equation using `numpy.linalg.solve` on the dense matrix I - h A, and confirm its first-order time error against `integrators.exponential`.",
)

EXTRAS["01_stencils_and_boundary_conditions"] = _x(
    [
        "derive finite-difference weights on any points with Fornberg's algorithm",
        "measure the order of a differentiation operator including its boundary closures",
        "impose Dirichlet, Neumann, Robin and periodic conditions by ghost points on node and cell grids",
        "use consistency checks and stretched grids",
    ],
    "notebook 00",
    "60 min",
    [
        md(r"""
## Inside the algorithm: a Neumann ghost by hand
"""),
        code(r"""u_in = np.array([1.0, 1.4, 1.9]); slope = -2.0; dx = 0.1
ghost = u_in[1] - 2 * dx * slope                   # even reflection about the end node, plus the slope
print("ghost by hand:", ghost, "; library:", bc.pad(u_in, dx, bc.Neumann(slope), bc.Neumann(0.0), kind="node")[0])
print("centred slope at the end node:", (u_in[1] - ghost) / (2 * dx))"""),
    ],
    [
        "Build a fourth-order Laplacian with fourth-order one-sided closures and a fourth-order Laplacian with only second-order closures (use `fd_weights` on 4 points at the ends). Measure both orders. Which closure order does the global error follow, and why?",
        "On a cell-centred grid impose u = 1 at x = 0 by the ghost formula and compare the resulting boundary flux k u_x with the exact one for the steady solution of u_xx = 0, u(0) = 1, u(1) = 0. What order is the flux error?",
    ],
    "**Implement it yourself:** write Fornberg's algorithm (or solve the Vandermonde system for the weights) for the first derivative on the points (0, 0.5, 1.2) at x = 0.3 and compare with `stencils.fd_weights`.",
)

EXTRAS["02_stiffness_and_choosing_the_integrator"] = _x(
    [
        "compute the spectrum and stiffness ratio of a semi-discrete system",
        "find the stability limit of an explicit method from its stability function",
        "distinguish A-stable from L-stable methods and recognise ringing",
        "estimate explicit versus implicit cost",
    ],
    "notebook 00; the ODE notebooks of engnumeric help",
    "75 min",
    [
        md(r"""
## Inside the algorithm: the stability limit from the stability polynomial
"""),
        code(r"""R = lambda z: 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24     # RK4
z = np.linspace(-3, 0, 30001)
print("RK4 stable down to z =", z[np.argmax(np.abs(R(z)) <= 1)], " -> h_max = 2.785/|lambda_min| =", 2.785 / abs(lam[0].real), "; library:", h_rk4)"""),
    ],
    [
        "Repeat the integrator comparison for the advection equation `problems.advection_periodic(scheme='centred')` whose spectrum is imaginary. Which explicit methods are stable at all, and what does BDF2 do to the pulse's amplitude after one period?",
        "Start Crank-Nicolson (the trapezoidal rule) on the hot band with two backward-Euler half steps (Rannacher's trick) and compare the ringing with the plain start.",
    ],
    "**Implement it yourself:** write the trapezoidal rule for the linear semi-discrete heat equation with one factorisation of (I - hA/2) and reproduce the ringing figure; then halve h until the ringing disappears and relate that h to |R(h lambda_max)|.",
)

EXTRAS["03_banded_newton_and_the_cost_of_implicit_steps"] = _x(
    [
        "recognise the band structure of a method-of-lines Jacobian",
        "compute a banded Jacobian by column colouring",
        "use banded LU and know its cost",
        "explain the modified Newton iteration and when the Jacobian must be refreshed",
    ],
    "notebooks 00 and 02",
    "60 min",
    [
        md(r"""
## Inside the algorithm: colouring by hand
"""),
        code(r"""f0 = f(u); Jc = np.zeros((31, 31)); h_ = 1e-7
for c in range(3):                               # three colours for a tridiagonal Jacobian
    cols = np.arange(c, 31, 3)
    up = u.copy(); up[cols] += h_
    df = (f(up) - f0) / h_
    for j in cols:
        for i in range(max(0, j - 1), min(31, j + 2)):
            Jc[i, j] = df[i]
print("coloured Jacobian vs dense:", f"{np.max(np.abs(Jc - J)):.1e}", "- from 3 evaluations instead of 31")"""),
    ],
    [
        "Measure the Jacobian evaluations and Newton iterations of `problems.porous_medium()` for BDF1, BDF2 and BDF4 at the same step. Which order is cheapest per unit of accuracy against the Barenblatt solution?",
        "For a 2-D grid of n x n points the Jacobian band is n. Tabulate the banded LU cost 2 N b^2 (N = n^2) and the dense cost 2 N^3/3 for n = 16, 32, 64, 128, and the memory of the banded factor (2 b + 1) N numbers versus N^2.",
    ],
    "**Implement it yourself:** write a banded LU without pivoting for a tridiagonal matrix (the Thomas algorithm) and use it inside a backward-Euler step of the heat equation; compare the step with `integrators.bdf(order=1)`.",
)

EXTRAS["04_the_heat_equation_in_every_form"] = _x(
    [
        "solve the heat equation with every boundary type and check it against exact solutions",
        "run a space-time convergence study and read the two orders from it",
        "handle layered materials and radial geometry",
        "integrate a stiff engineering problem with large implicit steps",
    ],
    "notebooks 01 and 02",
    "60 min",
    [
        md(r"""
## Inside the algorithm: the Robin eigenvalue
"""),
        code(r"""from scipy.optimize import brentq
Bi = 2.0
mu = brentq(lambda m: m * np.tan(m / 2) - Bi, 1e-6, np.pi - 1e-6)
p = pr.heat_robin(); s = p.solve("bdf", 1e-3, order=4)
rate = -np.log(s.y[-1][20] / p.y0[20]) / s.t[-1]
print(f"measured decay rate {rate:.5f} vs mu^2 = {mu**2:.5f}")"""),
    ],
    [
        "The periodically heated slab has penetration depth 1/k with k = sqrt(omega / 2 kappa). Solve it on a domain of length 2/k and of length 8/k with the far boundary insulated instead of imposed; how does the far-boundary choice affect the solution near x = 0, and at what domain length does it stop mattering?",
        "Make the composite wall's second layer ten times more conductive and a tenth as thick (equal resistance). Confirm the same steady flux, and compare the time to reach 95 % of it.",
    ],
    "**Implement it yourself:** write the cylindrical diffusion operator with the control-volume treatment at r = 0 and reproduce the cooling-cylinder decay rate.",
)

EXTRAS["05_nonlinear_diffusion"] = _x(
    [
        "integrate nonlinear diffusion problems with Newton-based implicit methods",
        "formulate degenerate problems so that Newton converges",
        "check stationary solutions, conservation and energy decrease",
        "read the spectrum of bistable and fourth-order problems",
    ],
    "notebooks 03 and 04",
    "75 min",
    [
        md(r"""
## Inside the algorithm: the two formulations of the porous-medium operator
"""),
        code(r"""g_ = pr.porous_medium().grid
u_ = pr.porous_medium().y0
from engmolines import operators as op, boundary as bc
form_a = op.diffusion(np.maximum(u_, 0) ** 2, g_, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(0.0))              # (u^2)_xx
form_b = op.diffusion(u_, g_, lambda x, u: 2 * np.maximum(u, 0), bc.Dirichlet(0.0), bc.Dirichlet(0.0))    # (2u u_x)_x with harmonic means
print("the two forms agree inside the support to", f"{np.max(np.abs(form_a - form_b)[u_ > 0.1]):.1e}", "and differ at the front, where the harmonic mean of 0 and 2u is 0")"""),
    ],
    [
        "Run the porous-medium problem with m = 3 and m = 4 (the Barenblatt exponents change: alpha = beta = 1/(m+1), kk = (m-1)/(2m(m+1))) and confirm the front speed law t^(1/(m+1)).",
        "Perturb the Allen-Cahn kink by adding 0.01 sin(pi x) and integrate to t = 50. Does the kink return to its shape, and does it drift? Measure the drift against eps.",
    ],
    "**Implement it yourself:** write the Cahn-Hilliard right-hand side as two applications of the 3-point Laplacian with no-flux ghosts, and verify mass conservation and the band (2, 2) of its Jacobian with `linalg.banded_jacobian`.",
)

EXTRAS["06_reaction_diffusion_fronts_and_patterns"] = _x(
    [
        "measure a travelling-wave speed and recognise its slow approach to the KPP value",
        "split a problem into implicit diffusion and explicit reaction (IMEX)",
        "decide from the spectrum when an explicit method is the right tool",
        "integrate 2-D reaction-diffusion systems",
    ],
    "notebooks 02 and 05",
    "60 min",
    [
        md(r"""
## Inside the algorithm: one IMEX Euler step by hand
"""),
        code(r"""from engmolines import linalg as la
p1 = pr.fisher(); u = p1.y0.copy(); h = 0.02
A = la.from_banded(la.banded_jacobian(lambda w: p1.params["rhs_implicit"](0.0, w), u, 1, 1), 1, 1)   # the (linear) diffusion operator with its Neumann ghosts
rhs = u + h * p1.params["rhs_explicit"](0.0, u)          # explicit reaction
u_new = np.linalg.solve(np.eye(len(u)) - h * A, rhs)     # implicit diffusion: one linear solve, no Newton
lib = ti.imex(p1.params["rhs_explicit"], p1.params["rhs_implicit"], u, (0, h), h, band=(1, 1), order=1, n_out=1).y[-1]
print("one IMEX Euler step by hand vs library:", f"{np.max(np.abs(u_new - lib)):.1e}")"""),
    ],
    [
        "Measure the Fisher front speed on grids with dx = 0.3, 0.15, 0.075 and extrapolate the asymptotic speed at t = 15; compare with 2 and with Bramson's 2 - 3/(2t).",
        "Change Gray-Scott to F = 0.03, k = 0.062 and to F = 0.025, k = 0.05 (worms, then chaos). Does explicit RK4 remain adequate? Confirm with the spectrum at a developed pattern.",
    ],
    "**Implement it yourself:** write the IMEX CNAB2 step (Crank-Nicolson for diffusion, second-order Adams-Bashforth for the reaction) and reproduce the library's front position.",
)

EXTRAS["07_advection_the_hard_case"] = _x(
    [
        "explain why centred stencils need integrators that cover the imaginary axis",
        "derive and verify numerical diffusion from the modified equation",
        "use flux limiters and the TVD property",
        "impose inflow conditions and Danckwerts boundaries",
    ],
    "notebook 02",
    "75 min",
    [
        md(r"""
## Inside the algorithm: the upwind symbol and its circle
"""),
        code(r"""th = np.linspace(0, 2 * np.pi, 400)
sig = -analysis.symbol("first_upwind", th, p.grid.dx, 1.0)        # the operator is -a u_x
plt.plot(sig.real, sig.imag); plt.plot(lamu.real, lamu.imag, "k."); plt.gca().set_aspect("equal"); plt.title("symbol of upwind (line) and the operator's eigenvalues (dots)"); plt.show()"""),
    ],
    [
        "Compute the fully discrete amplification |R(h sigma(theta))| for upwind + Euler at Courant numbers 0.5, 1.0 and 1.1 with `analysis.amplification`, and for centred + RK4 at Courant numbers 2.5 and 3.0. Confirm the stability limits by integrating the pulse.",
        "Reverse the flow in the plug-flow reactor (u = -1, feed at x = 1) and solve it; then set the feed concentration to a square pulse in time and watch it exit. What happens if you (wrongly) impose a Dirichlet value at the outflow?",
    ],
    "**Implement it yourself:** write the second-order upwind-biased stencil (3 points upstream) and show from its modified equation that it is dispersive, not diffusive; measure the phase error of a sine wave after one period.",
)

EXTRAS["08_shocks_and_conservation_laws"] = _x(
    [
        "write finite-volume right-hand sides for scalar and system conservation laws",
        "verify shock speeds by Rankine-Hugoniot and entropy-satisfying rarefactions",
        "check conservation to rounding",
        "solve a hyperbolic system against an exact Riemann solution",
    ],
    "notebook 07",
    "75 min",
    [
        md(r"""
## Inside the algorithm: the Rusanov flux
"""),
        code(r"""uL, uR = 1.0, 0.0
F = lambda u: 0.5 * u * u  # noqa: E731
s_max = max(abs(uL), abs(uR))
print("Rusanov flux at the jump:", 0.5 * (F(uL) + F(uR)) - 0.5 * s_max * (uR - uL), "= average flux + dissipation s (uL - uR)/2")
print("the exact Godunov flux for this shock is F(uL) =", F(uL), "(the shock moves right, so the left state feeds the face)")"""),
    ],
    [
        "Start Burgers' equation from a smooth bump u0 = exp(-x^2) and measure the shock's position and the integral of u against the exact Rankine-Hugoniot prediction from the pre-shock profile (the equal-area rule).",
        "Dam break onto dry ground (hR = 0): what breaks in the shallow-water right-hand side, and how do real codes handle it? Try hR = 1e-3 and compare the front speed with 2 sqrt(g hL).",
    ],
    "**Implement it yourself:** write the Godunov flux for Burgers' equation (the exact Riemann solver for a convex flux) and compare its shock resolution with the Rusanov flux.",
)

EXTRAS["09_waves_strings_and_beams"] = _x(
    [
        "integrate second-order wave equations as first-order systems and with Stormer-Verlet",
        "monitor energy and recognise numerical damping",
        "relate the stiffness of strings and beams to their operators",
        "treat damping and moving loads",
    ],
    "notebooks 02 and 09 of engnumeric help",
    "60 min",
    [
        md(r"""
## Inside the algorithm: one Verlet step
"""),
        code(r"""u_, v_ = ps.y0[:101].copy(), ps.y0[101:].copy(); h_ = 0.005
a_ = ps.params["accel"](0.0, u_)
v_half = v_ + 0.5 * h_ * a_
u_ = u_ + h_ * v_half
v_ = v_half + 0.5 * h_ * ps.params["accel"](h_, u_)
lib = ti.verlet(ps.params["accel"], ps.y0[:101], ps.y0[101:], (0, h_), h_, n_out=1).y[-1]
print("one step by hand vs library:", np.max(np.abs(np.r_[u_, v_] - lib)))"""),
    ],
    [
        "Integrate the string with the trapezoidal rule (A-stable, non-dissipative) for 20 periods: is the energy conserved? Is the phase correct? Compare with Verlet at the same step.",
        "Give the beam a clamped left end and a free right end (`biharmonic(w, g, 'clamped', 'free')`) and find its first natural frequency from a long integration (the FFT of the tip displacement); compare with (1.875)^2 sqrt(EI/rhoA)/L^2.",
    ],
    "**Implement it yourself:** write the energy-conserving Verlet step for the wave equation with a damping term -2 a v (treat the damping implicitly in the velocity update) and verify the decay rate against the telegraph solution.",
)

EXTRAS["10_dispersive_waves_and_spectral_methods"] = _x(
    [
        "differentiate periodic functions spectrally and know the accuracy it buys",
        "recognise the n^3 stiffness of dispersive operators",
        "check soliton solutions and conserved quantities",
        "diagnose aliasing and apply the 2/3 rule",
    ],
    "notebook 09 of engnumeric (the FFT) helps",
    "60 min",
    [
        md(r"""
## Inside the algorithm: a spectral derivative by hand
"""),
        code(r"""n, L = 64, 40.0
k = 2 * np.pi * np.fft.fftfreq(n, L / n)
xx = spectral.grid(-L / 2, L / 2, n); f = 1 / np.cosh(xx) ** 2
d3_hand = np.fft.ifft((1j * k) ** 3 * np.fft.fft(f)).real
print("u_xxx of sech^2 by hand vs library:", np.max(np.abs(d3_hand - spectral.derivative(f, L, 3))))"""),
    ],
    [
        "Integrate the KdV soliton with an integrating factor: write v = exp(-D3 t) u in Fourier space so that the linear term disappears, and integrate only the nonlinear term explicitly. How much larger can the step be?",
        "Give the NLS soliton twice the amplitude (2 sech(2x) e^(4it) is also a soliton). Measure the step the adaptive integrator needs and explain it from the frequency 4.",
    ],
    "**Implement it yourself:** write the KdV right-hand side with the 2/3 dealiasing applied to the product u u_x, and compare the two-soliton invariants with and without it at n = 64 (under-resolved).",
)

EXTRAS["11_coupled_systems_and_daes"] = _x(
    [
        "stack coupled PDE-ODE systems into one state and find the Jacobian band",
        "check coupled problems by global balances",
        "handle streams in opposite directions",
        "write boundary conditions and constraints as algebraic equations and solve the DAE",
    ],
    "notebooks 03 and 07",
    "75 min",
    [
        md(r"""
## Inside the algorithm: the band of a coupled Jacobian
"""),
        code(r"""pe = pr.counterflow_exchanger(n=400)
J = la.dense_jacobian(lambda y: pe.rhs(0.0, y), pe.y0 + 0.1)
nz = np.abs(J) > 1e-9
rows, cols = np.where(nz)
print(f"Jacobian of the exchanger: nonzeros within {np.max(np.abs(rows - cols))} of the diagonal (the coupling T_h,i <-> T_c,i sits n + 1 = 401 off)")
plt.spy(nz, markersize=1); plt.show()"""),
    ],
    [
        "Replace the adsorber's linear isotherm by a Langmuir isotherm q* = q_max K c/(1 + K c). Does the front sharpen or spread (a favourable isotherm sharpens it)? Measure the breakthrough curve's width.",
        "Write the catalyst-pellet problem as a DAE with a film-resistance boundary condition at the surface: 0 = Sh (1 - c_s) - c_r(1). Solve with `dae_bdf` and compare the effectiveness with the no-resistance case.",
    ],
    "**Implement it yourself:** add a second adsorbing species with a different K to the column and show that the two breakthrough fronts travel at their own stoichiometric speeds.",
)

EXTRAS["12_two_dimensions"] = _x(
    [
        "flatten 2-D fields and know the Kronecker band of their Jacobian",
        "solve 2-D parabolic problems implicitly with banded Newton",
        "measure fronts in 2-D",
        "set up a vorticity-streamfunction flow solver and check it against a benchmark",
    ],
    "notebooks 03 and 06",
    "90 min",
    [
        md(r"""
## Inside the algorithm: the 2-D Laplacian from two 1-D ones
"""),
        code(r"""U = np.random.default_rng(0).normal(size=(6, 5))
by_hand = Ax.matvec(U) + Ay.matvec(U.T).T
print("kron_sum vs two 1-D applications:", np.max(np.abs(apply2(U.ravel()) - by_hand.ravel())), "; vs the dense Kronecker matrix:", np.max(np.abs(K @ U.ravel() - by_hand.ravel())))"""),
    ],
    [
        "Solve the plate problem on grids of 12, 24 and 48 intervals with BDF4 and record the Jacobian factorisation time (use `time.perf_counter` around `linalg.lu_banded` on the Jacobian). Confirm the n^4 cost law and compare with a dense `numpy.linalg.solve` of the same system.",
        "Run the cavity at Re = 400 on the 32 x 32 grid and on 48 x 48 (`cavity_flow(n=48, Re=400)`). Compare u_min on the centreline with Ghia's -0.3273 at y = 0.2813, and the time to steady state.",
    ],
    "**Implement it yourself:** write the 2-D heat right-hand side with Neumann ghosts on all four sides and verify that the total heat is conserved; then find the band of its Jacobian with `linalg.banded_jacobian` and check it against the dense one.",
)

EXTRAS["13_diagnosing_failures"] = _x(
    [
        "recognise instability, boundary-induced growth, ringing, undershoot, boundary leakage, aliasing, pinning errors and spurious roots from their symptoms",
        "use the spectrum, conservation integrals, invariants and boundary values as diagnostics",
        "apply the remedy that matches the diagnosis",
    ],
    "notebooks 02, 07 and 10",
    "75 min",
    [
        md(r"""
## Inside the diagnosis: the amplification of the sawtooth
"""),
        code(r"""p = pr.advection_periodic(scheme="centred")
sig = analysis.symbol("first_central", np.pi / 2, p.grid.dx)                 # the mode theta = pi/2 of the centred stencil
for method in ("euler", "rk4"):
    print(f"{method}: |R(h sigma)| at h = dx/a (Courant 1) = {abs(stiffness.stability_function(method, -p.grid.dx * sig)):.4f}")"""),
    ],
    [
        "Take the heat equation with Neumann ends and impose the Neumann condition by a *first-order* one-sided ghost (u_ghost = u_0 - dx g) instead of the centred one. Measure the global order of accuracy and the conservation error. Which diagnostic reveals the problem first?",
        "Integrate Fisher's equation with an explicit step just above the diffusion limit. Describe the symptom: where does the instability start, and how fast does it grow per step (compare with |R(h lambda_max)|)?",
    ],
    "**Implement it yourself:** write a function `diagnose(rhs, y, band)` that reports the spectrum's largest real part, the stiffness ratio, the RK4 and Euler step limits, and whether the spectrum touches the imaginary axis; run it on five problems of the library.",
)

EXTRAS["14_moving_boundaries_and_transformed_domains"] = _x(
    [
        "transform a moving-boundary problem to a fixed domain with an extra ODE",
        "recognise variable-coefficient parabolic problems as heat equations in transformed variables",
        "use stretched grids for boundary layers",
        "know the cell Peclet condition",
    ],
    "notebooks 01 and 05",
    "75 min",
    [
        md(r"""
## Inside the algorithm: the Stefan condition as the last equation
"""),
        code(r"""ps = pr.stefan()
y = ps.y0.copy()
rate = ps.rhs(0.0, y)
T, s0 = y[:-1], y[-1]
grad = (3 * T[-1] - 4 * T[-2] + T[-3]) / (2 * ps.grid.dx) / s0     # T_x at the front
print(f"ds/dt from the Stefan condition -T_x(s)/St = {-grad / 1.0:.5f}; the right-hand side's last entry: {rate[-1]:.5f}; exact lambda/sqrt(t0) = {ps.params['lambda'] / np.sqrt(ps.params['t0']):.5f}")"""),
    ],
    [
        "Price the Black-Scholes call on a grid in z = log S (uniform in z from log 1 to log 400): the equation becomes constant-coefficient. Compare the error with the S-grid at equal numbers of points, especially near the strike.",
        "Solve the Stefan problem with St = 0.1 and St = 10 and compare the front laws 2 lambda sqrt(t); what limits the time step in each case (look at the spectrum of the transformed system, which includes the front ODE)?",
    ],
    "**Implement it yourself:** write the steady dispersed reactor (-c_x + c_xx/Pe - Da c = 0 with the Danckwerts Robin inlet and a zero-gradient outlet) on a grid stretched towards the inlet, solve it by a direct linear solve, and show that the integral balance holds: inlet flux (c - c_x/Pe = 1) minus outlet flux (c(1)) equals the consumption Da * integral of c.",
)

EXTRAS["15_project_runaway_in_a_tubular_reactor"] = _x(
    [
        "build a coupled reactor model and integrate it to steady state with a stiff solver",
        "fit model parameters with the model inside the loop and choose a well-conditioned parametrisation",
        "quantify parameter uncertainty and correlation",
        "locate an ignition boundary by parametric sensitivity and report with caveats",
    ],
    "notebooks 02, 03, 07 and 11",
    "3 h",
    [
        md(r"""
## Inside the analysis: why k0 and E are not fitted directly
"""),
        code(r"""def residuals_k0(theta):                                   # the same residuals, parametrised by (ln k0, E)
    ln_k0, E = theta
    return residuals([ln_k0 - E / (R * T_ref), E])
theta0_k0 = [np.log(5e7 * 1.5), 55e3]                     # the same "50 % off, 5 kJ/mol off" start, now at fixed k0
print(f"rate at 320 K implied by the start: {np.exp(theta0_k0[0] - theta0_k0[1] / (R * T_ref)):.3e} 1/s against the true {5e7 * np.exp(-60e3 / (R * T_ref)):.3e}: tenfold")
try:
    sopt.least_squares(residuals_k0, theta0_k0, diff_step=[1e-3, 1e-3], xtol=1e-8)
except RuntimeError as e:
    print("the fit cannot even start:", e)"""),
    ],
    [
        "Repeat the fit with only the 310 and 320 K profiles. How do the standard errors and the predicted ignition temperature change? Then add a (synthetic) 335 K profile generated with the true parameters: what does a near-critical measurement do to the fit?",
        "The runaway boundary depends on the wall coefficient h_w. Map the ignition temperature as a function of h_w from 0.05 to 0.3 1/s (a runaway diagram) with the fitted kinetics.",
    ],
    "**Implement it yourself:** replace the steady-state-by-transient with a direct Newton solve of the steady equations rhs(y) = 0 (use `linalg.banded_jacobian` and `linalg.solve_banded`) started from the transient's state at t = 20 s, and compare the cost with integrating to t = 80 s.",
)


def apply_extras(name, cells):
    ex = EXTRAS[name]
    header = md(
        "**What you will learn**\n"
        + "\n".join(f"- {o}" for o in ex["objectives"])
        + f"\n\n**Before you start:** {ex['prereq']}  ·  **Time:** about {ex['time']}"
    )
    cells = [cells[0], header] + list(cells[1:])
    idx = [i for i, c in enumerate(cells) if c.cell_type == "markdown" and "## Exercises" in c.source]
    if idx:
        i = idx[0]
        n_items = len(re.findall(r"^\d+\. ", cells[i].source, flags=re.MULTILINE))
        cells[i] = md(cells[i].source.rstrip() + f"\n{n_items + 1}. {ex['implement']}")
        return cells[:i] + list(ex["inside"]) + cells[i:]
    items = ex["exercises"] + [ex["implement"]]
    exercises = md("## Exercises\n" + "\n".join(f"{k}. {t}" for k, t in enumerate(items, 1)))
    return cells + list(ex["inside"]) + [exercises]


def write_all(run: bool = True, only=()):
    import nbclient

    for name in sorted(NOTEBOOKS):
        if only and not name.startswith(tuple(only)):
            continue
        cells = apply_extras(name, NOTEBOOKS[name])
        nb = new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                "language_info": {"name": "python"},
            },
        )
        if run:
            nbclient.NotebookClient(
                nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}
            ).execute()
        nbformat.write(nb, HERE / f"{name}.ipynb")
        print("wrote", name)


if __name__ == "__main__":
    write_all(run="--no-run" not in sys.argv, only=[a for a in sys.argv[1:] if not a.startswith("--")])
