"""Build the worked solutions to every exercise of the course, as executed notebooks.

The exercise texts are taken from ../notebooks/build_notebooks.py, so questions and solutions cannot
drift apart; the build fails if a notebook's number of solutions differs from its number of exercises.

    python solutions/build_solutions.py            # build and execute all solution notebooks
    python solutions/build_solutions.py 05 12      # only those whose names start with 05 or 12
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("course", HERE.parent / "notebooks" / "build_notebooks.py")
course = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(course)
md, code = course.md, course.code
SOLUTIONS: dict[str, dict] = {}


def solutions(name, setup="", imports=()):
    """Register the solutions of one course notebook: a function returning one list of cells per exercise."""

    def deco(fn):
        SOLUTIONS[name] = {"setup": setup, "imports": list(imports), "answers": fn()}
        return fn

    return deco


def exercises(name) -> list[str]:
    cells = course.apply_extras(name, list(course.NOTEBOOKS[name]))
    text = [c.source for c in cells if c.cell_type == "markdown" and "## Exercises" in c.source][0]
    items = re.split(r"\n(?=\d+\. )", text.split("## Exercises")[1].strip())
    return [re.sub(r"^\d+\.\s*", "", it).strip() for it in items]


def notebook_cells(name) -> list:
    title = re.sub(r"^#\s*", "", course.NOTEBOOKS[name][0].source.splitlines()[0])
    entry = SOLUTIONS[name]
    ex = exercises(name)
    if len(ex) != len(entry["answers"]):
        raise ValueError(f"{name}: {len(ex)} exercises but {len(entry['answers'])} solutions")
    cells = [
        md(f"""
# Solutions · {title}

Worked solutions to the exercises of [notebook {name[:2]}](../notebooks/{name}.ipynb). Try each exercise
yourself before reading its solution - the learning happens in the attempt. Solutions are one way to
answer each question; other correct approaches exist.
"""),
        course.setup_cell(entry["imports"]),
    ] + ([code(entry["setup"])] if entry["setup"] else [])
    for i, (text, answer) in enumerate(zip(ex, entry["answers"]), 1):
        cells.append(md(f"## Exercise {i}\n\n{text}"))
        cells.extend(answer)
    return cells


def write_all(only=()):
    import nbclient

    for name in SOLUTIONS:
        if only and not name.startswith(tuple(only)):
            continue
        nb = new_notebook(
            cells=notebook_cells(name),
            metadata={
                "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                "language_info": {"name": "python"},
            },
        )
        nbclient.NotebookClient(
            nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}
        ).execute()
        path = HERE / f"{name[:2]}_solutions.ipynb"
        nbformat.write(nb, path)
        print("wrote", path.name)


@solutions(
    "00_what_is_the_method_of_lines",
    imports=[
        "from scipy import integrate as sint",
        "from engmolines import grids, boundary as bc, operators as op, integrators as ti, stencils as st, stiffness",
    ],
)
def _():
    return [
        [
            code(r"""n = 40
x = np.linspace(0, 1, n + 1); dx = x[1] - x[0]
def rhs4(t, u):
    d = np.zeros_like(u)
    d[2:-2] = (-u[4:] + 16 * u[3:-1] - 30 * u[2:-2] + 16 * u[1:-3] - u[:-4]) / (12 * dx**2)   # 5-point, fourth order
    d[1] = (u[2] - 2 * u[1] + u[0]) / dx**2                                                     # 3-point next to the ends
    d[-2] = (u[-1] - 2 * u[-2] + u[-3]) / dx**2
    return d
def rhs2(t, u):
    d = np.zeros_like(u); d[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2; return d
exact = np.sin(np.pi * x) * np.exp(-np.pi**2 * 0.1)
for name, f in (("3-point", rhs2), ("5-point interior", rhs4)):
    s = sint.solve_ivp(f, (0, 0.1), np.sin(np.pi * x), method="BDF", rtol=1e-10, atol=1e-12)
    lam = np.linalg.eigvals(stiffness.jacobian(f, np.sin(np.pi * x)))
    print(f"{name:<17}: spatial error {np.max(np.abs(s.y[:, -1] - exact)):.2e}; extreme eigenvalue {lam.real.min():.0f}")"""),
            md(r"""
The fourth-order interior stencil cuts the spatial error by almost three orders of magnitude on the same grid, even
with second-order closures next to the boundaries (their local error is small because the solution is flat
there; in general the global error follows the closure order one less, see notebook 01). Its extreme
eigenvalue is −16/(3Δx²) ≈ −8533 instead of −4/Δx² = −6400: a higher-order stencil is somewhat stiffer.
"""),
        ],
        [
            code(r"""def mol(n):
    x = np.linspace(0, 1, n + 1); dx = x[1] - x[0]
    def f(t, u):
        d = np.zeros_like(u); d[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2; return d
    return f, np.sin(np.pi * x)
rows = []
for n in (20, 40, 80, 160):
    f, u0 = mol(n)
    s_rk = sint.solve_ivp(f, (0, 0.1), u0, method="RK45", rtol=1e-6, atol=1e-9)
    s_bdf = sint.solve_ivp(f, (0, 0.1), u0, method="BDF", rtol=1e-6, atol=1e-9)
    rows.append((n, len(s_rk.t) - 1, len(s_bdf.t) - 1))
tab = pd.DataFrame(rows, columns=["n", "RK45 steps", "BDF steps"])
print(tab.to_string(index=False))
print("RK45 step ratio per doubling of n:", np.round(tab["RK45 steps"].to_numpy()[1:] / tab["RK45 steps"].to_numpy()[:-1], 2), " (theory: 4, the n^2 law)")"""),
            md(r"""
RK45's step count quadruples with every doubling of n - the stability limit h < 2.8Δx²/4 shrinks fourfold -
while BDF's stays essentially constant, set by the accuracy of the smooth solution. Explicit cost grows like
n³ (steps × work per step), implicit like n.
"""),
        ],
        [
            code(r"""g = grids.uniform(0, 1, 40)
A = st.diff_operator(g.x, 2, 2).to_dense(); A[0, :] = A[-1, :] = 0.0
u0 = np.sin(np.pi * g.x)
ref = ti.exponential(A, u0, (0, 0.1)).y[-1]
errs = []
for h in (0.01, 0.005, 0.0025):
    M = np.eye(41) - h * A
    u = u0.copy()
    for _ in range(int(round(0.1 / h))):
        u = np.linalg.solve(M, u)                       # backward Euler: (I - hA) u_{n+1} = u_n
    errs.append(np.max(np.abs(u - ref)))
print("backward Euler errors:", ["%.2e" % e for e in errs], "-> ratios", np.round(np.array(errs[:-1]) / np.array(errs[1:]), 2), "(first order: 2)")"""),
            md(r"""
Backward Euler solves a linear system per step with the fixed matrix I − hA; against the exact semi-discrete
solution its error halves when the step halves. (Factorising I − hA once instead of calling `solve` every
step is what the library's banded Newton does.)
"""),
        ],
    ]


@solutions(
    "01_stencils_and_boundary_conditions",
    imports=["from engmolines import grids, stencils as st, boundary as bc, operators as op, analysis"],
)
def _():
    return [
        [
            code(r"""f = lambda x: np.sin(3 * x); d2 = lambda x: -9 * np.sin(3 * x)          # noqa: E731
def laplacian(x, closure_order):
    n = len(x); rows = {}
    for i in range(n):
        if 2 <= i <= n - 3:
            idx = np.arange(i - 2, i + 3)
        else:
            npts = 6 if closure_order == 4 else 4
            idx = np.arange(0, npts) if i < 2 else np.arange(n - npts, n)
        for j, wj in zip(idx, st.fd_weights(x[i], x[idx], 2)):
            rows.setdefault(j - i, np.zeros(n))[i] += wj
    offs = np.array(sorted(rows)); return st.StencilOperator(offs, np.array([rows[o] for o in offs]))
for closure in (4, 2):
    errs = [np.max(np.abs(laplacian(np.linspace(0, 1, n + 1), closure)(f(np.linspace(0, 1, n + 1))) - d2(np.linspace(0, 1, n + 1)))) for n in (20, 40, 80, 160)]
    print(f"closure order {closure}: errors {['%.1e' % e for e in errs]}, orders {np.round(analysis.observed_order(errs), 2)}")"""),
            md(r"""
With fourth-order closures the whole operator is fourth order; with second-order closures the global error
is second order - the boundary rows dominate the error, however good the interior stencil. (For time-dependent
problems the rule is slightly more forgiving: closures one order lower than the interior keep the global order
for parabolic problems, because the boundary error is smoothed; for the operator itself, as here, it is not.)
"""),
        ],
        [
            code(r"""for n in (10, 20, 40, 80):
    g = grids.cell_centred(0, 1, n)
    u = 1 - g.x                                                     # the steady solution of u_xx = 0, u(0) = 1, u(1) = 0
    p = bc.pad(u, g.dx, bc.Dirichlet(1.0), bc.Dirichlet(0.0), kind="cell")
    flux = -(p[1] - p[0]) / g.dx                                    # -k u_x at the left face from the ghost
    print(f"n = {n:3d}: boundary flux {flux:.6f} (exact 1)")"""),
            md(r"""
For a linear solution the ghost formula is exact: the odd reflection about the face reproduces the linear
profile and its slope. For curved solutions the flux error is second order in Δx, like the ghost formula
itself.
"""),
        ],
        [
            code(r"""xs = np.array([0.0, 0.5, 1.2]); x0 = 0.3
V = np.vander(xs - x0, increasing=True).T                             # rows: powers of (x_j - x0)
rhs = np.array([0.0, 1.0, 0.0])                                      # we want the first derivative: d/dx of 1, (x - x0), (x - x0)^2 at x0
w = np.linalg.solve(V, rhs)
print("weights from the Vandermonde system:", w, "; Fornberg:", st.fd_weights(x0, xs, 1))"""),
            md(r"""
Requiring the formula to be exact for 1, (x − x₀) and (x − x₀)² gives three linear equations for the three
weights; Fornberg's recurrence solves the same system stably for any number of points and any derivative.
"""),
        ],
    ]


@solutions(
    "02_stiffness_and_choosing_the_integrator",
    imports=[
        "from engmolines import problems as pr, integrators as ti, stiffness, grids, boundary as bc, operators as op, stencils as st"
    ],
)
def _():
    return [
        [
            code(r"""p = pr.advection_periodic(scheme="centred")
lam = stiffness.spectrum(p.rhs, p.y0)
for m in ("euler", "heun", "rk3", "rk4"):
    print(f"{m:<5}: step limit {stiffness.explicit_step_limit(lam, m):.2e}")
s = ti.bdf(p.rhs, p.y0, (0, p.t_end), 0.005, 2, band=p.band, n_out=1)
print(f"BDF2 after one period: amplitude {s.y[-1].max():.3f} of the initial 1.000 - the A-stable method damps undamped waves")"""),
            md(r"""
Euler and Heun have no stable step at all on a purely imaginary spectrum; RK3 and RK4 do (their regions cross
the imaginary axis). BDF2, A-stable, is stable at any step - and damps the pulse by a tenth in one period,
because its |R| < 1 off the real axis: stability is not fidelity for non-dissipative problems.
"""),
        ],
        [
            code(r"""g = grids.uniform(0, 1, 40)
f = lambda t, u: op.diffusion(u, g, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t)   # noqa: E731
u0 = np.where((g.x > 0.4) & (g.x < 0.6), 1.0, 0.0); h = 0.02
plain = ti.trapezoidal(f, u0, (0, 0.1), h, band=(1, 1), n_out=5)
start = ti.bdf(f, u0, (0, h), h / 2, 1, band=(1, 1), n_out=1).y[-1]           # two backward-Euler half steps
rann = ti.trapezoidal(f, start, (h, 0.1), h, band=(1, 1), n_out=4)
A = st.diff_operator(g.x, 2, 2).to_dense(); A[0, :] = A[-1, :] = 0.0
ref = ti.exponential(A, u0, (0, 0.1)).y[-1]
print(f"plain trapezoidal: min value {plain.y[-1].min():+.4f}, error {np.max(np.abs(plain.y[-1] - ref)):.2e}")
print(f"Rannacher start:   min value {rann.y[-1].min():+.4f}, error {np.max(np.abs(rann.y[-1] - ref)):.2e}")"""),
            md(r"""
Two backward-Euler half steps damp the sawtooth modes that the trapezoidal rule would carry forever; after
them Crank-Nicolson proceeds at second order with no ringing. The error against the exact semi-discrete
solution drops by an order of magnitude.
"""),
        ],
        [
            code(r"""lam_max = np.linalg.eigvalsh(A[1:-1, 1:-1]).min()
for h in (0.02, 0.01, 0.005, 0.0025, 0.00125):
    M = np.linalg.inv(np.eye(41) - h * A / 2) @ (np.eye(41) + h * A / 2)     # the trapezoidal one-step matrix (dense here for clarity)
    u = u0.copy()
    for _ in range(int(round(0.1 / h))):
        u = M @ u
    R = (1 + h * lam_max / 2) / (1 - h * lam_max / 2)
    print(f"h = {h:7.5f}: |R(h lambda_max)| = {abs(R):.3f}, min value {u.min():+.4f}, error {np.max(np.abs(u - ref)):.1e}")"""),
            md(r"""
The negative values disappear once |R(hλ_max)| is comfortably below 1 - at h = 0.005, where the factor is
0.88 - and the error against the exact semi-discrete solution reaches its second-order regime a step later;
this is the step at which the fastest modes are resolved rather than merely stable. With a Rannacher start
one can keep the large step instead.
"""),
        ],
    ]


@solutions(
    "03_banded_newton_and_the_cost_of_implicit_steps",
    imports=[
        "from engmolines import problems as pr, integrators as ti, linalg as la, grids, boundary as bc, operators as op"
    ],
)
def _():
    return [
        [
            code(r"""p = pr.porous_medium()
for order in (1, 2, 4):
    s = p.solve("bdf", 2e-3, order=order)
    print(f"BDF{order}: error vs Barenblatt {p.error(s):.2e}, Newton {s.newton_iterations} ({s.newton_iterations / s.steps:.1f}/step), Jacobians {s.jacobian_evaluations}")"""),
            md(r"""
At the same step the higher orders cost about the same number of Newton iterations and give similar errors:
on this problem the error is dominated by the spatial discretisation of the front, and the time order barely
matters. BDF2 is the practical choice - order enough, the smallest start-up cost.
"""),
        ],
        [
            code(r"""rows = []
for n in (16, 32, 64, 128):
    N, b = n * n, n
    rows.append((n, N, 2 * N * b**2, 2 * N**3 / 3, (2 * b + 1) * N * 8 / 1e6, N * N * 8 / 1e9))
print(pd.DataFrame(rows, columns=["n", "N = n^2", "banded LU flops", "dense LU flops", "banded factor (MB)", "dense (GB)"]).to_string(index=False, float_format="%.3g"))"""),
            md(r"""
The banded factorisation of an n × n grid's Jacobian costs 2n⁴ flops and stores (2n + 1)n² numbers; the
dense one costs 2n⁶/3 and stores n⁴. At n = 128 that is 10⁹ flops and 33 MB against 3×10¹² flops and 2 GB -
the difference between an interactive computation and an impossible one.
"""),
        ],
        [
            code(r"""def thomas(a, b, c, d):                                    # sub-, main, super-diagonal, right-hand side
    n = len(d); cp, dp = np.zeros(n), np.zeros(n)
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.zeros(n); x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x
g = grids.uniform(0, 1, 40); h = 0.005
u0 = np.sin(np.pi * g.x)
r = h / g.dx**2
a = np.r_[0.0, -r * np.ones(39), 0.0]; b = np.r_[1.0, (1 + 2 * r) * np.ones(39), 1.0]; c = np.r_[0.0, -r * np.ones(39), 0.0]
u = thomas(a, b, c, u0)                                     # one backward-Euler step
lib = ti.bdf(lambda t, v: op.diffusion(v, g, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t), u0, (0, h), h, 1, band=(1, 1), n_out=1).y[-1]
print("Thomas step vs library backward Euler:", f"{np.max(np.abs(u - lib)):.1e}")"""),
            md(r"""
The Thomas algorithm is banded LU without pivoting for a tridiagonal matrix: forward elimination of the
sub-diagonal and back substitution, 8n flops. The library's step uses pivoted banded LU through a Newton
iteration that converges in one step for this linear problem; the two agree to rounding.
"""),
        ],
    ]


@solutions(
    "04_the_heat_equation_in_every_form",
    imports=["from engmolines import problems as pr, integrators as ti, grids, boundary as bc, operators as op"],
)
def _():
    return [
        [
            code(r"""kappa, omega = 1.0, 2.0
k = np.sqrt(omega / (2 * kappa))
ex = lambda x, t: np.exp(-k * x) * np.cos(omega * t - k * x)      # noqa: E731
for L_over in (2, 8):
    L = L_over / k
    g = grids.uniform(0, L, 40 * L_over // 2)
    f = lambda t, u, g=g: op.diffusion(u, g, kappa, bc.Dirichlet(lambda t: ex(0.0, t)), bc.Neumann(0.0), t)
    s = ti.bdf(f, ex(g.x, 0.0), (0, 3 * np.pi), 0.01, 2, band=(1, 1), n_out=1)
    near = g.x < 1 / k
    print(f"domain {L_over}/k with an insulated far end: error near x = 0 (x < 1/k) {np.max(np.abs(s.y[-1][near] - ex(g.x[near], s.t[-1]))):.2e}; at the far end {abs(s.y[-1][-1] - ex(L, s.t[-1])):.2e}")"""),
            md(r"""
With the far boundary at 2/k the insulated wall reflects the thermal wave and the error near the surface is a
few percent; at 8/k the wave has decayed by e⁻⁸ before reaching it and the choice of far condition no longer
matters. The rule: put artificial boundaries several penetration depths away, or impose the decaying solution
there.
"""),
        ],
        [
            code(r"""for name, k2, L2 in (("original", 5.0, 0.5), ("ten times more conductive, a tenth as thick", 50.0, 0.05)):
    g = grids.cell_centred(0, 0.5 + L2, 200)
    kk = np.where(g.x < 0.5, 1.0, k2)
    f = lambda t, u, g=g, kk=kk: op.diffusion(u, g, kk, bc.Dirichlet(1.0), bc.Dirichlet(0.0), t)
    s = ti.bdf(f, np.zeros(200), (0, 3.0), 0.01, 2, band=(1, 1), n_out=60)
    q = [(1.0 - y[0]) / (g.dx / 2) * 1.0 for y in s.y]              # flux at the left face from the first cell
    q_ss = (1 - 0) / (0.5 / 1.0 + L2 / k2)
    t95 = s.t[np.argmax(np.array(q) <= 1.05 * q_ss)]
    print(f"{name:<44}: steady flux {q[-1]:.4f} (series resistances {q_ss:.4f}); within 5 % after t = {t95:.2f}")"""),
            md(r"""
Equal resistance means an identical steady flux, but the thin conductive layer holds far less heat (its
thermal mass is a tenth), so the wall reaches its steady state sooner: the transient is governed by the
capacities, the steady state by the resistances.
"""),
        ],
        [
            code(r"""def cyl_rhs(u, r, kappa):
    dr = r[1] - r[0]; n = len(u)
    out = np.zeros(n)
    rf = np.r_[0.0, 0.5 * (r[:-1] + r[1:])]                      # inner face radius of each control volume
    rr = np.r_[0.5 * (r[:-1] + r[1:]), r[-1] + dr / 2]            # outer face radius
    flux = np.r_[0.0, kappa * np.diff(u) / dr, 0.0]               # k du/dr at the faces (zero at the axis; outer handled by Dirichlet below)
    vol = (rr**2 - rf**2) / 2
    out[:-1] = (rr[:-1] * flux[1:-1] - rf[:-1] * flux[:-2]) / vol[:-1]
    return out                                                     # last node fixed at 0: Dirichlet
from scipy.special import j0
p = pr.cooling_cylinder(n=60); r = p.x; u0 = j0(2.404825557695773 * r)
s = ti.bdf(lambda t, u: cyl_rhs(u, r, 1.0), u0, (0, 0.2), 1e-3, 4, band=(1, 1), n_out=1)
print(f"hand-written cylindrical operator: error vs exp(-j01^2 t) J0 = {np.max(np.abs(s.y[-1] - np.exp(-2.404825557695773**2 * 0.2) * u0)):.1e}")"""),
            md(r"""
The control-volume form - fluxes through faces weighted by face radius, divided by the volume between faces -
has no 1/r to divide by at the axis: the innermost volume is a disc of radius Δr/2 with a single outer face.
It reproduces the exact Bessel decay rate like the library's operator.
"""),
        ],
    ]


@solutions(
    "05_nonlinear_diffusion",
    imports=["from engmolines import problems as pr, integrators as ti, grids, boundary as bc, operators as op"],
)
def _():
    return [
        [
            code(r"""for m in (3, 4):
    p = pr.porous_medium(n=400, m=m)
    s = p.solve("bdf", 2e-3, order=2, n_out=5)
    edges = [p.x[y > 1e-6][-1] for y in s.y]
    t0 = 0.1
    fit = np.polyfit(np.log(s.t + t0), np.log(edges), 1)[0]
    print(f"m = {m}: error vs Barenblatt {p.error(s):.1e}; front position ~ t^{fit:.3f} (theory 1/(m+1) = {1 / (m + 1):.3f})")"""),
            md(r"""
The Barenblatt profile's exponent and the front law t^(1/(m+1)) change with m, and the discretisation
follows them: the measured exponent of the front position matches the theory to the second decimal for both.
"""),
        ],
        [
            code(r"""eps = 0.01
p = pr.allen_cahn(eps=eps)
def zero_crossing(y):
    i = int(np.argmax(y > 0))
    return p.x[i - 1] - y[i - 1] * (p.x[i] - p.x[i - 1]) / (y[i] - y[i - 1])
for name, pert in (("odd perturbation 0.01 sin(pi x)", 0.01 * np.sin(np.pi * p.x)), ("even perturbation 0.01 cos(pi x / 2)", 0.01 * np.cos(np.pi * p.x / 2))):
    s = ti.bdf(p.rhs, p.y0 + pert, (0, 50.0), 0.05, 2, band=(1, 1), n_out=10)
    pos = np.array([zero_crossing(y) for y in s.y])
    shape_err = [np.max(np.abs(y - np.tanh((p.x - x0) / np.sqrt(2 * eps)))) for y, x0 in zip(s.y, pos)]
    print(f"{name}: kink position {['%.1e' % v for v in np.round(pos[::2], 12)]}; deviation from a shifted kink {['%.1e' % e for e in shape_err[::2]]}")"""),
            md(r"""
Both perturbations decay in a time of order one (the stable eigenvalues of the kink) and the profile returns
to the tanh shape to the grid's accuracy. The kink's position is a neutral mode: it shifts by the
perturbation's projection on the kink's derivative and then stays. The odd perturbation sin πx is orthogonal
to that even mode and moves the kink by nothing (10⁻¹⁷); the even perturbation shifts it, and the shift is
never undone, because nothing restores a neutral mode. The settled position's constancy over 50 time units
is the test that the scheme adds no spurious drift of its own.
"""),
        ],
        [
            code(r"""p = pr.cahn_hilliard()
g, eps = p.grid, 0.02
lap = lambda v: op.diffusion(v, g, 1.0, bc.Neumann(0.0), bc.Neumann(0.0))   # noqa: E731
rhs_hand = lambda t, u: lap(u**3 - u - eps**2 * lap(u))                      # noqa: E731
u = p.y0
print("hand-written vs library right-hand side:", f"{np.max(np.abs(rhs_hand(0, u) - p.rhs(0, u))):.1e}")
from engmolines import linalg as la
Jb = la.from_banded(la.banded_jacobian(lambda v: rhs_hand(0, v), u, 2, 2), 2, 2)
Jd = la.dense_jacobian(lambda v: rhs_hand(0, v), u)
print("band (2, 2) captures the Jacobian:", f"{np.max(np.abs(Jb - Jd)):.1e}", "; rates sum to", f"{np.sum(rhs_hand(0, u) * g.widths):.1e}", "(mass conserved)")"""),
            md(r"""
Two applications of the three-point Laplacian give a five-point operator, hence the pentadiagonal Jacobian;
the no-flux ghosts make each application conservative, so the composite is too. The library's right-hand
side is this composition.
"""),
        ],
    ]


@solutions(
    "06_reaction_diffusion_fronts_and_patterns",
    imports=[
        "from engmolines import problems as pr, integrators as ti, stiffness, grids, boundary as bc, operators as op"
    ],
)
def _():
    return [
        [
            code(r"""rows = []
for dx in (0.3, 0.15, 0.075):
    p = pr.fisher(n=int(round(60 / dx)))
    s = p.solve("bdf", 0.02, order=2, n_out=15)
    pos = np.array([pr.front_position(p.x, y) for y in s.y])
    speed = (pos[-1] - pos[-2]) / (s.t[-1] - s.t[-2])
    rows.append((dx, speed))
print(pd.DataFrame(rows, columns=["dx", "speed at t = 14.5"]).to_string(index=False))
print(f"Bramson at t = 14.5: 2 - 3/(2 t) = {2 - 3 / (2 * 14.5):.3f}")"""),
            md(r"""
The measured speed hardly changes with the grid: the front is resolved already at Δx = 0.3 (its width is of
order 1). The remaining gap to 2 is the slow approach itself - Bramson's 2 − 3/(2t) gives 1.90 at t = 14.5,
which is what the computation shows. A convergence study distinguishes a discretisation error from a slow
transient; without it the two look alike.
"""),
        ],
        [
            code(r"""for F, k, label in ((0.03, 0.062, "worms"), (0.025, 0.05, "chaotic spots")):
    p = pr.gray_scott(n=48, F=F, k=k)
    s = p.solve("rk4", 1.0, t_end=2000.0, n_out=1)
    lam = stiffness.spectrum(p.rhs, s.y[-1])
    print(f"F = {F}, k = {k} ({label}): spectrum from {lam.real.min():.3f} to {lam.real.max():+.4f}; RK4 limit {stiffness.explicit_step_limit(lam, 'rk4'):.2f}; v range [{s.y[-1][48*48:].min():.3f}, {s.y[-1][48*48:].max():.3f}]")"""),
            md(r"""
For both parameter pairs the eigenvalues stay within a few units of the origin (the diffusive part, 4D/Δx²
≈ 0.2, is tiny), so RK4's step limit is of order one and h = 1 remains adequate - as long as the grid is
coarse. On a finer grid the diffusive eigenvalues grow like 1/Δx² and explicit integration eventually becomes
the costlier choice; the spectrum, recomputed, says when.
"""),
        ],
        [
            code(r"""p = pr.fisher()
g = p.grid
fe = p.params["rhs_explicit"]; fi = p.params["rhs_implicit"]
from engmolines import linalg as la
A = la.from_banded(la.banded_jacobian(lambda w: fi(0.0, w), p.y0, 1, 1), 1, 1)       # the linear diffusion operator
h = 0.02; u = p.y0.copy(); fe_prev = None
M = np.eye(len(u)) - 0.5 * h * A
for k in range(int(round(15 / h))):
    f_now = fe(0, u)
    if fe_prev is None:
        rhs = u + h * f_now                                   # first step: IMEX Euler (no history)
        u = np.linalg.solve(np.eye(len(u)) - h * A, rhs)
    else:
        rhs = u + h * (1.5 * f_now - 0.5 * fe_prev) + 0.5 * h * (A @ u)
        u = np.linalg.solve(M, rhs)
    fe_prev = f_now
lib = p.solve("imex", h, n_out=1).y[-1]
print(f"hand-written CNAB2 vs library: front at {pr.front_position(p.x, u):.3f} vs {pr.front_position(p.x, lib):.3f}; max difference {np.max(np.abs(u - lib)):.1e}")"""),
            md(r"""
CNAB2 applies the trapezoidal rule to the linear diffusion (one fixed matrix I − hA/2) and second-order
Adams-Bashforth to the reaction; the first step, lacking history, is an IMEX Euler step. The hand-written
loop reproduces the library's front.
"""),
        ],
    ]


@solutions(
    "07_advection_the_hard_case",
    imports=[
        "from engmolines import problems as pr, integrators as ti, analysis, stiffness, grids, boundary as bc, operators as op"
    ],
)
def _():
    return [
        [
            code(r"""dx = 1 / 200
for stencil, method, nus in (("first_upwind", "euler", (0.5, 1.0, 1.1)), ("first_central", "rk4", (2.5, 3.0))):
    for nu in nus:
        amp = np.max(analysis.amplification(stencil, method, nu * dx, dx, sign=-1.0))
        g = grids.cell_centred(0, 1, 200)
        sch = "upwind" if stencil == "first_upwind" else "centred"
        f = lambda t, u, sch=sch: op.advection(u, g, 1.0, bc.Periodic(), bc.Periodic(), t, sch)
        tab = ti.EULER if method == "euler" else ti.RK4
        s = ti.explicit_rk(f, np.exp(-((g.x - 0.3) / 0.05) ** 2), (0, 2.0), nu * dx, tab, n_out=1)
        print(f"{stencil:<13} + {method}: Courant {nu}: max amplification {amp:.4f}; max |u| after t = 2 ({s.steps} steps): {np.max(np.abs(s.y[-1])):.3g}")"""),
            md(r"""
The von Neumann factor is at most 1 exactly up to the Courant limits (1 for upwind + Euler, 2.83 for centred +
RK4) and exceeds it beyond; the integrations confirm it: bounded below the limit, growing above - though the
growth needs enough steps to amplify the sawtooth from round-off, which is why the runs go to t = 2 (a run
to t = 0.5 at Courant 3 still looks innocent). The analysis and the experiment agree because the problem is
linear with periodic boundaries - the case von Neumann's method covers exactly.
"""),
        ],
        [
            code(r"""Pe, Da = 10.0, 2.0
g = grids.uniform(0, 1, 100)
def rhs_reverse(t, c, feed=lambda t: 1.0):
    left, right = bc.Neumann(0.0), bc.Robin(1.0, 1 / Pe, feed(t))           # feed at x = 1 for flow to the left: c + c_x/Pe = c_feed
    return op.advection(c, g, -1.0, left, right, t, "upwind") + op.diffusion(c, g, 1 / Pe, left, right, t) - Da * c
s = ti.bdf(rhs_reverse, np.zeros(101), (0, 4.0), 0.01, 2, band=(1, 1), n_out=1)
print(f"reversed flow: outlet (x = 0) conversion {1 - s.y[-1][0]:.4f} (same Danckwerts value as before: {1 - pr.plug_flow_reactor().exact(np.array([1.0]))[0]:.4f})")
pulse = lambda t: 1.0 if 0.5 < t < 1.0 else 0.0      # noqa: E731
s = ti.bdf(lambda t, c: rhs_reverse(t, c, pulse), np.zeros(101), (0, 3.0), 0.005, 2, band=(1, 1), n_out=6)
print("outlet concentration over time:", np.round(s.y[:, 0], 4))
def rhs_wrong(t, c):
    out = rhs_reverse(t, c)
    out[0] = 0.0                                             # (wrongly) pinning the outflow value at 0 too
    return out
s = ti.bdf(rhs_wrong, np.zeros(101), (0, 4.0), 0.01, 2, band=(1, 1), n_out=1)
print(f"with a Dirichlet value forced at the outflow: concentration next to it {s.y[-1][1]:.4f} - a spurious boundary layer, flux inconsistent with the balance")"""),
            md(r"""
Reversing the flow is a matter of swapping the inflow condition to the other end and the upwind direction
follows; the conversion is the same. A pulse in the feed exits after the residence time, dispersed. Forcing a
value at the outflow over-determines the hyperbolic part and produces an unphysical boundary layer in the
last cells: a hyperbolic equation takes its boundary data where the characteristics enter, and nowhere else.
"""),
        ],
        [
            code(r"""g = grids.cell_centred(0, 1, 200); dx = g.dx
def upwind2(u):                                            # second-order upwind-biased: (3 u_i - 4 u_{i-1} + u_{i-2}) / (2 dx)
    return -(3 * u - 4 * np.roll(u, 1) + np.roll(u, 2)) / (2 * dx)
me = analysis.modified_equation("upwind2", 1.0, dx)
print(f"modified equation: diffusion {me['diffusion']}, dispersion {me['dispersion']:.3e} u_xxx (and a fourth-derivative damping {me['fourth']:.2e})")
u0 = np.sin(2 * np.pi * g.x)
s = ti.dopri45(lambda t, u: upwind2(u), u0, (0, 1.0), 1e-9, 1e-11, n_out=1)
phase = np.angle(np.fft.fft(s.y[-1])[1]) - np.angle(np.fft.fft(u0)[1])
print(f"after one period the sine wave is shifted by {-phase / (2 * np.pi):.4f} wavelengths (dispersion: the predicted phase speed error is a dx^2 k^2/3 = {dx**2 * (2 * np.pi)**2 / 3:.4f})")
print(f"amplitude {np.abs(np.fft.fft(s.y[-1])[1]) / np.abs(np.fft.fft(u0)[1]):.4f} (slight damping from the fourth-derivative term)")"""),
            md(r"""
The three-point upwind-biased stencil has no second-derivative error term - no numerical diffusion - but a
third-derivative one: it is dispersive, and a sine wave is advected at the wrong phase speed, by a Δx²k²/3
relative error that the measured phase shift reproduces. The small amplitude loss comes from the next term, a
fourth-derivative damping. Second order upwinding trades diffusion for dispersion, and limiters are needed
to tame the resulting oscillations at jumps.
"""),
        ],
    ]


@solutions(
    "08_shocks_and_conservation_laws",
    imports=["from engmolines import problems as pr, integrators as ti, grids, operators as op"],
)
def _():
    return [
        [
            code(r"""g = grids.cell_centred(-3, 6, 600)
u0 = np.exp(-g.x**2)
rhs = lambda t, u: op.conservation_law(u, g, lambda w: 0.5 * w * w, lambda w: np.abs(w), "transmissive", "transmissive", t)   # noqa: E731
s = ti.dopri45(rhs, u0, (0, 4.0), 1e-7, 1e-9, n_out=4)
for t, y in zip(s.t, s.y):
    plt.plot(g.x, y, label=f"t = {t:.0f}")
plt.legend(); plt.show()
print(f"integral of u: {[round(g.integrate(y), 6) for y in s.y]} (conserved: the bump never reaches the boundaries)")
t_b = 1 / np.max(-np.gradient(u0, g.x))
print(f"breaking time 1/max(-u0') = {t_b:.3f}; at t = 4 the shock sits at x = {g.x[np.argmax(np.diff(s.y[-1]) < -0.05 * s.y[-1].max())]:.2f}")
print("the shock position follows from the equal-area rule on the multivalued characteristic solution; the integral of u is conserved exactly by the finite-volume scheme")"""),
            md(r"""
A smooth bump breaks at t = 1/max(−u₀′) ≈ 1.17 and the shock then moves so that the area under the
multivalued characteristic solution is preserved (the equal-area rule, the geometric form of
Rankine-Hugoniot). The finite-volume scheme conserves the integral exactly and places the shock accordingly;
as the shock weakens the solution tends to an N-wave of decaying amplitude.
"""),
        ],
        [
            code(r"""for hR in (1e-3,):
    p = pr.shallow_water_dam_break(hR=hR)
    try:
        s = p.solve("dopri", rtol=1e-7, atol=1e-9, n_out=1)
        h = s.y[-1][:400]
        front = p.x[len(p.x) - 1 - np.argmax(h[::-1] > 2 * hR)]
        print(f"hR = {hR}: front at x = {front:.3f} after t = 0.2 (dry-bed limit 2 sqrt(g hL) t = {2 * np.sqrt(9.81) * 0.2:.3f}); min h {h.min():.1e}")
    except Exception as e:
        print(f"hR = {hR}: {type(e).__name__}: {e}")
print("with hR = 0 exactly the flux u = q/h divides by zero: real codes use a wetting/drying threshold and desingularised velocities")"""),
            md(r"""
Even with the bed a thousandth as deep the front is still a bore moving at 4.7, well below the dry-bed limit
2√(gh_L) = 6.3 of a pure rarefaction: the limit is approached only logarithmically as h_R → 0, because a thin
layer of water still has to be pushed aside at a shock. At h_R = 0 exactly the velocity q/h is undefined; the
Rusanov scheme as written fails, and production shallow-water codes carry a wet/dry threshold, a
desingularised velocity u = 2hq/(h² + max(h², ε)²), and positivity-preserving limiters.
"""),
        ],
        [
            code(r"""def godunov_flux(uL, uR):                                      # exact Riemann solver for Burgers' flux u^2/2
    f = lambda u: 0.5 * u * u  # noqa: E731
    s = 0.5 * (uL + uR)
    shock = np.where(s > 0, f(uL), f(uR))
    rare = np.where(uL > 0, f(uL), np.where(uR < 0, f(uR), 0.0))
    return np.where(uL > uR, shock, rare)
g = grids.cell_centred(-1, 1, 400)
u0 = np.where(g.x < 0, 1.0, 0.0)
def rhs_godunov(t, u):
    up = np.r_[u[0], u, u[-1]]
    F = godunov_flux(up[:-1], up[1:])
    return -np.diff(F) / g.dx
def rhs_rusanov1(t, u):                                         # first-order Rusanov (no reconstruction) for a fair comparison
    up = np.r_[u[0], u, u[-1]]
    uL, uR = up[:-1], up[1:]
    F = 0.5 * (0.5 * uL**2 + 0.5 * uR**2) - 0.5 * np.maximum(np.abs(uL), np.abs(uR)) * (uR - uL)
    return -np.diff(F) / g.dx
exact = np.where(g.x < 0.4, 1.0, 0.0)
for name, f in (("Godunov", rhs_godunov), ("Rusanov (first order)", rhs_rusanov1)):
    s = ti.dopri45(f, u0, (0, 0.8), 1e-7, 1e-9, n_out=1)
    width = np.sum((s.y[-1] > 0.05) & (s.y[-1] < 0.95)) * g.dx
    print(f"{name:<22}: shock smeared over {width:.3f} (= {width / g.dx:.0f} cells); L1 error {np.mean(np.abs(s.y[-1] - exact)):.4f}")"""),
            md(r"""
Godunov's flux uses the exact solution of each face's Riemann problem and adds the least dissipation
consistent with the entropy condition; Rusanov's uses the largest wave speed as a uniform dissipation and
smears the shock over more cells. Both are first order here (no reconstruction); the library's MUSCL
reconstruction sharpens either.
"""),
        ],
    ]


@solutions(
    "09_waves_strings_and_beams",
    imports=["from engmolines import problems as pr, integrators as ti, analysis, operators as op, boundary as bc"],
)
def _():
    return [
        [
            code(r"""p = pr.wave_string(mode=2); n = 101; T = p.t_end
tz = ti.trapezoidal(p.rhs, p.y0, (0, 20 * T), 0.005, band=(n + 1, n + 1), n_out=20)
sv = ti.verlet(p.params["accel"], p.y0[:n], p.y0[n:], (0, 20 * T), 0.005, n_out=20)
for name, s in (("trapezoidal", tz), ("Verlet", sv)):
    E = np.array([analysis.wave_energy(y[:n], y[n:], 1.0, p.grid) for y in s.y])
    print(f"{name:<12}: energy change over 20 periods {E[-1] / E[0] - 1:+.2e}; position error {np.max(np.abs(s.y[-1][:n] - p.exact(p.x, s.t[-1]))):.2e}")"""),
            md(r"""
The trapezoidal rule has |R(iy)| = 1 exactly: it conserves the energy of a linear wave to rounding, like
Verlet. Both make a phase error of order (hω)², so their position errors after twenty periods are similar;
the trapezoidal rule is implicit and A-stable (any step is stable, if inaccurate) while Verlet is explicit
with the limit hω ≤ 2 - for undamped linear waves either is a sound choice, and BDF is not.
"""),
        ],
        [
            code(r"""from engmolines import grids
L, n = 1.0, 60
g = grids.uniform(0, L, n)
accel = lambda t, w: -op.biharmonic(w, g, "clamped", "free")   # noqa: E731
w0 = (g.x / L) ** 2 * (3 - 2 * g.x / L) * 0.01                 # a smooth clamped-free deflection as the start
s = ti.verlet(accel, w0, np.zeros(n + 1), (0, 20.0), 0.4 * g.dx**2, n_out=4000)
tip = s.y[:, n]
freqs = np.fft.rfftfreq(len(tip), s.t[1] - s.t[0])
spec = np.abs(np.fft.rfft(tip - tip.mean()))
f1 = freqs[np.argmax(spec[1:]) + 1]
print(f"first natural frequency from the tip's spectrum: {f1:.4f} Hz; theory (1.875)^2 sqrt(EI/rhoA)/(2 pi L^2) = {1.8751**2 / (2 * np.pi):.4f} Hz")"""),
            md(r"""
A clamped-free beam's tip displacement, Fourier-transformed, peaks at the first natural frequency
(βL)²√(EI/ρA)/(2πL²) with βL = 1.875 - within the resolution of a 20-time-unit record. The `biharmonic`
operator's free-end ghosts impose w″ = w‴ = 0 at the tip.
"""),
        ],
        [
            code(r"""p = pr.telegraph(a=0.5); n = 101; h = 0.005
u, v = p.y0[:n].copy(), p.y0[n:].copy()
accel = lambda t, w: op.diffusion(w, p.grid, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t)   # noqa: E731
a = 0.5
acc = accel(0, u)
for k in range(int(round(3.0 / h))):
    v_half = (v + 0.5 * h * acc) / (1 + a * h)             # damping -2 a v treated implicitly over the half step
    u = u + h * v_half
    acc = accel(0, u)
    v = (v_half + 0.5 * h * acc) / (1 + a * h)
    u[0] = u[-1] = 0.0; v[0] = v[-1] = 0.0
print(f"damped Verlet: error vs e^(-at) sin(pi x) cos(omega t) = {np.max(np.abs(u - p.exact(p.x, 3.0))):.2e} (the amplitude has decayed by e^(-3a) = {np.exp(-1.5):.4f})")"""),
            md(r"""
Treating the damping implicitly in each velocity half-update keeps the scheme explicit in the stiff wave
operator and unconditionally stable in the damping, and preserves second order; the decay rate e^(−at) of
the telegraph solution is reproduced.
"""),
        ],
    ]


@solutions(
    "10_dispersive_waves_and_spectral_methods",
    imports=["from engmolines import problems as pr, integrators as ti, spectral"],
)
def _():
    return [
        [
            code(r"""p = pr.kdv_soliton(); n, L = 256, 40.0
k = spectral.wavenumbers(n, L)
E = -(1j * k) ** 3                                           # the linear operator -u_xxx in Fourier space: multiply by -(ik)^3
def rhs_if(t, vhat_ri):                                     # integrating factor: v = exp(-E t) u_hat; v' = exp(-E t) N(u)
    vhat = vhat_ri[:n] + 1j * vhat_ri[n:]
    uhat = np.exp(E * t) * vhat
    u = np.fft.ifft(uhat).real
    N = np.fft.fft(-6 * u * spectral.derivative(u, L, 1))
    d = np.exp(-E * t) * N
    return np.r_[d.real, d.imag]
v0 = np.fft.fft(p.y0)
s = ti.dopri45(rhs_if, np.r_[v0.real, v0.imag], (0, 4.0), 1e-8, 1e-10, n_out=1)
u_end = np.fft.ifft(np.exp(E * 4.0) * (s.y[-1][:n] + 1j * s.y[-1][n:])).real
print(f"integrating factor: {s.steps} steps (mean h {np.mean(s.extra['h']):.1e}) vs {p.solve('dopri', rtol=1e-8, atol=1e-10).steps} without; error {np.max(np.abs(u_end - p.exact(p.x, 4.0))):.1e}")"""),
            md(r"""
Solving the linear dispersive part exactly by the integrating factor removes its stiffness from the explicit
integrator, which now sees only the nonlinear term: the step is set by the soliton's own time scale and the
step count drops by an order of magnitude at the same accuracy. (The exponential grows with t and is better
re-centred every step in a production code; here t ≤ 4 and it is harmless.)
"""),
        ],
        [
            code(r"""n, L = 256, 40.0
x = spectral.grid(-L / 2, L / 2, n)
for A in (1.0, 2.0):
    psi0 = A / np.cosh(A * x) + 0j                             # A sech(Ax) exp(i A^2 t) is a soliton of i psi_t + psi_xx + 2|psi|^2 psi = 0
    def rhs(t, y):
        psi = y[:n] + 1j * y[n:]
        d = 1j * (spectral.derivative(psi, L, 2) + 2 * np.abs(psi) ** 2 * psi)
        return np.r_[d.real, d.imag]
    s = ti.dopri45(rhs, np.r_[psi0.real, psi0.imag], (0, 2.0), 1e-8, 1e-10, n_out=1)
    psi = s.y[-1][:n] + 1j * s.y[-1][n:]
    print(f"amplitude {A}: {s.steps} steps; error vs A sech(Ax) e^(i A^2 t) {np.max(np.abs(psi - psi0 * np.exp(1j * A**2 * 2.0))):.1e}")"""),
            md(r"""
The amplitude-2 soliton oscillates at frequency A² = 4, four times faster, and its width is half - yet the
integrator needs only about 10 % more steps. The reason is that the step was already set by the spectral
stiffness of ψ_xx (eigenvalues up to k_max² ≈ 400, the same in both runs) rather than by the solution's own
frequency; the faster soliton merely raises the accuracy requirement a little. Only when the solution's
frequency approaches the stiffness-limited rate does the step count rise in proportion.
"""),
        ],
        [
            code(r"""for dealiased in (False, True):
    for n in (64, 256):
        p = pr.kdv_two_solitons(n=n); L = 40.0
        def rhs(t, u, dealiased=dealiased):
            ux = spectral.derivative(u, L, 1)
            prod = spectral.dealias(spectral.dealias(u) * spectral.dealias(ux)) if dealiased else u * ux
            return -6 * prod - spectral.derivative(u, L, 3)
        with np.errstate(all="ignore"):
            s = ti.dopri45(rhs, p.y0, (0, 8.0), 1e-8, 1e-10, n_out=1, max_steps=200000)
        mom = p.grid.integrate(s.y[-1] ** 2) - p.params["momentum"] if np.all(np.isfinite(s.y[-1])) else np.nan
        print(f"{'dealiased' if dealiased else 'plain    '} n = {n:3d}: reached t = {s.t[-1]:.1f}, momentum drift {mom:+.2e}, max |u| {np.max(np.abs(s.y[-1])):.3g}")"""),
            md(r"""
At n = 64 the plain product aliases enough energy into the resolved modes to blow the run up; the 2/3-rule
product keeps it stable, though with 64 points the solitons are still poorly resolved and the invariants
drift. At n = 256 both are exact to rounding. Dealiasing removes a *spurious* instability; it does not supply
resolution.
"""),
        ],
    ]


@solutions(
    "11_coupled_systems_and_daes",
    imports=["from engmolines import problems as pr, integrators as ti, grids, boundary as bc, operators as op"],
)
def _():
    return [
        [
            code(r"""n, K, q_max, k_ldf, eps, u_, D = 100, 5.0, 2.0, 5.0, 0.4, 1.0, 1e-3
g = grids.uniform(0, 1, n)
left, right = bc.Robin(1.0, -D / u_, 1.0), bc.Neumann(0.0)
def column(isotherm):
    def rhs(t, y):
        c, q = y[: n + 1], y[n + 1 :]
        dq = k_ldf * (isotherm(c) - q)
        return np.r_[op.advection(c, g, u_, left, right, t, "upwind") + op.diffusion(c, g, D, left, right, t) - (1 - eps) / eps * dq, dq]
    return ti.bdf(rhs, np.zeros(2 * (n + 1)), (0, 15.0), 0.02, 2, band=(n + 2, n + 2), n_out=150)
for name, iso in (("linear q* = K c", lambda c: K * c), ("Langmuir q* = q_max K c/(1 + K c)", lambda c: q_max * K * c / (1 + K * c))):
    s = column(iso)
    c_out = s.y[:, n]
    t10, t90 = np.interp(0.1, c_out, s.t), np.interp(0.9, c_out, s.t)
    print(f"{name:<34}: breakthrough from 10 % to 90 % of the feed takes {t90 - t10:.2f} time units (midpoint at {np.interp(0.5, c_out, s.t):.2f})")"""),
            md(r"""
A favourable (Langmuir) isotherm makes the high concentrations travel faster than the low ones (the
front is self-sharpening and reaches a constant pattern), so the breakthrough curve is steeper than with the
linear isotherm, whose front spreads diffusively. The method of lines handles the nonlinear isotherm by
putting it in the uptake term; nothing else changes.
"""),
        ],
        [
            code(r"""phi, Sh = 3.0, 5.0
g = grids.radial(1.0, 60, "spherical"); n = g.n; dx = g.dx
def F(t, y, yp):
    c = y
    res = np.empty(n)
    interior = op.diffusion(c, g, 1.0, None, bc.Dirichlet(c[-1]), t, source=lambda x, c, t: -(phi**2) * c)     # the Dirichlet value is c's own last entry: no constraint from the operator
    res[:-1] = yp[:-1] - interior[:-1]
    res[-1] = Sh * (1 - c[-1]) - (3 * c[-1] - 4 * c[-2] + c[-3]) / (2 * dx)        # film resistance: Sh (1 - c_s) = c_r(1)
    return res
y0 = np.zeros(n); y0[-1] = Sh / (Sh + 1)
s = ti.dae_bdf(F, y0, (0, 2.0), 0.005, order=2, band=(2, 2))
c = s.y[-1]
flux = (3 * c[-1] - 4 * c[-2] + c[-3]) / (2 * dx)
eta_film = 3 * flux / phi**2
eta_exact = 3 * (phi / np.tanh(phi) - 1) / phi**2 / (1 + (phi / np.tanh(phi) - 1) / Sh)      # with film resistance
print(f"surface concentration {c[-1]:.4f} (below 1: the film); effectiveness {eta_film:.4f} vs the exact film-resistance value {eta_exact:.4f}; without film {3 * (phi / np.tanh(phi) - 1) / phi**2:.4f}")"""),
            md(r"""
The film condition Sh(1 − c_s) = c_r at the surface is an algebraic equation in the surface unknown; written
as the last row of a DAE it is enforced exactly at every step, and the effectiveness factor drops by the
factor 1/(1 + (φ coth φ − 1)/Sh) of the series resistances - which the DAE solution reproduces.
"""),
        ],
        [
            code(r"""n, eps, k_ldf, u_, D = 100, 0.4, 5.0, 1.0, 1e-3
Ks = (2.0, 8.0)
g = grids.uniform(0, 1, n)
left, right = bc.Robin(1.0, -D / u_, 1.0), bc.Neumann(0.0)
def rhs(t, y):
    out = []
    for j, K in enumerate(Ks):
        c, q = y[2 * j * (n + 1):(2 * j + 1) * (n + 1)], y[(2 * j + 1) * (n + 1):(2 * j + 2) * (n + 1)]
        dq = k_ldf * (K * c - q)
        out += [op.advection(c, g, u_, left, right, t, "upwind") + op.diffusion(c, g, D, left, right, t) - (1 - eps) / eps * dq, dq]
    return np.concatenate(out)
s = ti.bdf(rhs, np.zeros(4 * (n + 1)), (0, 20.0), 0.02, 2, band=(n + 2, n + 2), n_out=200)
for j, K in enumerate(Ks):
    c_out = s.y[:, 2 * j * (n + 1) + n]
    print(f"species with K = {K}: breakthrough at t = {np.interp(0.5, c_out, s.t):.2f} (stoichiometric 1 + (1-eps)K/eps = {1 + (1 - eps) / eps * K:.2f})")"""),
            md(r"""
Two independently adsorbing species travel at their own stoichiometric speeds, u/(1 + (1 − ε)K/ε), and
break through at different times - the principle of chromatography. With independent isotherms the species
do not interact; the state vector simply stacks both pairs and the band structure repeats.
"""),
        ],
    ]


@solutions(
    "12_two_dimensions",
    imports=[
        "import time",
        "from engmolines import problems as pr, integrators as ti, linalg as la, grids, boundary as bc, operators as op",
    ],
)
def _():
    return [
        [
            code(r"""rows = []
for n in (12, 24, 48):
    p = pr.plate_heat_2d(n=n)
    f = lambda u, p=p: p.rhs(0.0, u)
    t0 = time.perf_counter(); Jb = la.banded_jacobian(f, p.y0, *p.band); t_jac = time.perf_counter() - t0
    M = -0.01 * Jb; M[p.band[1], :] += 1.0                                   # I - h J in banded storage, as the Newton iteration factorises it
    t0 = time.perf_counter(); la.lu_banded(M, *p.band); t_lu = time.perf_counter() - t0
    N = p.n
    t0 = time.perf_counter(); np.linalg.solve(np.eye(N) - 0.01 * la.from_banded(Jb, *p.band), np.ones(N)); t_dense = time.perf_counter() - t0
    rows.append((n, N, p.band[0], t_jac, t_lu, t_dense))
tab = pd.DataFrame(rows, columns=["n", "N", "band", "coloured Jacobian (s)", "banded LU (s)", "dense solve (s)"])
print(tab.to_string(index=False, float_format="%.4f"))
print("banded LU time ratios per doubling of n:", np.round(tab["banded LU (s)"].to_numpy()[1:] / tab["banded LU (s)"].to_numpy()[:-1], 1), "(n^4 law: 16)")"""),
            md(r"""
The banded factorisation's time grows by roughly the n⁴ factor of 16 per doubling (measured ratios vary
with the machine and the Python overhead of the small cases); the dense solve grows like n⁶ and is already
slower at n = 48 despite running in LAPACK. For larger grids the banded structure is not an optimisation
but the difference between feasible and not.
"""),
        ],
        [
            code(r"""ghia400 = {"u_min": -0.3273, "y": 0.2813}
for n in (32, 48):
    p = pr.cavity_flow(n=n, Re=400.0)
    s = p.solve("rk4", 0.004 if n == 32 else 0.002, t_end=40.0, n_out=8)
    omega = s.y[-1].reshape(n + 1, n + 1); psi = p.params["streamfunction"](omega); u, v = p.params["velocity"](psi)
    uc = u[n // 2, :]
    change = np.max(np.abs(s.y[-1] - s.y[-2])) / np.max(np.abs(s.y[-1]))
    print(f"Re = 400, {n} x {n}: u_min {uc.min():.4f} at y = {p.x[np.argmin(uc)]:.3f} (Ghia {ghia400['u_min']} at {ghia400['y']}); relative change over the last 5 time units {change:.1e}")"""),
            md(r"""
At Re = 400 the primary vortex is stronger and its centre lower; the coarse grid under-resolves the thin
boundary layers and under-predicts u_min, the 48 × 48 grid comes closer. The approach to steady state is
slower than at Re = 100 (the viscous time scale is four times longer), so the run needs about 40 time
units.
"""),
        ],
        [
            code(r"""n = 20
g2 = grids.tensor(grids.uniform(0, 1, n), grids.uniform(0, 1, n))
bcs = {k: bc.Neumann(0.0) for k in ("left", "right", "bottom", "top")}
f = lambda t, y: op.laplacian2d(y.reshape(n + 1, n + 1), g2, bcs, t).ravel()         # noqa: E731
rng = np.random.default_rng(0)
u0 = rng.random((n + 1) ** 2)
s = ti.bdf(f, u0, (0, 0.5), 0.01, 2, band=(n + 1, n + 1), n_out=5)
heat = [g2.integrate(y.reshape(n + 1, n + 1)) for y in s.y]
print("total heat over time:", np.round(heat, 10), "-> conserved")
Jb = la.from_banded(la.banded_jacobian(lambda y: f(0.0, y), u0, n + 1, n + 1), n + 1, n + 1)
print(f"band n + 1 = {n + 1} captures the Jacobian: max difference from the dense one {np.max(np.abs(Jb - la.dense_jacobian(lambda y: f(0.0, y), u0))):.1e}")"""),
            md(r"""
Neumann ghosts on all four sides make the 2-D operator conservative - the total heat of a random initial field
stays fixed as it diffuses to the uniform mean - and the Jacobian's band is n + 1, as the Kronecker structure
predicts, which the coloured Jacobian reproduces exactly.
"""),
        ],
    ]


@solutions(
    "13_diagnosing_failures",
    imports=[
        "from engmolines import problems as pr, integrators as ti, stiffness, analysis, grids, boundary as bc, operators as op, spectral"
    ],
)
def _():
    return [
        [
            code(r"""def heat_neumann_firstorder(n):
    g = grids.uniform(0, 1, n); dx = g.dx
    def rhs(t, u):
        up = np.r_[u[0], u, u[-1]]                     # first-order ghosts: u_ghost = u_0 (zero slope by a one-sided difference)
        return (up[2:] - 2 * up[1:-1] + up[:-2]) / dx**2
    return g, rhs
errs, cons = [], []
for n in (20, 40, 80):
    g, rhs = heat_neumann_firstorder(n)
    u0 = np.cos(np.pi * g.x)
    s = ti.bdf(rhs, u0, (0, 0.2), 1e-3, 4, band=(1, 1), n_out=1)
    errs.append(np.max(np.abs(s.y[-1] - np.cos(np.pi * g.x) * np.exp(-np.pi**2 * 0.2))))
    cons.append(abs(g.integrate(s.y[-1]) - g.integrate(u0)))
print("errors:", ["%.2e" % e for e in errs], "orders", np.round(analysis.observed_order(errs), 2), "(the centred ghost gives 2)")
print("conservation errors:", ["%.1e" % c for c in cons], "(the centred ghost gives 1e-16)")"""),
            md(r"""
The first-order ghost makes the whole solution first-order accurate - the boundary error dominates - and,
more visibly, it breaks conservation: the flux through the end is no longer exactly zero, so the heat content
drifts. The conservation check reveals the problem from the first run; the order study needs three.
"""),
        ],
        [
            code(r"""p = pr.fisher(n=200)
lam = stiffness.spectrum(p.rhs, p.y0, band=(1, 1)); lam = lam[np.abs(lam) > 1e-9]
h_lim = stiffness.explicit_step_limit(lam, "rk4")
with np.errstate(all="ignore"):
    s = ti.explicit_rk(p.rhs, p.y0, (0, 60 * 1.05 * h_lim), 1.05 * h_lim, n_out=6)
growth = [np.max(np.abs(y - p.y0)) for y in s.y]
print(f"RK4 at 1.05 x the limit (h = {1.05 * h_lim:.4f}): deviation from the initial state every 10 steps {['%.1e' % g for g in growth]}")
R = abs(stiffness.stability_function("rk4", 1.05 * h_lim * lam[0].real))
print(f"|R(h lambda_max)| = {R:.3f} -> expected growth per 10 steps {R**10:.2f}x; the sawtooth appears first where the solution is roughest (the step's edge)")"""),
            md(r"""
Just above the limit the sawtooth mode grows by |R(hλ_max)| = 1.23 per step, eightfold per ten steps once
it dominates; at first the deviation shown is mostly the physical evolution of the front, and the geometric
growth takes over after a few dozen steps and ends in overflow. The sawtooth is seeded where the solution has
the most high-frequency content, at the step in the initial data. The symptom - a grid-scale oscillation
growing geometrically, independent of the physics - identifies a time-step instability as surely as the
spectrum does.
"""),
        ],
        [
            code(r"""def diagnose(rhs, y, band=None):
    lam = stiffness.spectrum(rhs, y, band=band); lam = lam[np.abs(lam) > 1e-9]
    decaying = lam[lam.real < 0]
    return {"max Re": float(lam.real.max()), "|lambda| ratio": float(np.abs(lam).max() / np.abs(lam).min()), "RK4 limit": stiffness.explicit_step_limit(decaying, "rk4"),
            "Euler limit": stiffness.explicit_step_limit(decaying, "euler"), "imaginary-axis modes": bool(np.any((np.abs(lam.real) < 1e-6) & (np.abs(lam.imag) > 1e-6)))}
rows = []
for make in (pr.heat_dirichlet, pr.advection_periodic, pr.fisher, pr.wave_string, pr.cahn_hilliard):
    p = make()
    d = diagnose(p.rhs, p.y0, p.band)
    rows.append((p.name, *d.values()))
print(pd.DataFrame(rows, columns=["problem", "max Re", "|lambda| ratio", "RK4 limit", "Euler limit", "imaginary modes"]).to_string(index=False, float_format="%.3g"))"""),
            md(r"""
One function, five diagnoses: the heat equation is stiff and real; the string sits on the imaginary axis (no
Euler step is stable; RK4's is set by the CFL condition) and the limited advection scheme's nonlinear
Jacobian sits near it; Fisher has a positive real part (the unstable state u = 0 that the front invades, a
physical instability); Cahn-Hilliard, linearised about a random mixture, has large positive eigenvalues - the
spinodal instability that drives phase separation - and is extremely stiff. Running such a function before
choosing an integrator is the discipline of the course.
"""),
        ],
    ]


@solutions(
    "14_moving_boundaries_and_transformed_domains",
    imports=[
        "from engmolines import problems as pr, integrators as ti, grids, boundary as bc, stencils as st, stiffness"
    ],
)
def _():
    return [
        [
            code(r"""K, r, sigma, T = 100.0, 0.05, 0.2, 1.0
from scipy.special import ndtr
def bs(S, tau):
    with np.errstate(divide="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return np.where(S > 0, S * ndtr(d1) - K * np.exp(-r * tau) * ndtr(d2), 0.0)
n = 400
for name, grid_kind in (("uniform in S", "S"), ("uniform in z = log S", "z")):
    if grid_kind == "S":
        p = pr.black_scholes(n=n); s = p.solve("bdf", 1e-3, order=2, n_out=1); S, V = p.x, s.y[-1]
    else:
        z = np.linspace(np.log(1.0), np.log(400.0), n + 1); dz = z[1] - z[0]; S = np.exp(z)
        def rhs(tau, V):
            Vp = np.r_[2 * V[0] - V[1], V, 2 * V[-1] - V[-2]]
            Vz = (Vp[2:] - Vp[:-2]) / (2 * dz); Vzz = (Vp[2:] - 2 * Vp[1:-1] + Vp[:-2]) / dz**2
            out = 0.5 * sigma**2 * Vzz + (r - 0.5 * sigma**2) * Vz - r * V         # constant coefficients in z
            out[0] = 0.0; out[-1] = r * K * np.exp(-r * tau)
            return out
        V = ti.bdf(rhs, np.maximum(S - K, 0.0), (0, T), 1e-3, 2, band=(1, 1), n_out=1).y[-1]
    near = (S > 80) & (S < 120)
    print(f"{name:<22}: max error {np.max(np.abs(V - bs(S, T))[(S > 1) & (S < 390)]):.2e}; near the strike (80 < S < 120) {np.max(np.abs(V - bs(S, T))[near]):.2e}; spacing at the strike {np.diff(S)[np.argmin(np.abs(S - K))]:.2f}")"""),
            md(r"""
In z = log S the equation has constant coefficients, but a grid uniform in z from S = 1 to 400 is *coarser*
at the strike (spacing 1.5 against 1.0) because it spends half its points below S = 20, where the option is
worthless; its error near the strike is nevertheless slightly smaller, as the constant-coefficient operator
is better behaved. Neither grid is good: the error lives at the strike's kink, and a grid stretched around
S = K is what a pricing code uses.
"""),
        ],
        [
            code(r"""for St in (0.1, 10.0):
    p = pr.stefan(St=St)
    s = p.solve("dopri", rtol=1e-8, atol=1e-10, n_out=1)
    lam = stiffness.spectrum(p.rhs, p.y0); lam = lam[np.abs(lam) > 1e-9]
    print(f"St = {St:4}: lambda = {p.params['lambda']:.4f}, front at t = 1: {s.y[-1, -1]:.4f} (2 lambda sqrt(t) = {2 * p.params['lambda'] * np.sqrt(1 + p.params['t0']):.4f}); {s.steps} steps; most negative eigenvalue {lam.real.min():.0f} (diffusion on xi/s: ~ -4/(dxi s)^2)")"""),
            md(r"""
In this formulation the Stefan number divides the front speed (ṡ = −T_x/St), so a *large* St means a slow
front: λ is small, the liquid stays thin, and the transformed diffusion term 1/s² is huge - the system is far
stiffer and the explicit solver takes thirty times more steps. A small St lets the front run, s grows, and
the effective diffusivity on the fixed ξ-grid falls with it. The front ODE itself is never the stiff part;
for slow fronts an implicit integrator on the transformed system is the right tool.
"""),
        ],
        [
            code(r"""Pe, Da = 10.0, 2.0
exact = pr.plug_flow_reactor(Pe=Pe, Da=Da).exact
for name, g in (("uniform, 80 intervals", grids.uniform(0, 1, 80)), ("stretched towards the inlet, 80 intervals", grids.stretched(0, 1, 80, cluster=0.2, strength=0.7))):
    D1, D2 = st.diff_operator(g.x, 1, 2).to_dense(), st.diff_operator(g.x, 2, 2).to_dense()
    A = -D1 + D2 / Pe - Da * np.eye(g.n)
    A[0, :] = -D1[0, :] / Pe; A[0, 0] += 1.0                  # inlet: c - c_x/Pe = 1 (Danckwerts)
    A[-1, :] = D1[-1, :]                                      # outlet: c_x = 0
    b = np.zeros(g.n); b[0] = 1.0
    c = np.linalg.solve(A, b)
    flux_in, flux_out = 1.0, c[-1]                            # c - c_x/Pe at the inlet is imposed; at the outlet c_x = 0 leaves c
    consumed = Da * g.integrate(c)
    print(f"{name:<42}: spacing {np.diff(g.x).min():.4f}..{np.diff(g.x).max():.4f}; max error vs Danckwerts {np.max(np.abs(c - exact(g.x))):.1e}; balance: in - out - consumed = {flux_in - flux_out - consumed:+.1e}")"""),
            md(r"""
The steady reactor's integral balance - what enters minus what leaves equals what reacts - holds to the
discretisation error, which is the check that the boundary rows are consistent with the interior; the
conversion matches Danckwerts' closed form. The stretched grid is *worse* than the uniform one: at Pe = 10 the
inlet jump is 1/Pe = 0.1 wide and a uniform spacing of 0.0125 resolves it already, so clustering points there
only coarsens the rest of the reactor, where the second-order stencils on a non-uniform grid lose accuracy.
Stretching pays when a layer is narrower than the uniform spacing (notebook 14's boundary layer at Pe = 200),
not before.
"""),
        ],
    ]


@solutions(
    "15_project_runaway_in_a_tubular_reactor",
    imports=[
        "from scipy import optimize as sopt",
        "from engmolines import problems as pr, integrators as ti, linalg as la, datasets",
    ],
    setup=r"""data = pd.read_csv(datasets.path("reactor_profiles.csv"), comment="#")
data = data[data.T_K > 0].reset_index(drop=True)
R, T_ref = 8.314, 320.0
x_tc = np.linspace(0, 2, 11)

def profile(T_in, k0, E, n=60):
    pm = pr.tubular_reactor(n=n, T_in=T_in, k0=k0, E=E)
    sm = pm.solve("bdf", 0.5, order=2, n_out=1)
    return np.interp(x_tc, pm.x, sm.y[-1][pm.n // 2 :])

def fit(runs):
    def residuals(theta):
        k0 = np.exp(theta[0] + theta[1] / (R * T_ref))
        res = []
        for T_in, grp in runs.groupby("T_in_K"):
            res.append(np.interp(grp.x_m, x_tc, profile(T_in, k0, theta[1])) - grp.T_K.to_numpy())
        return np.concatenate(res)
    f = sopt.least_squares(residuals, [np.log(5e7 * np.exp(-60e3 / (R * T_ref)) * 1.5), 55e3], diff_step=[1e-3, 1e-3], xtol=1e-8)
    J, r_ = f.jac, f.fun
    cov = (r_ @ r_) / max(len(r_) - 2, 1) * np.linalg.inv(J.T @ J)
    return f.x, np.sqrt(np.diag(cov))

def ignition(k0, E, hw=0.1):
    T_ins = np.arange(300.0, 346.0, 2.5)
    rises = np.array([pr.tubular_reactor(n=60, T_in=T, k0=k0, E=E, hw=hw).solve("bdf", 0.5, order=2, n_out=1).y[-1][61:].max() - T for T in T_ins])
    return T_ins[np.argmax(np.gradient(rises, T_ins))]""",
)
def _():
    return [
        [
            code(r"""for label, runs in (("310 and 320 K only", data[data.T_in_K < 325]), ("all three", data)):
    theta, se = fit(runs)
    k0 = np.exp(theta[0] + theta[1] / (R * T_ref))
    print(f"{label:<18}: k(320) = {np.exp(theta[0]):.3e} (+/- {100 * se[0]:.1f} %), E = {theta[1] / 1e3:.1f} +/- {se[1] / 1e3:.1f} kJ/mol; ignition at {ignition(k0, theta[1]):.1f} K")
rng = np.random.default_rng(1)
syn = profile(335.0, 5e7, 60e3) + rng.normal(0, 0.5, 11)
extra = pd.DataFrame({"T_in_K": 335.0, "thermocouple": np.arange(1, 12), "x_m": x_tc, "T_K": syn})
theta, se = fit(pd.concat([data, extra], ignore_index=True))
k0 = np.exp(theta[0] + theta[1] / (R * T_ref))
print(f"{'with a 335 K run':<18}: k(320) = {np.exp(theta[0]):.3e} (+/- {100 * se[0]:.1f} %), E = {theta[1] / 1e3:.1f} +/- {se[1] / 1e3:.1f} kJ/mol; ignition at {ignition(k0, theta[1]):.1f} K")"""),
            md(r"""
With only the two coolest runs the activation energy is poorly determined (the hot-spot rises are 3 and 8 K,
little above the noise) and the predicted ignition temperature is less certain. A run near the critical
point carries far more information - the hot spot there is exquisitely sensitive to the kinetics - and
tightens both parameters; it is also, of course, the run a plant would least like to perform.
"""),
        ],
        [
            code(r"""theta, se = fit(data)
k0, E = np.exp(theta[0] + theta[1] / (R * T_ref)), theta[1]
hws = np.array([0.05, 0.1, 0.2, 0.3])
T_ign = [ignition(k0, E, hw) for hw in hws]
print(pd.DataFrame({"h_w (1/s)": hws, "ignition T_in (K)": T_ign}).to_string(index=False))
plt.plot(hws, T_ign, "o-"); plt.xlabel("wall heat-transfer coefficient (1/s)"); plt.ylabel("ignition temperature (K)"); plt.title("runaway diagram"); plt.show()"""),
            md(r"""
Better cooling raises the ignition temperature: the runaway diagram is the operating map the fitted kinetics
provide. Its slope - a few kelvin per 0.1 s⁻¹ here - says how much a fouled wall (lower h_w) narrows the safe
window.
"""),
        ],
        [
            code(r"""p = pr.tubular_reactor(n=100, T_in=330.0)
s20 = p.solve("bdf", 0.5, order=2, t_end=20.0, n_out=1)
y = s20.y[-1].copy()
f = lambda v: p.rhs(0.0, v)   # noqa: E731
evals = 0
for k in range(20):
    Jb = la.banded_jacobian(f, y, *p.band); evals += p.band[0] + p.band[1] + 2
    fact, piv = la.lu_banded(Jb, *p.band)
    dy = la.lu_banded_solve(fact, piv, -f(y)); y += dy
    if np.max(np.abs(dy)) < 1e-10 * np.max(np.abs(y)):
        break
s80 = p.solve("bdf", 0.5, order=2, n_out=1)
print(f"Newton on rhs(y) = 0 from the t = 20 s state: {k + 1} iterations, {evals} function evaluations; differs from the t = 80 s transient by {np.max(np.abs(y - s80.y[-1])):.1e} K")
print(f"the transient to 80 s cost {s80.newton_iterations} Newton iterations (each a banded solve) and {s80.steps} steps")"""),
            md(r"""
A direct Newton solve of the steady equations converges in a handful of iterations from a reasonable start
(the transient at 20 s), each iteration one coloured Jacobian and one banded solve - several times cheaper
than integrating to 80 s. The transient is still the right tool when the start is poor (Newton on the steady
reactor equations has multiple solutions near ignition) or when the dynamics themselves matter.
"""),
        ],
    ]


if __name__ == "__main__":
    write_all(only=[a for a in sys.argv[1:] if not a.startswith("--")])
