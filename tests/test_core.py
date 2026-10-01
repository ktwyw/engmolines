import numpy as np
import pytest

from engmolines import (
    analysis,
    grids,
    stiffness,
)
from engmolines import (
    boundary as bc,
)
from engmolines import (
    integrators as ti,
)
from engmolines import (
    linalg as la,
)
from engmolines import (
    operators as op,
)
from engmolines import (
    stencils as st,
)


def test_grids():
    g = grids.cell_centred(0, 2, 4)
    assert g.dx == 0.5 and g.faces.tolist() == [0, 0.5, 1, 1.5, 2] and g.integrate(np.ones(4)) == 2.0
    s = grids.stretched(0, 1, 10, cluster=0.3, strength=1.5)
    assert s.x[0] == 0 and s.x[-1] == 1 and np.all(np.diff(s.x) > 0) and not s.uniform
    assert grids.stretched(0, 1, 4, strength=0).uniform
    g2 = grids.tensor(grids.uniform(0, 1, 2), grids.uniform(0, 2, 4))
    assert g2.shape == (3, 5) and g2.X.shape == (3, 5) and g2.integrate(np.ones((3, 5))) == pytest.approx(2.0)
    assert repr(ti.explicit_rk(lambda t, y: -y, [1.0], (0, 1), 0.1)).startswith("Solution(")


def test_stencils_and_linalg():
    D = st.diff_operator(np.linspace(0, 1, 6), 1, 2)
    assert np.allclose((2.0 * D).to_dense(), 2 * D.to_dense())
    with pytest.raises(ValueError):
        st.diff_operator(np.linspace(0, 1, 6), 1, 2, periodic=True).to_banded()
    with pytest.raises(ZeroDivisionError):
        la.lu_banded(np.zeros((3, 4)), 1, 1)
    A = np.diag([1.0, 2.0, 3.0])
    assert np.allclose(la.solve_banded(la.to_banded(A, 0, 0), 0, 0, [1, 2, 3]), 1.0)


def test_boundary_errors():
    with pytest.raises(TypeError):
        bc.pad(np.ones(3), 0.1, "bad", bc.Neumann(0.0))
    with pytest.raises(ValueError):
        bc.beam_ghosts(np.ones(4), 0.1, "welded", "left")
    assert bc.dirichlet_rhs(np.ones(3), bc.Dirichlet(1.0), bc.Dirichlet(2.0), kind="cell").tolist() == [1, 1, 1]


def test_operators_variants():
    g = grids.uniform(0, 1, 10)
    u = g.x**2
    r = op.diffusion(
        u, g, lambda x, u: 1 + 0 * x, bc.Neumann(0.0), bc.Robin(1.0, 1.0, 3.0), source=lambda x, u, t: 0 * x
    )
    assert r.shape == u.shape
    a = op.advection(u, g, -1.0, bc.Dirichlet(0.0), bc.Dirichlet(1.0), scheme="minmod")
    assert a.shape == u.shape
    with pytest.raises(ValueError):
        op.biharmonic(u, g, "clamped", "welded")
    assert stiffness.operator_extremes("advection_upwind", 0.1, 2.0) == -40.0
    with pytest.raises(ValueError):
        stiffness.operator_extremes("nothing", 0.1)
    with pytest.raises(ValueError):
        stiffness.stability_function("rk9", 0.0)
    with pytest.raises(ValueError):
        analysis.symbol("nothing", 0.0, 0.1)
    with pytest.raises(ValueError):
        analysis.modified_equation("nothing", 1.0, 0.1)
    assert analysis.modified_equation("upwind2", 1.0, 0.1)["dispersion"] == pytest.approx(0.01 / 3)


def test_integrator_edge_cases():
    f = lambda t, y: -y  # noqa: E731
    s = ti.dopri45(f, [1.0], (0, 1), n_out=0)
    assert s.t[-1] == 1.0 and abs(s.y[-1, 0] - np.exp(-1)) < 1e-5
    s = ti.dopri45(f, [1.0], (0, 1), max_steps=2, h0=0.1)
    assert s.message != "ok"
    s = ti.bdf(f, [1.0], (0, 1), 0.1, 3, jac=lambda t, y: np.array([[-1.0]]))
    assert abs(s.y[-1, 0] - np.exp(-1)) < 1e-3
    bad = ti.bdf(lambda t, y: np.array([np.sqrt(y[0])]), [1.0], (0, 3), 0.5, 2, newton_tol=1e-30)
    assert "failed" in bad.message or bad.steps == 6
    s = ti.trapezoidal(f, [1.0], (0, 1), 0.1, jac=lambda t, y: np.array([[-1.0]]))
    assert abs(s.y[-1, 0] - np.exp(-1)) < 1e-3
    s = ti.imex(lambda t, y: 0 * y, f, [1.0], (0, 1), 0.1, order=1)
    assert abs(s.y[-1, 0] - np.exp(-1)) < 5e-2
    cost = stiffness.cost_model(10, np.array([-100.0, -1.0]), 1.0, 1)
    assert cost["ratio"] > 0
    study = analysis.convergence_study(
        lambda n, h: (np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1) ** 2 + h + 1 / n**2),
        [10, 20],
        [0.1, 0.05],
        lambda x: x**2,
    )
    assert study["errors"].shape == (2, 2)
    with pytest.raises(ValueError):
        analysis.symbol("nothing", 0.0, 0.1)


def test_problems_api():
    from engmolines import problems as pr

    p = pr.heat_dirichlet(n=10)
    assert p.n == 11 and p.x[-1] == 1.0
    for m in ("bdf", "trapezoidal", "rk4", "dopri"):
        s = p.solve(m, 1e-3 if m != "rk4" else 2e-4, t_end=0.01)
        assert p.error(s) < 1e-2
    with pytest.raises(ValueError):
        p.solve("magic")
    bad = pr.Problem(
        "no implicit solution", p.grid, lambda t, y: y**3, np.ones(11), (1, 1)
    )  # y - h y^3 = 1 has no real root for h = 1
    with pytest.raises(RuntimeError):
        bad.solve("bdf", 1.0, order=1, t_end=3.0)
    assert bad.solve("bdf", 1.0, order=1, t_end=3.0, allow_failure=True).message != "ok"
    assert (
        len(pr.PARABOLIC) == 16
        and len(pr.HYPERBOLIC) == 9
        and len(pr.DISPERSIVE) == 3
        and len(pr.COUPLED) == 5
        and len(pr.TWO_D) == 3
        and len(pr.PROJECT) == 1
        and len(pr.ALL) == 37
    )
    assert pr.front_position(np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0])) == 0.0


def test_spectral_and_conservation_law():
    from engmolines import spectral

    x = spectral.grid(0, 2 * np.pi, 16)
    u = np.sin(x)
    assert np.allclose(spectral.derivative(u, 2 * np.pi, 3), -np.cos(x))
    assert np.allclose(spectral.dealias(u), u)
    c = np.exp(1j * x)
    assert np.iscomplexobj(spectral.derivative(c, 2 * np.pi, 1))
    assert spectral.max_eigenvalue(16, 2 * np.pi, 2) == 64.0
    g = grids.cell_centred(0, 1, 20)
    w = np.vstack([np.ones(20), np.zeros(20)])
    r = op.conservation_law(w, g, lambda W: W, lambda W: np.ones(W.shape[1]), "periodic", "periodic")
    assert r.shape == (2, 20) and np.allclose(r, 0)
    per = grids.periodic(0, 1, 4)
    assert per.widths.tolist() == [0.25] * 4
