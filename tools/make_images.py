"""README images: the hero animation and the gallery, computed with engmolines. Run: python tools/make_images.py"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from engmolines import boundary as bc  # noqa: E402
from engmolines import grids, stiffness  # noqa: E402
from engmolines import integrators as ti
from engmolines import operators as op
from engmolines import problems as pr

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def hero():
    g = grids.cell_centred(0, 1, 160)
    u0 = np.exp(-(((g.x - 0.3) / 0.05) ** 2))
    frames = 60
    runs = {}
    for sch in ("upwind", "centred", "van-leer"):
        f = lambda t, u, sch=sch: op.advection(u, g, 1.0, bc.Periodic(), bc.Periodic(), t, sch)
        runs[sch] = ti.dopri45(f, u0, (0, 1.0), 1e-7, 1e-9, n_out=frames).y
    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=80)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 1.2)
    ax.set_xlabel("x")
    ax.set_title("u_t + u_x = 0 by the method of lines: three spatial schemes, one adaptive time integrator")
    (ex_line,) = ax.plot(g.x, u0, "k--", lw=1, label="exact")
    lines = {sch: ax.plot(g.x, u0, label=sch)[0] for sch in runs}
    ax.legend(loc="upper right", fontsize=8)

    def update(k):
        t = k / frames
        ex_line.set_ydata(np.exp(-(((((g.x - 0.3 - t) + 0.5) % 1.0 - 0.5) / 0.05) ** 2)))
        for sch, ln in lines.items():
            ln.set_ydata(runs[sch][k])
        return [ex_line, *lines.values()]

    fig.tight_layout()
    FuncAnimation(fig, update, frames=frames + 1, blit=True).save(OUT / "hero.gif", writer=PillowWriter(fps=12))
    plt.close(fig)


def spectrum_and_limits():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    g = grids.uniform(0, 1, 40)
    f = lambda t, u: op.diffusion(u, g, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t)  # noqa: E731
    lam = stiffness.spectrum(f, np.zeros(41), band=(1, 1))
    lam = lam[np.abs(lam) > 1e-9]
    pa = pr.advection_periodic(scheme="centred")
    lam_a = stiffness.spectrum(pa.rhs, pa.y0)
    axes[0].plot(lam.real, lam.imag, "o", ms=3, label="heat (real, stiff)")
    axes[0].plot(lam_a.real, lam_a.imag, "s", ms=3, label="centred advection (imaginary)")
    axes[0].axvline(0, color="k", lw=0.5)
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Re λ")
    axes[0].set_ylabel("Im λ")
    axes[0].set_title("spectra of semi-discrete operators")
    ns = np.array([10, 20, 40, 80, 160])
    lim = []
    for n in ns:
        gn = grids.uniform(0, 1, n)
        ln = stiffness.spectrum(
            lambda t, u, gn=gn: op.diffusion(u, gn, 1.0, bc.Dirichlet(0.0), bc.Dirichlet(0.0), t),
            np.zeros(n + 1),
            band=(1, 1),
        )
        lim.append(stiffness.explicit_step_limit(ln[np.abs(ln) > 1e-9], "rk4"))
    axes[1].loglog(ns, lim, "o-", label="RK4 stability limit")
    axes[1].loglog(ns, 2.785 / (4 * ns**2), "k--", lw=1, label="2.785 Δx²/4")
    axes[1].set_xlabel("grid intervals n")
    axes[1].set_ylabel("largest stable step")
    axes[1].legend(fontsize=8)
    axes[1].set_title("explicit steps shrink like Δx²")
    fig.tight_layout()
    fig.savefig(OUT / "stiffness.png")
    plt.close(fig)


def kdv_collision():
    p = pr.kdv_two_solitons()
    s = p.solve("dopri", rtol=1e-8, atol=1e-10, n_out=160)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.imshow(s.y, aspect="auto", origin="lower", extent=(-20, 20, 0, 8), cmap="magma")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("KdV: two solitons collide (spectral + Dormand–Prince)")
    fig.tight_layout()
    fig.savefig(OUT / "kdv_collision.png")
    plt.close(fig)


def dam_break():
    p = pr.shallow_water_dam_break()
    s = p.solve("dopri", rtol=1e-7, atol=1e-9, n_out=4)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for t, y in zip(s.t, s.y):
        ax.plot(p.x, y[:400], label=f"t = {t:.2f}")
    ax.plot(p.x, p.exact(p.x, s.t[-1])[0], "k:", lw=1, label="Stoker (exact)")
    ax.legend(fontsize=7)
    ax.set_xlabel("x")
    ax.set_ylabel("depth")
    ax.set_title("shallow-water dam break, MUSCL–Rusanov")
    fig.tight_layout()
    fig.savefig(OUT / "dam_break.png")
    plt.close(fig)


def gray_scott():
    p = pr.gray_scott(n=96)
    s = p.solve("rk4", 1.0, t_end=4000.0, n_out=1)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(s.y[-1][96 * 96 :].reshape(96, 96).T, origin="lower", cmap="magma")
    ax.axis("off")
    ax.set_title("Gray–Scott pattern (2-D reaction–diffusion)")
    fig.tight_layout()
    fig.savefig(OUT / "gray_scott.png")
    plt.close(fig)


def cavity():
    p = pr.cavity_flow(n=32)
    s = p.solve("rk4", 0.004, n_out=1)
    n = 33
    psi = p.params["streamfunction"](s.y[-1].reshape(n, n))
    u, v = p.params["velocity"](psi)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    axes[0].contour(p.x, p.x, psi.T, 24, cmap="RdBu")
    axes[0].set_aspect("equal")
    axes[0].set_title("lid-driven cavity, Re = 100: streamlines")
    ghia_y = [
        0.0,
        0.0547,
        0.0625,
        0.0703,
        0.1016,
        0.1719,
        0.2813,
        0.4531,
        0.5,
        0.6172,
        0.7344,
        0.8516,
        0.9531,
        0.9609,
        0.9688,
        0.9766,
        1.0,
    ]
    ghia_u = [
        0.0,
        -0.03717,
        -0.04192,
        -0.04775,
        -0.06434,
        -0.10150,
        -0.15662,
        -0.21090,
        -0.20581,
        -0.13641,
        0.00332,
        0.23151,
        0.68717,
        0.73722,
        0.78871,
        0.84123,
        1.0,
    ]
    axes[1].plot(u[16, :], p.x, label="engmolines, 32 × 32")
    axes[1].plot(ghia_u, ghia_y, "ko", ms=4, label="Ghia et al. (1982)")
    axes[1].set_xlabel("u on the vertical centreline")
    axes[1].set_ylabel("y")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cavity.png")
    plt.close(fig)


def runaway():
    T_ins = np.arange(300.0, 346.0, 2.5)
    rises = [pr.tubular_reactor(n=60, T_in=T).solve("bdf", 0.5, order=2, n_out=1).y[-1][61:].max() - T for T in T_ins]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(T_ins, rises, "o-")
    ax.set_xlabel("inlet / wall temperature (K)")
    ax.set_ylabel("hot-spot rise (K)")
    ax.set_title("tubular reactor: ignition by parametric sensitivity")
    ax.axvspan(332.5, 337.5, color="C3", alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUT / "runaway.png")
    plt.close(fig)


if __name__ == "__main__":
    hero()
    spectrum_and_limits()
    kdv_collision()
    dam_break()
    gray_scott()
    cavity()
    runaway()
    for p_ in sorted(OUT.iterdir()):
        print(f"{p_.name:<22} {p_.stat().st_size / 1024:7.1f} kB")
