"""Write the course data file into src/engmolines/data: steady temperature profiles of a cooled tubular reactor,
generated from the model of problems.tubular_reactor with stated parameters, noise and a dead thermocouple
(seed 2026). Run: python tools/make_course_data.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from engmolines import problems as pr  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "src" / "engmolines" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)

x_tc = np.linspace(0.0, 2.0, 11)
rows = []
for T_in in (310.0, 320.0, 330.0):
    p = pr.tubular_reactor(T_in=T_in)
    s = p.solve("bdf", 0.5, order=2, n_out=1)
    T = np.interp(x_tc, p.x, s.y[-1][p.n // 2 :])
    T = np.round(T + rng.normal(0, 0.5, T.size), 2)
    for i, (xi, Ti) in enumerate(zip(x_tc, T)):
        rows.append([T_in, i + 1, round(xi, 2), -999.0 if (T_in == 320.0 and i == 7) else Ti])
header = [
    "Steady wall-cooled tubular reactor, first-order exothermic reaction A -> B: temperatures (K) at 11 thermocouples along the",
    "2 m reactor for three inlet temperatures. Generated from the plug-flow/axial-dispersion model of engmolines.problems.tubular_reactor",
    "with k0 = 5e7 1/s, E = 60 kJ/mol, adiabatic rise 120 K, wall coefficient 0.1 1/s, wall at the inlet temperature, and 0.5 K noise.",
    "Fault: thermocouple 8 of the 320 K run is disconnected (-999). Columns: T_in_K, thermocouple, x_m, T_K.",
]
lines = [f"# {h}" for h in header] + ["T_in_K,thermocouple,x_m,T_K"] + [",".join(str(v) for v in r) for r in rows]
(OUT / "reactor_profiles.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote reactor_profiles.csv", len(rows), "rows")
