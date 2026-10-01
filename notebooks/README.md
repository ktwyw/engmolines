# Course notebooks

Sixteen executed notebooks on the numerical method of lines, built on one working method: discretise space,
read the spectrum of what you get, choose the time integrator accordingly, separate the spatial and temporal
errors, and check conservation, invariants and boundary values. Every notebook states its learning
objectives, prerequisites and study time, rebuilds a key algorithm in plain Python ("Inside the algorithm",
checked against the library), and ends with exercises including an "Implement it yourself" task. Worked
solutions are in [`../solutions`](../solutions).

| # | Notebook | Time | Before you start | Solutions |
|---|---|---|---|---|
| [00](00_what_is_the_method_of_lines.ipynb) | What is the method of lines? | 45 min | an introductory ODE course and basic NumPy | [solutions](../solutions/00_solutions.ipynb) |
| [01](01_stencils_and_boundary_conditions.ipynb) | Stencils and boundary conditions | 60 min | notebook 00 | [solutions](../solutions/01_solutions.ipynb) |
| [02](02_stiffness_and_choosing_the_integrator.ipynb) | Stiffness and choosing the integrator | 75 min | notebook 00; the ODE notebooks of engnumeric help | [solutions](../solutions/02_solutions.ipynb) |
| [03](03_banded_newton_and_the_cost_of_implicit_steps.ipynb) | Banded Newton and the cost of implicit steps | 60 min | notebooks 00 and 02 | [solutions](../solutions/03_solutions.ipynb) |
| [04](04_the_heat_equation_in_every_form.ipynb) | The heat equation in every form | 60 min | notebooks 01 and 02 | [solutions](../solutions/04_solutions.ipynb) |
| [05](05_nonlinear_diffusion.ipynb) | Nonlinear diffusion | 75 min | notebooks 03 and 04 | [solutions](../solutions/05_solutions.ipynb) |
| [06](06_reaction_diffusion_fronts_and_patterns.ipynb) | Reaction-diffusion: fronts and patterns | 60 min | notebooks 02 and 05 | [solutions](../solutions/06_solutions.ipynb) |
| [07](07_advection_the_hard_case.ipynb) | Advection: the hard case | 75 min | notebook 02 | [solutions](../solutions/07_solutions.ipynb) |
| [08](08_shocks_and_conservation_laws.ipynb) | Shocks and conservation laws | 75 min | notebook 07 | [solutions](../solutions/08_solutions.ipynb) |
| [09](09_waves_strings_and_beams.ipynb) | Waves, strings and beams | 60 min | notebooks 02 and 09 of engnumeric help | [solutions](../solutions/09_solutions.ipynb) |
| [10](10_dispersive_waves_and_spectral_methods.ipynb) | Dispersive waves and spectral methods | 60 min | notebook 09 of engnumeric (the FFT) helps | [solutions](../solutions/10_solutions.ipynb) |
| [11](11_coupled_systems_and_daes.ipynb) | Coupled systems and DAEs | 75 min | notebooks 03 and 07 | [solutions](../solutions/11_solutions.ipynb) |
| [12](12_two_dimensions.ipynb) | Two dimensions | 90 min | notebooks 03 and 06 | [solutions](../solutions/12_solutions.ipynb) |
| [13](13_diagnosing_failures.ipynb) | Diagnosing failures | 75 min | notebooks 02, 07 and 10 | [solutions](../solutions/13_solutions.ipynb) |
| [14](14_moving_boundaries_and_transformed_domains.ipynb) | Moving boundaries and transformed domains | 75 min | notebooks 01 and 05 | [solutions](../solutions/14_solutions.ipynb) |
| [15](15_project_runaway_in_a_tubular_reactor.ipynb) | Project: runaway in a tubular reactor | 3 h | notebooks 02, 03, 07 and 11 | [solutions](../solutions/15_solutions.ipynb) |

**Suggested order:** foundations (00-03), parabolic problems (04-06), hyperbolic and dispersive problems
(07-10), coupled systems and two dimensions (11-12), diagnosis and transformations (13-14), and the project
(15), which draws on all of them. The companion course [engnumeric](https://github.com/ktwyw/engnumeric)
develops the time integrators and finite differences from first principles.

The notebooks are generated from `build_notebooks.py` (`python notebooks/build_notebooks.py` rebuilds and executes
them; pass notebook numbers to rebuild only some). They open in Google Colab, where the first cell installs
`engmolines` from GitHub.
