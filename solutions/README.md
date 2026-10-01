# Worked solutions

One executed notebook per course notebook, with a worked solution to every exercise (48 in all): code that
computes the answer, followed by an explanation of what the result means - and, where the result was not what
one might expect (a stretched grid that makes things worse, a perturbation that cannot move a front, a
Stefan number that works the other way round), why.

The solutions are generated from `build_solutions.py`, which takes the exercise texts from the course builder
so that they never drift apart: `python solutions/build_solutions.py` rebuilds and executes them all; pass
notebook numbers to rebuild only some.

**For instructors:** if you use the exercises for assessed work, keep the solutions in a private repository
and remove the `solutions` glob from `.github/workflows/tests.yml`.
