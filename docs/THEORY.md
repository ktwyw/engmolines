# Theory notes

A compact map of the ideas behind each module, what the validation suite checks, and where to read more.

## The method of lines

Discretise the spatial derivatives of u_t = L(u) on a grid; the unknowns u_i(t) then satisfy a system of ODEs
u′ = F(u), the *semi-discrete* system, to which any time integrator applies. The total error has two parts
that can be measured separately: the spatial error (the semi-discrete solution against the PDE) and the
temporal error (the integrator against the exact semi-discrete solution, computable for linear problems as
exp(At) u₀). Because the fastest grid modes have rates of order Δx⁻² (diffusion), Δx⁻¹ (advection) and
Δx⁻⁴ (fourth-order operators), the system is stiff, and the choice of integrator is the central decision.
*Schiesser, The Numerical Method of Lines (1991); Hundsdorfer & Verwer, Numerical Solution of Time-Dependent
Advection-Diffusion-Reaction Equations (2003); LeVeque, Finite Difference Methods for ODEs and PDEs (2007).*

## Stencils and boundaries (`stencils`, `boundary`, `operators`)

Fornberg's recurrence gives the weights of the formula Σ wⱼ f(xⱼ) ≈ f^(m)(x₀) exact for polynomials of the
highest degree on any points; centred stencils inside and one-sided stencils of the same accuracy at the ends
give an operator of uniform order (for the operator itself the closures' order is the global order; for
parabolic problems closures one order lower suffice). Boundary conditions are imposed by ghost values chosen
so that the condition holds to second order - odd reflection for Dirichlet, even reflection plus the slope for
Neumann, the small linear equation for Robin - and the interior stencil is applied everywhere. Conservative
forms (fluxes through faces, harmonic averaging of diffusivities, control volumes at r = 0) keep integrals
exact. *Fornberg, SIAM Review 40 (1998); Patankar, Numerical Heat Transfer and Fluid Flow (1980).*

## Stiffness and stability (`stiffness`, `analysis`)

On u′ = λu an integrator multiplies the solution by R(hλ); stability needs |R| ≤ 1 for every eigenvalue of
the Jacobian. Explicit methods have bounded regions (RK4 reaches −2.785 on the real axis and ±2.83i; Euler
−2 and nothing on the imaginary axis), so the stiffest mode sets the step: h < 2.785Δx²/(4κ) for RK4 on
diffusion, a Courant number ≤ 1 for upwind with Euler. A-stable methods (trapezoidal, BDF2, Gauss, Radau)
have no such limit; L-stable ones (backward Euler, BDF, Radau IIA) also damp the fastest modes in one step,
which the trapezoidal rule does not (R → −1: ringing on rough data, cured by a Rannacher start). Von Neumann
analysis composes a stencil's Fourier symbol σ(θ) with R to give the amplification of each mode of the fully
discrete scheme; the modified equation expands the scheme's truncation error into effective diffusion and
dispersion terms (upwind: (aΔx/2) u_xx). *Hairer & Wanner, Solving ODEs II (1996); Strikwerda, Finite
Difference Schemes and PDEs (2004); Warming & Hyett, J. Comput. Phys. 14 (1974).*

## Implicit steps and their cost (`linalg`, `integrators`)

An implicit step solves y − hβ f(t, y) = r by Newton's method. The Jacobian of a method-of-lines system has the
stencil's band, so it can be computed by colouring the columns (b + 1 evaluations, Curtis-Powell-Reid 1974)
and factorised by banded LU in O(n b²); the factorisation is reused over many steps (modified Newton) and
refreshed when the iteration's convergence rate exceeds a threshold. In 2-D the band is the grid width
(Kronecker structure), still far cheaper than a dense solve. IMEX methods treat a stiff linear part implicitly
and a nonlinear part explicitly; the exponential integrator solves linear systems exactly; Störmer-Verlet
preserves a modified energy for second-order conservative systems; fully implicit BDF solves index-1 DAEs
whose algebraic rows are boundary conditions or constraints. *Ascher & Petzold, Computer Methods for ODEs and
DAEs (1998); Ascher, Ruuth & Wetton, SIAM J. Numer. Anal. 32 (1995).*

## Hyperbolic problems and finite volumes

Centred stencils for advection have purely imaginary spectra and no dissipation; upwind stencils are
dissipative and first order. Godunov's theorem forbids linear schemes that are both second order and
monotone; flux limiters (minmod, superbee, van Leer, MC) are nonlinear schemes that are TVD and second order
away from extrema. For nonlinear conservation laws the finite-volume form - cell averages updated by face
fluxes from an approximate Riemann solver (Rusanov's local Lax-Friedrichs, or the exact Godunov flux) with
MUSCL reconstruction - conserves exactly, places shocks at the Rankine-Hugoniot speed and selects
entropy-satisfying rarefactions. *LeVeque, Finite Volume Methods for Hyperbolic Problems (2002); Toro, Riemann
Solvers and Numerical Methods for Fluid Dynamics (2009).*

## Spectral methods and dispersive equations (`spectral`)

On a periodic grid the derivative of a smooth function is computed exactly in Fourier space by multiplying
by (ik)ᵐ, with errors that decay exponentially with the number of points; products of resolved modes can
alias into the kept range unless the upper third of the spectrum is removed (the 2/3 rule). The spectral
third derivative has eigenvalues up to (πn/L)³ on the imaginary axis, so dispersive equations are stiff in
proportion to n³; integrating factors or exponential integrators remove the linear part's stiffness.
*Trefethen, Spectral Methods in MATLAB (2000); Boyd, Chebyshev and Fourier Spectral Methods (2001).*

## The examples

Each problem in `problems` carries its exact solution or benchmark: separation of variables for heat
conduction (Bessel modes in cylinders, sin(πr)/(πr) in spheres, μ tan(μ/2) = Bi for Robin ends), the
cosh/sinh fin, the Black-Scholes formula, Barenblatt's similarity solution of the porous-medium equation,
the KPP front speed 2√(rD) and Bramson's correction, the Allen-Cahn kink, Neumann's Stefan solution, Burgers'
characteristics and Riemann solutions, Stoker's dam break, standing waves and beam modes, KdV and NLS
solitons, Danckwerts' reactor profile, Thiele's effectiveness factor, the NTU effectiveness of a counterflow
exchanger, and the Ghia-Ghia-Shin cavity benchmark. *Carslaw & Jaeger, Conduction of Heat in Solids (1959);
Whitham, Linear and Nonlinear Waves (1974); Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982); Froment, Bischoff
& De Wilde, Chemical Reactor Analysis and Design (2011).*
