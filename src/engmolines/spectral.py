"""Fourier spectral differentiation for periodic problems: derivatives of any order by multiplying the Fourier
coefficients by (ik)^m, the 2/3-rule dealiasing for quadratic nonlinearities, and the wavenumber grid. On a
periodic grid with n points and length L the largest wavenumber is pi n / L, so a spectral u_xxx has
eigenvalues up to (pi n / L)^3 on the imaginary axis - the stiffness of dispersive problems.

>>> import numpy as np
>>> from engmolines import spectral
>>> x = spectral.grid(0, 2 * np.pi, 32)
>>> bool(np.max(np.abs(spectral.derivative(np.sin(x), 2 * np.pi, 1) - np.cos(x))) < 1e-12)
True
"""

from __future__ import annotations

import numpy as np


def grid(a: float, b: float, n: int) -> np.ndarray:
    """n equispaced points on [a, b) (the right end is the left end again)."""
    return a + (b - a) * np.arange(n) / n


def wavenumbers(n: int, L: float) -> np.ndarray:
    """Wavenumbers k = 2 pi j / L in FFT order for n points on a period L."""
    return 2 * np.pi * np.fft.fftfreq(n, L / n)


def derivative(u, L: float, order: int = 1) -> np.ndarray:
    """m-th derivative of a real periodic function sampled on n points of a period L."""
    u = np.asarray(u)
    n = len(u)
    k = wavenumbers(n, L)
    mult = (1j * k) ** order
    if n % 2 == 0 and order % 2 == 1:
        mult[n // 2] = 0.0  # the Nyquist mode has no well-defined odd derivative
    out = np.fft.ifft(mult * np.fft.fft(u))
    return out.real if np.isrealobj(u) else out


def dealias(u, L: float | None = None) -> np.ndarray:
    """Zero the upper third of the spectrum (2/3 rule): products of two dealiased fields then alias nothing back
    into the kept modes."""
    u = np.asarray(u)
    n = len(u)
    U = np.fft.fft(u)
    k = np.fft.fftfreq(n) * n
    U[np.abs(k) > n / 3] = 0.0
    out = np.fft.ifft(U)
    return out.real if np.isrealobj(u) else out


def max_eigenvalue(n: int, L: float, order: int) -> float:
    """|(i k_max)^order|: the extreme eigenvalue of the spectral derivative operator."""
    return (np.pi * n / L) ** order
