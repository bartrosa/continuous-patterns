"""Boundary-condition aware spectral transform operators for 2D fields."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array
from jax.scipy.fft import dct, idct
from jax.typing import ArrayLike

from continuous_patterns.core.spectral import k_vectors


@dataclass(frozen=True)
class SpectralOps(ABC):
    """Spectral transform pair + Laplacian/biharmonic symbols for given BC."""

    n: int
    L: float

    @abstractmethod
    def forward(self, field: ArrayLike) -> Array:
        """Map spatial field to spectral coefficients."""

    @abstractmethod
    def inverse(self, field_hat: ArrayLike) -> Array:
        """Map spectral coefficients to spatial field."""

    @abstractmethod
    def laplacian_symbol(self) -> Array:
        """Return ``k_sq`` such that ``∇²φ = -inverse(k_sq * forward(φ))``."""

    @abstractmethod
    def biharmonic_symbol(self) -> Array:
        """Return ``k_four`` such that ``∇⁴φ = inverse(k_four * forward(φ))``."""

    @abstractmethod
    def aniso_laplacian_symbol(self, kappa_x: float, kappa_y: float) -> Array:
        """Return ``kappa_x*kx_sq + kappa_y*ky_sq`` for anisotropic stiff term."""

    @abstractmethod
    def kx_sq_symbol(self) -> Array:
        """Return x-axis squared wavenumber/eigenvalue grid."""

    @abstractmethod
    def ky_sq_symbol(self) -> Array:
        """Return y-axis squared wavenumber/eigenvalue grid."""

    @abstractmethod
    def kx_wave_symbol(self) -> Array:
        """Return x-axis angular wavenumbers (or eigenvalue roots) grid."""

    @abstractmethod
    def ky_wave_symbol(self) -> Array:
        """Return y-axis angular wavenumbers grid."""


@dataclass(frozen=True)
class PeriodicOps(SpectralOps):
    """FFT2 in both axes, equivalent to legacy periodic pseudospectral operators."""

    _k_sq: Array = field(init=False, repr=False)
    _kx_sq: Array = field(init=False, repr=False)
    _ky_sq: Array = field(init=False, repr=False)
    _kx_wave: Array = field(init=False, repr=False)
    _ky_wave: Array = field(init=False, repr=False)
    _k_four: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        k_sq, kx_sq, ky_sq, kx_wave, ky_wave, k_four = k_vectors(L=self.L, n=self.n)
        object.__setattr__(self, "_k_sq", k_sq)
        object.__setattr__(self, "_kx_sq", kx_sq)
        object.__setattr__(self, "_ky_sq", ky_sq)
        object.__setattr__(self, "_kx_wave", kx_wave)
        object.__setattr__(self, "_ky_wave", ky_wave)
        object.__setattr__(self, "_k_four", k_four)

    def forward(self, field: ArrayLike) -> Array:
        return jnp.fft.fft2(jnp.asarray(field))

    def inverse(self, field_hat: ArrayLike) -> Array:
        return jnp.fft.ifft2(jnp.asarray(field_hat))

    def laplacian_symbol(self) -> Array:
        return self._k_sq

    def biharmonic_symbol(self) -> Array:
        return self._k_four

    def aniso_laplacian_symbol(self, kappa_x: float, kappa_y: float) -> Array:
        return (jnp.asarray(kappa_x) * self.kx_sq_symbol()) + (
            jnp.asarray(kappa_y) * self.ky_sq_symbol()
        )

    def kx_sq_symbol(self) -> Array:
        return self._kx_sq

    def ky_sq_symbol(self) -> Array:
        return self._ky_sq

    def kx_wave_symbol(self) -> Array:
        return self._kx_wave

    def ky_wave_symbol(self) -> Array:
        return self._ky_wave


@dataclass(frozen=True)
class NeumannPeriodicOps(SpectralOps):
    """DCT-II in x (Neumann), FFT in y (periodic) for slab geometry."""

    def _kx_ky_symbols(self) -> tuple[Array, Array, Array, Array]:
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        dx = self.L / self.n
        mode = jnp.arange(self.n)
        kx_1d = (jnp.pi / jnp.asarray(self.L)) * mode
        ky_1d = (2.0 * jnp.pi) * jnp.fft.fftfreq(self.n, d=dx)
        kx_wave, ky_wave = jnp.broadcast_arrays(kx_1d[:, None], ky_1d[None, :])
        kx_sq = kx_wave * kx_wave
        ky_sq = ky_wave * ky_wave
        return kx_sq, ky_sq, kx_wave, ky_wave

    def forward(self, field: ArrayLike) -> Array:
        field_arr = jnp.asarray(field)
        tmp = jnp.fft.fft(field_arr, axis=1)
        return dct(tmp, type=2, axis=0, norm="ortho")

    def inverse(self, field_hat: ArrayLike) -> Array:
        coeff = jnp.asarray(field_hat)
        tmp = idct(coeff, type=2, axis=0, norm="ortho")
        return jnp.fft.ifft(tmp, axis=1)

    def laplacian_symbol(self) -> Array:
        kx_sq, ky_sq, _, _ = self._kx_ky_symbols()
        return kx_sq + ky_sq

    def biharmonic_symbol(self) -> Array:
        k_sq = self.laplacian_symbol()
        return k_sq * k_sq

    def aniso_laplacian_symbol(self, kappa_x: float, kappa_y: float) -> Array:
        return (jnp.asarray(kappa_x) * self.kx_sq_symbol()) + (
            jnp.asarray(kappa_y) * self.ky_sq_symbol()
        )

    def kx_sq_symbol(self) -> Array:
        kx_sq, _, _, _ = self._kx_ky_symbols()
        return kx_sq

    def ky_sq_symbol(self) -> Array:
        _, ky_sq, _, _ = self._kx_ky_symbols()
        return ky_sq

    def kx_wave_symbol(self) -> Array:
        _, _, kx_wave, _ = self._kx_ky_symbols()
        return kx_wave

    def ky_wave_symbol(self) -> Array:
        _, _, _, ky_wave = self._kx_ky_symbols()
        return ky_wave
