"""Diagnostic: bitwise comparison of PeriodicOps vs legacy fft2/k_vectors.

Run with:
    uv run python scripts/check_periodic_ops_bitwise.py

Should print zero or near-machine-epsilon differences.
"""

import jax
import jax.numpy as jnp

from continuous_patterns.core.spectral import k_vectors
from continuous_patterns.core.spectral_ops import PeriodicOps


def main():
    n, L = 64, 10.0
    seed = 42

    # 1. Forward transform comparison
    field = jax.random.normal(jax.random.PRNGKey(seed), (n, n), dtype=jnp.float32)
    legacy_hat = jnp.fft.fft2(field)
    ops = PeriodicOps(n=n, L=L)
    ops_hat = ops.forward(field)
    max_abs_diff_fwd = jnp.max(jnp.abs(legacy_hat - ops_hat))
    print(f"forward: max_abs_diff = {float(max_abs_diff_fwd):.3e}")

    # 2. Inverse transform comparison
    legacy_back = jnp.real(jnp.fft.ifft2(legacy_hat))
    ops_back = jnp.real(ops.inverse(ops_hat))
    max_abs_diff_inv = jnp.max(jnp.abs(legacy_back - ops_back))
    print(f"inverse: max_abs_diff = {float(max_abs_diff_inv):.3e}")

    # 3. Symbol comparison
    k_sq_legacy, kx_sq_legacy, ky_sq_legacy, kx_wave_legacy, ky_wave_legacy, k_four_legacy = (
        k_vectors(L=L, n=n)
    )
    k_sq_ops = ops.laplacian_symbol()
    k_four_ops = ops.biharmonic_symbol()

    print(f"k_sq:    max_abs_diff = {float(jnp.max(jnp.abs(k_sq_legacy - k_sq_ops))):.3e}")
    print(f"k_four:  max_abs_diff = {float(jnp.max(jnp.abs(k_four_legacy - k_four_ops))):.3e}")

    # 4. Component symbols (kx_sq, ky_sq, kx_wave, ky_wave)
    # PeriodicOps must expose these — check API:
    if hasattr(ops, "kx_sq"):
        print(f"kx_sq:   max_abs_diff = {float(jnp.max(jnp.abs(kx_sq_legacy - ops.kx_sq))):.3e}")
        print(f"ky_sq:   max_abs_diff = {float(jnp.max(jnp.abs(ky_sq_legacy - ops.ky_sq))):.3e}")
        print(
            f"kx_wave: max_abs_diff = {float(jnp.max(jnp.abs(kx_wave_legacy - ops.kx_wave))):.3e}"
        )
        print(
            f"ky_wave: max_abs_diff = {float(jnp.max(jnp.abs(ky_wave_legacy - ops.ky_wave))):.3e}"
        )
    else:
        print("WARNING: PeriodicOps does not expose kx_sq/ky_sq/kx_wave/ky_wave directly")

    # 5. Anisotropic Laplacian symbol
    aniso_legacy = 1.0 * kx_sq_legacy + 2.0 * ky_sq_legacy
    aniso_ops = ops.aniso_laplacian_symbol(kappa_x=1.0, kappa_y=2.0)
    print(f"aniso(1,2): max_abs_diff = {float(jnp.max(jnp.abs(aniso_legacy - aniso_ops))):.3e}")

    # 6. Round-trip
    roundtrip = jnp.real(ops.inverse(ops.forward(field)))
    max_abs_diff_rt = jnp.max(jnp.abs(field - roundtrip))
    print(f"round-trip: max_abs_diff = {float(max_abs_diff_rt):.3e}")


if __name__ == "__main__":
    main()
