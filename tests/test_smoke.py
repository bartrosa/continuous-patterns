"""Smoke tests for environment and imports."""

from __future__ import annotations


def test_package_version() -> None:
    import continuum_patterns as cp

    assert cp.__version__


def test_jax_import() -> None:
    import jax

    assert jax.__version__
