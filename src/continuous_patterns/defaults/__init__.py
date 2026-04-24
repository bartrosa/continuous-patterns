"""Bundled default configurations for solver-global settings.

Access the defaults via :mod:`importlib.resources` (see ``solver_defaults.yaml``):

    from importlib import resources

    text = (
        resources.files("continuous_patterns.defaults")
        .joinpath("solver_defaults.yaml")
        .read_text(encoding="utf-8")
    )
"""
