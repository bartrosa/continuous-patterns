"""Model composition layer (cavity + rim / bulk relaxation).

Import submodules explicitly, e.g. ``from continuous_patterns.models import cavity_reactive`` then
``cavity_reactive.simulate(cfg)``. The CLI in ``continuous_patterns.experiments.run`` dispatches on
``cfg["experiment"]["model"]``.
"""

from . import bulk_relaxation, cavity_reactive

__all__ = ["bulk_relaxation", "cavity_reactive"]
