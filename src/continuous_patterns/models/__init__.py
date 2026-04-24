"""Model composition layer (Stage I cavity CH, Stage II bulk relaxation).

Import submodules explicitly, e.g. ``from continuous_patterns.models import agate_ch`` then
``agate_ch.simulate(cfg)``. The CLI in ``continuous_patterns.experiments.run`` dispatches on
``cfg["experiment"]["model"]``.
"""

from . import agate_ch, agate_stage2

__all__ = ["agate_ch", "agate_stage2"]
