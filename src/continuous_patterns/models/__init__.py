"""Model drivers (Stage I agate CH, Stage II bulk, …).

Public API: import submodules explicitly, e.g. ``agate_ch.simulate(cfg)`` or
``agate_stage2.simulate(cfg)``. ``experiments.run`` dispatches on
``cfg['experiment']['model']``.
"""

from . import agate_ch, agate_stage2

__all__ = ["agate_ch", "agate_stage2"]
