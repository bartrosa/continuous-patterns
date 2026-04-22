# Run configurations

Configs are grouped **by experiment / solver package** so the repo can grow with multiple domains without a flat `configs/` dump.

| Directory | Package / experiment |
|-----------|----------------------|
| [`agate_ch/`](agate_ch/) | Agate Cahn–Hilliard falsification (`continuous_patterns.agate_ch`) |

Add a sibling folder when you introduce a new experiment (e.g. `configs/reaction_diffusion/`) and keep the CLI default paths for that package under the same name.
