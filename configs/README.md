# Run configurations

Configs are grouped **by experiment / solver package** so the repo can grow with multiple domains without a flat `configs/` dump.

| Directory | Package / experiment |
|-----------|----------------------|
| [`agate_ch/`](agate_ch/) | Agate Cahn–Hilliard falsification (`continuous_patterns.agate_ch`) |
| [`agate_ch/stage_sequence/`](agate_ch/stage_sequence/) | Experiment 2: `run_a_stage1.yaml`, `run_b_stage2.yaml`, **`run_b_long.yaml`** (long horizon); used by `run_sequence`, `run_b_long`, `diagnose_stage_seq` |
| [`agate_ch/gravity/`](agate_ch/gravity/) | Experiment 4: rim Dirichlet gradient `physics.c0_alpha` (`alpha_0_*.yaml`); sweep via `python -m continuous_patterns.agate_ch.run_gravity_sweep` |
| [`agate_stage2/`](agate_stage2/) | Pure Stage II periodic bulk CH (`continuous_patterns.agate_stage2`) — baseline + γ-scan YAMLs |

Add a sibling folder when you introduce a new experiment (e.g. `configs/reaction_diffusion/`) and keep the CLI default paths for that package under the same name.

Optional keys (e.g. **`physics.c0_alpha`** for Experiment 4) must default in code so **old YAML files without those keys** still flatten and run like previous releases.
