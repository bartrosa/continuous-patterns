# Run configurations

Configs are grouped **by experiment / solver package** so the repo can grow with multiple domains without a flat `configs/` dump.

| Directory | Package / experiment |
|-----------|----------------------|
| [`agate_ch/`](agate_ch/) | Agate Cahn–Hilliard falsification (`continuous_patterns.agate_ch`) |
| [`agate_ch/stage_sequence/`](agate_ch/stage_sequence/) | Experiment 2: `run_a_stage1.yaml`, `run_b_stage2.yaml`, **`run_b_long.yaml`** (long horizon); used by `run_sequence`, `run_b_long`, `diagnose_stage_seq` |
| [`agate_stage2/`](agate_stage2/) | Pure Stage II periodic bulk CH (`continuous_patterns.agate_stage2`) — baseline + γ-scan YAMLs |

Add a sibling folder when you introduce a new experiment (e.g. `configs/reaction_diffusion/`) and keep the CLI default paths for that package under the same name.
