# Terminology

## Heaney model

The **Heaney model** denotes the one-dimensional kinetic-hysteresis banding
framework in which a silica concentration at a moving deposition front toggles
between high- and low-deposition surface states (historically labelled H/L in
code). In this repository it is integrated by the `slab_kinetic` solver and,
in two-phase form, by the `slab_twophase` hysteresis precipitation mode
(moving-front Heaney clock). Canonical citation: **[Heaney & Davis 1995]** —
placeholder pending confirmation by Bart.

Publication-facing text, run labels, sweep names, and comparison configs use
**Heaney** only (`E1_Heaney_open`, `d2h_heaney_movingfront`, etc.).

## Poster reference validation

Numerical parameter sets and semi-infinite Green's function checks against the
UW agate poster live in `poster_kinetic_analytical.py` and
`run_poster_kinetic_replication.py`. That is a **validation reference**, not a
separate model name.

## Cahn-Hilliard (CH) model

The **Cahn-Hilliard model** is the phase-field / γ-immiscibility demixing
mechanism coupled to silica transport in `slab_twophase`: moganite vs
chalcedony fractions evolve by CH exchange behind a moving front or within a
closed cavity. In figures, CH rows are grouped under **Cahn-Hilliard model**.
Provisional mineral labels (moganite/chalcedony) may appear in reports but not
on publication figures — band colours denote **SiO₂ phase A / phase B /
mixture** only.
