"""Single IMEX time step for Model C CH + reaction + optional stress.

One ``imex_step(state, geom, prm, dt)`` shared by Stage I and Stage II; branches
on ``SimParams.reaction_active`` / ``dirichlet_active`` (``docs/ARCHITECTURE.md``
§3.4, §4.2).
"""
