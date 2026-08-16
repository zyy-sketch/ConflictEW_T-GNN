"""ConflictEW T-GNN: reproducible research pipeline (config-driven).

Turns the browser dashboard prototype into an end-to-end, re-runnable pipeline:
    data -> Kalman feature extraction -> XGBoost (blocked temporal CV)
         -> DeLong / bootstrap CIs -> publication vector figures.

All hyperparameters, seeds and paths live in `config.yaml` (see SCI_OPTIMIZATION_PLAN.md).
"""

__version__ = "0.2.0"
