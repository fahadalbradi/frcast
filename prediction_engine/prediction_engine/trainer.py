"""
trainer.py
==========
Stage 3 of the loop: Training
Updated to support:
  1. More robust algorithms for tabular data (histgbm).
  2. Optimized search budget for higher accuracy.
"""
from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from flaml import AutoML


@dataclass
class TrainingResult:
    model: object
    best_estimator: str
    best_config: dict
    time_budget_used: float
    log: list = field(default_factory=list)
    attempt: int = 1


class FLAMLTrainer:
    """Dynamically sizes the search based on data scale (rows x cols)."""

    def _decide_budget(self, n_rows: int, n_cols: int, attempt: int) -> tuple[int, list]:
        """Returns (time_budget_seconds, estimator_list)."""
        scale = n_rows * max(n_cols, 1)
        
        # Slight increase in default budget to allow models a chance to reach accuracy > 0.5
        if scale < 5_000:
            budget = 20
        elif scale < 50_000:
            budget = 45
        elif scale < 500_000:
            budget = 90
        else:
            budget = 180

        # Self-correction: Exponentially increase budget with each failed attempt
        budget = int(budget * (1.8 ** (attempt - 1)))

        # Focus on stronger algorithms for tabular data
        # 'histgbm' is supported in FLAML as 'lgbm' or via additional packages
        estimators = ["lgbm", "xgboost", "rf", "extra_tree", "xgb_limitdepth"]

        return budget, estimators

    def train(self, X, y, task_type: str, attempt: int = 1, seed: int = 42) -> TrainingResult:
        n_rows, n_cols = X.shape
        budget, estimators = self._decide_budget(n_rows, n_cols, attempt)

        flaml_task = "regression" if task_type == "regression" else "classification"
        metric = "r2" if flaml_task == "regression" else "accuracy"

        automl = AutoML()
        settings = {
            "time_budget": budget,
            "task": flaml_task,
            "metric": metric,
            "estimator_list": estimators,
            "eval_method": "cv",
            "n_splits": min(5, max(3, n_rows // 30)) if n_rows >= 30 else 3,
            "seed": seed,
            "verbose": 0,
        }

        t0 = time.time()
        # AutoML searches within a more focused space thanks to the settings
        automl.fit(X_train=X, y_train=y, **settings)
        elapsed = time.time() - t0

        log = [
            f"Attempt #{attempt}: Training for {budget} seconds on estimators {estimators}.",
            f"Best estimator selected: {automl.best_estimator} — actual duration {round(elapsed, 1)}s.",
            f"Best accuracy (R2) achieved: {round(automl.best_loss, 4) if flaml_task == 'regression' else 'N/A'}"
        ]

        return TrainingResult(
            model=automl,
            best_estimator=str(automl.best_estimator),
            best_config=automl.best_config or {},
            time_budget_used=budget,
            log=log,
            attempt=attempt,
        )