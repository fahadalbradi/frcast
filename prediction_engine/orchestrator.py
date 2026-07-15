"""
orchestrator.py
================
The ReAct-style Workflow Orchestration loop:
    Profile -> Preprocess -> Train -> Evaluate -> (Self-Correct? retry) -> Result

Updated to support inverse logarithmic transformation (np.expm1) in predict mode.
"""
from __future__ import annotations
import traceback
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split

from .profiler import DataProfiler, DataFingerprint, _coerce_numeric_like
from .preprocessor import Preprocessor, PreprocessingResult
from .trainer import FLAMLTrainer, TrainingResult
from .evaluator import Evaluator, EvaluationResult
from .confidence import ConfidenceEngine, ConfidenceScore


@dataclass
class EngineResult:
    success: bool
    fingerprint: DataFingerprint | None = None
    preprocessing: PreprocessingResult | None = None
    training: TrainingResult | None = None
    evaluation: EvaluationResult | None = None
    confidence: ConfidenceScore | None = None
    attempts: int = 0
    run_log: list = field(default_factory=list)
    error: str | None = None
    recommendation: str = ""

    def summary(self) -> dict:
        if not self.success:
            return {"success": False, "error": self.error, "run_log": self.run_log}
        return {
            "success": True,
            "attempts": self.attempts,
            "data_fingerprint": self.fingerprint.to_dict(),
            "preprocessing_log": self.preprocessing.log,
            "best_estimator": self.training.best_estimator,
            "statistical": self.evaluation.statistical,
            "stability": self.evaluation.stability,
            "top_features": self.evaluation.explainability.get("top_features", []),
            "passed_quality_gate": self.evaluation.passed_quality_gate,
            "confidence": {
                "score": self.confidence.score,
                "label": self.confidence.label,
                "breakdown": self.confidence.breakdown,
            },
            "recommendation": self.recommendation,
            "run_log": self.run_log,
        }


class PredictionEngine:
    """Generalist Multi-Agent prediction engine (single-process orchestrator)."""

    MAX_ATTEMPTS = 3
    TEST_SIZE = 0.2   # held-out fraction used for the final (honest) metrics

    def __init__(self):
        self.profiler = DataProfiler()
        self.preprocessor = Preprocessor()
        self.trainer = FLAMLTrainer()
        self.evaluator = Evaluator()
        self.confidence_engine = ConfidenceEngine()

    def run(self, df: pd.DataFrame, target_col: str | None = None) -> EngineResult:
        run_log: list[str] = []

        # ---- Stage 0: numeric values stored as text ("$10,300") -> numeric ----
        # Must happen before profiling AND before preprocessing, since both consume this df.
        try:
            df, coerced_cols = _coerce_numeric_like(df)
            if coerced_cols:
                run_log.append(
                    f"[Profiling] Coerced text-formatted numeric column(s) to numeric: "
                    f"{', '.join(coerced_cols)}"
                )
        except Exception as e:
            run_log.append(f"[Profiling] Numeric coercion skipped: {e}")

        # ---- Stage 1: Profiling ----
        try:
            fingerprint = self.profiler.profile(df, target_col=target_col)
            run_log.append(f"[Profiling] {fingerprint.n_rows} rows × {fingerprint.n_cols} cols | "
                           f"Task: {fingerprint.task_type} | Target: {fingerprint.target_col}")
        except Exception as e:
            return EngineResult(success=False, error=f"Profiling failed: {e}", run_log=run_log + [traceback.format_exc()])

        # ---- Split BEFORE preprocessing (row labels only) ----
        # The Target Encoding inside the preprocessor must be fitted on training rows only,
        # so the split has to be decided first. Outlier removal / imputation are unchanged.
        stratify = None
        if fingerprint.task_type == "classification" and df[fingerprint.target_col].value_counts().min() >= 2:
            stratify = df[fingerprint.target_col]
        train_idx, test_idx = train_test_split(
            df.index, test_size=self.TEST_SIZE, random_state=42, stratify=stratify
        )

        # ---- Stage 2: Preprocessing ----
        try:
            prep = self.preprocessor.run(df, fingerprint, train_index=train_idx)
            run_log.append(f"[Preprocessing] {len(prep.log)} cleaning steps.")
            run_log.extend([f"[Preprocessing] {l}" for l in prep.log])
        except Exception as e:
            return EngineResult(success=False, fingerprint=fingerprint, error=f"Preprocessing failed: {e}", run_log=run_log + [traceback.format_exc()])

        X = prep.df[prep.feature_cols]
        y = prep.df[prep.target_col]

        # ---- [LOG] Shape of the data actually entering training ----
        run_log.append(f"[Preprocessing] Training Rows: {X.shape[0]}")
        run_log.append(f"[Preprocessing] Training Columns: {X.shape[1]}")

        if len(X) < 10:
            return EngineResult(success=False, fingerprint=fingerprint, preprocessing=prep, error="Insufficient data for training.", run_log=run_log)

        # ---- Apply the pre-decided split to the preprocessed rows ----
        # (rows dropped by outlier removal simply fall out of whichever side they were on)
        tr = X.index.intersection(pd.Index(train_idx))
        te = X.index.intersection(pd.Index(test_idx))
        X_train, y_train = X.loc[tr], y.loc[tr]
        X_test, y_test = X.loc[te], y.loc[te]
        run_log.append(
            f"[Split] Train: {len(X_train)} rows | Test (held-out): {len(X_test)} rows "
            f"| test_size={self.TEST_SIZE}"
        )

        # ---- Stage 3+4: Train -> Evaluate ----
        training: TrainingResult | None = None
        evaluation: EvaluationResult | None = None
        attempt = 0
        while attempt < self.MAX_ATTEMPTS:
            attempt += 1
            try:
                training = self.trainer.train(X_train, y_train, fingerprint.task_type, attempt=attempt)
                run_log.extend([f"[Training] {l}" for l in training.log])
                run_log.append(f"[Training] FLAML final selected model: {training.best_estimator}")

                # Statistical metrics on the held-out test set.
                # Stability (CV) is run on the training set only.
                evaluation = self.evaluator.evaluate(
                    training.model, X_test, y_test, fingerprint.task_type,
                    X_cv=X_train, y_cv=y_train,
                )
                run_log.append(
                    f"[Evaluation] Metrics computed on {len(X_test)} held-out rows "
                    f"(unseen during training) | CV run on {len(X_train)} training rows."
                )
                if evaluation.passed_quality_gate:
                    break
                run_log.append(f"[Self-Correction] Attempt #{attempt} did not pass quality gate.")
            except Exception as e:
                run_log.append(f"[Self-Healing] Error in attempt #{attempt}: {e}")
                continue

        if training is None or evaluation is None:
            return EngineResult(success=False, error="Training attempts exhausted.", attempts=attempt, run_log=run_log)

        confidence = self.confidence_engine.compute(fingerprint, evaluation)
        recommendation = self._build_recommendation(fingerprint, evaluation, confidence)

        return EngineResult(
            success=True, fingerprint=fingerprint, preprocessing=prep,
            training=training, evaluation=evaluation, confidence=confidence,
            attempts=attempt, run_log=run_log, recommendation=recommendation,
        )

    def predict(self, engine_result: EngineResult, new_rows: pd.DataFrame) -> list:
        """'Predict' mode — Forecast actual price after reversing logarithmic transformation."""
        if not engine_result.success:
            raise RuntimeError("No trained model available.")
        
        # 1. Preprocessing
        X_new = engine_result.preprocessing.transform_new(new_rows)
        
        # 2. Prediction
        raw_preds = engine_result.training.model.predict(X_new)

        # 3. Classification -> the model returns class labels ("<=50K", "Yes", ...).
        #    They must be returned as-is: no expm1, no float() cast.
        task_type = engine_result.fingerprint.task_type if engine_result.fingerprint else "regression"
        if task_type == "classification":
            return [p.item() if hasattr(p, "item") else p for p in raw_preds]

        # 4. Regression — inverse logarithmic transformation (if applied)
        if getattr(engine_result.preprocessing, 'target_log_transformed', False):
            preds = np.expm1(raw_preds)
        else:
            preds = raw_preds

        return [float(p) for p in preds]

    def _build_recommendation(self, fingerprint, evaluation, confidence) -> str:
        if not evaluation.passed_quality_gate:
            return "Digital prediction quality is low. It is recommended to use LLM Forecast or collect additional data."
        return "Data quality is sufficient — prediction can be relied upon."