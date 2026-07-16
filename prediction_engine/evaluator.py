"""
evaluator.py
============
Stage 4 of the loop: Evaluation
Updated: Inverse log transformation applied before statistical metrics calculation 
to ensure R2 and MAE reflect real price values.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
import shap


@dataclass
class EvaluationResult:
    task_type: str
    statistical: dict
    stability: dict
    explainability: dict
    passed_quality_gate: bool
    rejection_reasons: list = field(default_factory=list)


class Evaluator:
    # Rejection thresholds: The system rejects statistically weak results to prevent misleading predictions
    MIN_R2 = 0.35
    MIN_ACCURACY = 0.55
    MAX_CV_STD_RATIO = 0.5   # std/mean from CV scores

    def evaluate(self, trained_model, X, y, task_type: str,
                 X_cv=None, y_cv=None) -> EvaluationResult:
        """X / y  -> held-out data. Statistical metrics are computed ONLY on these.
        X_cv / y_cv -> training data, used for Cross-Validation stability only.
        """
        # Prediction on held-out data
        preds_log = trained_model.predict(X)

        # 1) Statistical
        if task_type == "regression":
            # Restore real values — the regression target was log1p-transformed
            y_true = np.expm1(y)
            preds = np.expm1(preds_log)
            statistical = {
                "r2": round(float(r2_score(y_true, preds)), 4),
                "mae": round(float(mean_absolute_error(y_true, preds)), 4),
            }
            primary_score = statistical["r2"]
        else:
            statistical = {
                "accuracy": round(float(accuracy_score(y, preds_log)), 4), # Classification does not need log
                "f1_weighted": round(float(f1_score(y, preds_log, average="weighted")), 4),
            }
            primary_score = statistical["accuracy"]

        # 2) Stability — Cross-Validation on the training split (falls back to X/y)
        X_stab = X_cv if X_cv is not None else X
        y_stab = y_cv if y_cv is not None else y
        stability = self._stability(trained_model, X_stab, y_stab, task_type)

        # 3) Explainability (SHAP) — computed on the held-out data
        explainability = self._explainability(trained_model, X, task_type)

        # Quality gate
        reasons = []
        if task_type == "regression" and primary_score < self.MIN_R2:
            reasons.append(f"R2 ({primary_score}) is below the minimum acceptable threshold ({self.MIN_R2}).")
        if task_type == "classification" and primary_score < self.MIN_ACCURACY:
            reasons.append(f"Accuracy ({primary_score}) is below the minimum acceptable threshold ({self.MIN_ACCURACY}).")
        
        cv_mean = stability.get("cv_mean")
        cv_std = stability.get("cv_std")
        if cv_mean is not None and cv_mean != 0 and (cv_std / abs(cv_mean)) > self.MAX_CV_STD_RATIO:
            reasons.append("High volatility between Cross-Validation folds — model is unstable.")

        return EvaluationResult(
            task_type=task_type, statistical=statistical, stability=stability,
            explainability=explainability, passed_quality_gate=(len(reasons) == 0),
            rejection_reasons=reasons,
        )

    def _stability(self, trained_model, X, y, task_type: str) -> dict:
        try:
            estimator = trained_model.model.estimator
            n_splits = min(5, max(3, len(X) // 30)) if len(X) >= 30 else 3
            if task_type == "regression":
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                scores = cross_val_score(estimator, X, y, cv=cv, scoring="r2")
            else:
                # cross_val_score REFITS a clone of the raw estimator, bypassing the label
                # encoding FLAML applies internally. XGBClassifier then rejects text labels
                # ("Invalid classes inferred from unique values of `y`. Expected: [0 1],
                #  got ['<=50K' '>50K']"), so all folds fail. Encode the labels here — for
                # the CV only. Accuracy is invariant to class renaming, so the score is
                # unchanged; this merely makes it computable.
                y_cv = LabelEncoder().fit_transform(np.asarray(y).astype(str))
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                scores = cross_val_score(estimator, X, y_cv, cv=cv, scoring="accuracy")
            return {
                "cv_mean": round(float(np.mean(scores)), 4),
                "cv_std": round(float(np.std(scores)), 4),
                "n_splits": n_splits,
                "fold_scores": [round(float(s), 4) for s in scores],
            }
        except Exception as e:
            return {"cv_mean": None, "cv_std": None, "error": str(e)}

    def _explainability(self, trained_model, X, task_type: str, max_rows: int = 200) -> dict:
        try:
            estimator = trained_model.model.estimator
            X_sample = X.sample(min(max_rows, len(X)), random_state=42) if len(X) > max_rows else X
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_values = np.mean(np.abs(np.array(shap_values)), axis=0)
            else:
                shap_values = np.abs(shap_values)
                if shap_values.ndim == 3:
                    shap_values = shap_values.mean(axis=2)

            mean_abs = np.mean(shap_values, axis=0)
            importance = sorted(
                zip(X_sample.columns, mean_abs), key=lambda t: t[1], reverse=True
            )
            top_features = [{"feature": f, "importance": round(float(v), 5)} for f, v in importance[:10]]
            return {"top_features": top_features, "method": "TreeExplainer(SHAP)"}
        except Exception as e:
            return {"top_features": [], "method": None, "error": str(e)}