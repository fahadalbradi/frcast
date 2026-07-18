"""
confidence.py
=============
Attaches a Confidence Score (0-1) to every prediction, combining:
  - model quality (statistical score)
  - stability (CV std)
  - data adequacy (row count, missingness, duplicates)
This is what lets the system be transparent about *how much to trust* a result.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ConfidenceScore:
    score: float          # 0..1
    label: str             # "High" | "Medium" | "Low"
    breakdown: dict


class ConfidenceEngine:
    def compute(self, fingerprint, evaluation) -> ConfidenceScore:
        # --- quality component ---
        if evaluation.task_type == "regression":
            quality = _clip01(evaluation.statistical.get("r2", 0))
        else:
            quality = _clip01(evaluation.statistical.get("accuracy", 0))

        # --- stability component ---
        cv_mean = evaluation.stability.get("cv_mean")
        cv_std = evaluation.stability.get("cv_std")
        if cv_mean is None:
            stability = 0.5
        else:
            spread = abs(cv_std) / (abs(cv_mean) + 1e-6)
            stability = _clip01(1 - min(spread, 1.0))

        # --- data adequacy component ---
        row_score = _clip01(fingerprint.n_rows / 500)   # saturates at 500 rows
        missing_penalty = _clip01(1 - fingerprint.overall_missing_pct / 100)
        dup_penalty = _clip01(1 - (fingerprint.duplicate_rows / max(fingerprint.n_rows, 1)))
        data_adequacy = (row_score + missing_penalty + dup_penalty) / 3

        # weighted blend
        score = 0.5 * quality + 0.3 * stability + 0.2 * data_adequacy
        score = round(_clip01(score), 3)

        if score >= 0.75:
            label = "مرتفعة (High)"
        elif score >= 0.5:
            label = "متوسطة (Medium)"
        else:
            label = "منخفضة (Low)"

        return ConfidenceScore(
            score=score, label=label,
            breakdown={
                "quality": round(quality, 3),
                "stability": round(stability, 3),
                "data_adequacy": round(data_adequacy, 3),
            },
        )


def _clip01(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = 0.0
    return max(0.0, min(1.0, v))