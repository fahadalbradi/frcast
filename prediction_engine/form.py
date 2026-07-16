"""
form.py
=======
Smart Prediction Form — decides WHICH raw columns the user actually has to fill in.

The problem this solves: SHAP reports importances for POST-preprocessing features
("district_Olaya", "date__year", "native_country" after target encoding), while the user
fills in RAW columns ("district", "date", "native_country"). This module walks the
preprocessing artefacts backwards, folds every derived feature back onto the raw column it
came from, and sums their importance.

Required  = the few raw columns that carry most of the model's signal.
Optional  = everything else, pre-filled with a sensible default so the user can ignore them.

It reads the frozen core's outputs; it changes nothing in it. No ML logic here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .profiler import _coerce_numeric_like


@dataclass
class FieldSpec:
    name: str                       # raw column name
    role: str                       # "numeric" | "categorical" | "other"
    importance: float               # summed |SHAP| of every feature derived from it
    default: object = None          # median / mode — used for optional fields
    options: list = field(default_factory=list)   # categorical choices


@dataclass
class FormSpec:
    required: list[FieldSpec]
    optional: list[FieldSpec]
    excluded: list[str]             # columns the pipeline dropped — never shown
    coverage: float                 # share of total importance covered by `required`

    def defaults(self) -> dict:
        return {f.name: f.default for f in self.required + self.optional}


def _origin_of(feature: str, prep, raw_cols: list[str]) -> str | None:
    """Map one post-preprocessing feature name back to its raw column."""
    # 1) one-hot: {"district": ["district_Olaya", "district_Malaz", ...]}
    for orig, dummies in getattr(prep, "onehot_cols", {}).items():
        if feature in dummies:
            return orig

    # 2) renamed for XGBoost safety: {"ocean_proximity_<1H OCEAN": "ocean_proximity__1H_OCEAN"}
    reverse_renamed = {v: k for k, v in getattr(prep, "renamed_cols", {}).items()}
    original = reverse_renamed.get(feature, feature)

    # the renamed name may itself be a dummy
    for orig, dummies in getattr(prep, "onehot_cols", {}).items():
        if original in dummies or original.startswith(f"{orig}_"):
            return orig

    # 3) datetime decomposition: "signup__year" -> "signup"
    if "__" in original:
        stem = original.split("__")[0]
        if stem in raw_cols:
            return stem

    # 4) label-encoded / target-encoded / passthrough numeric keep their name
    if original in raw_cols:
        return original

    # 5) last resort: longest raw column that prefixes the feature (one-hot on a value
    #    containing an underscore, e.g. "Contract_Month-to-month")
    candidates = [c for c in raw_cols if original.startswith(f"{c}_")]
    if candidates:
        return max(candidates, key=len)

    return None


def build_form_spec(engine_result, raw_df: pd.DataFrame,
                    target_coverage: float = 0.85,
                    min_required: int = 3,
                    max_required: int = 8) -> FormSpec:
    """Split the raw feature columns into required / optional using SHAP importance.

    `target_coverage`: keep adding columns until they account for this share of the total
    importance (bounded by min_required / max_required).
    """
    fp = engine_result.fingerprint
    prep = engine_result.preprocessing
    target_col = fp.target_col

    # ROOT CAUSE OF THE OLD TypeError:
    # orchestrator.run() coerces text-formatted numbers ("12,450", "$10,300") to float
    # BEFORE profiling, so the fingerprint's roles describe the COERCED frame — while this
    # function was handed the RAW frame. A column could therefore be role="numeric" and
    # dtype=string at the same time, and .median() blew up.
    # Fix: view the data exactly as the profiler did, using the engine's own coercion.
    view_df, _coerced = _coerce_numeric_like(raw_df)

    raw_cols = [c for c in raw_df.columns if c != target_col]
    profile_by_name = {c.name: c for c in fp.columns}

    # columns the pipeline threw away entirely — asking for them would be pointless
    excluded = [c for c in raw_cols
                if c in getattr(prep, "dropped_cols", [])
                or profile_by_name.get(c) and profile_by_name[c].role in ("id_like", "text")
                and c in getattr(prep, "dropped_cols", [])]

    askable = [c for c in raw_cols if c not in excluded]

    # ---- fold SHAP importances back onto raw columns ----
    top_features = engine_result.evaluation.explainability.get("top_features", []) or []
    importance: dict[str, float] = {c: 0.0 for c in askable}
    for item in top_features:
        origin = _origin_of(item["feature"], prep, raw_cols)
        if origin in importance:
            importance[origin] += abs(float(item["importance"]))

    total = sum(importance.values())

    # SHAP failed, or nothing mapped: fall back to the profiler's linear correlations,
    # and if that is empty too, ask for everything rather than guessing.
    if total <= 0:
        corr = fp.correlated_with_target or {}
        for c in askable:
            importance[c] = abs(float(corr.get(c, 0.0)))
        total = sum(importance.values())
        if total <= 0:
            fields = [_make_field(c, profile_by_name.get(c), view_df, 0.0) for c in askable]
            return FormSpec(required=fields, optional=[], excluded=excluded, coverage=1.0)

    ranked = sorted(askable, key=lambda c: importance[c], reverse=True)

    # ---- pick the required set ----
    required_names, acc = [], 0.0
    for c in ranked:
        if len(required_names) >= max_required:
            break
        if acc / total >= target_coverage and len(required_names) >= min_required:
            break
        required_names.append(c)
        acc += importance[c]

    required = [_make_field(c, profile_by_name.get(c), view_df, importance[c])
                for c in required_names]
    optional = [_make_field(c, profile_by_name.get(c), view_df, importance[c])
                for c in ranked if c not in required_names]

    return FormSpec(required=required, optional=optional, excluded=excluded,
                    coverage=round(acc / total, 4) if total else 0.0)


def _make_field(name: str, prof, df: pd.DataFrame, importance: float) -> FieldSpec:
    """`df` must be the COERCED view (the one the profiler saw), not the raw upload."""
    s = df[name]
    role = prof.role if prof is not None else "other"

    # Never trust `role` on its own: it describes the frame the profiler saw. If anything
    # ever drifts, verify against the actual dtype and coerce defensively rather than
    # calling .median() on strings.
    wants_numeric = (role == "numeric") or pd.api.types.is_numeric_dtype(s)

    if wants_numeric:
        num = s if pd.api.types.is_numeric_dtype(s) else pd.to_numeric(s, errors="coerce")
        if num.notna().any():
            return FieldSpec(name, "numeric", importance, round(float(num.median()), 4))
        # role said numeric but nothing converts -> fall through and treat it as a category

    mode = s.mode(dropna=True)
    default = mode.iloc[0] if not mode.empty else ""
    options = [v for v in s.dropna().unique().tolist()]
    return FieldSpec(name, "categorical", importance, default, options)
