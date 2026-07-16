"""
preprocessor.py
================
Stage 2 of the loop: Preprocessing
Updated to support:
  1. Outlier removal.
  2. Log transformation for the target column.
  3. Target encoding for text columns instead of dropping.
"""
from __future__ import annotations
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass, field

from .profiler import DataFingerprint, _is_textlike


# --- Feature-name sanitation -------------------------------------------------
# XGBoost/LightGBM reject feature names that are not plain strings or that contain
# '[', ']' or '<'. One-Hot Encoding builds names as "{col}_{category}", so any special
# character present in a category value leaks into the column name
# (e.g. California Housing: ocean_proximity + "<1H OCEAN" -> "ocean_proximity_<1H OCEAN").
_UNSAFE_CHARS = re.compile(r"[\[\]<>{}\"',:\s]+")


def _sanitize_name(name) -> str:
    s = _UNSAFE_CHARS.sub("_", str(name)).strip("_")
    if not s:
        s = "feature"
    if s[0].isdigit():
        s = "f_" + s
    return s


def _build_rename_map(columns, protected=()) -> dict:
    """Deterministic original -> safe name map, with collision suffixes."""
    rename_map: dict = {}
    used = set(protected)
    for col in columns:
        if col in protected:
            continue
        safe = _sanitize_name(col)
        if safe in used:
            i = 2
            while f"{safe}_{i}" in used:
                i += 1
            safe = f"{safe}_{i}"
        used.add(safe)
        rename_map[col] = safe
    return rename_map


@dataclass
class PreprocessingResult:
    df: pd.DataFrame
    feature_cols: list
    target_col: str
    encoders: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
    dropped_cols: list = field(default_factory=list)
    impute_values: dict = field(default_factory=dict)
    onehot_cols: dict = field(default_factory=dict)
    labelenc_cols: list = field(default_factory=list)
    passthrough_numeric: list = field(default_factory=list)
    # Added for log transformation handling
    target_log_transformed: bool = False
    # original -> XGBoost-safe column name (applied AFTER encoding)
    renamed_cols: dict = field(default_factory=dict)
    # col -> {"map": {category: mean}, "default": global_mean}  (replayed at predict time)
    target_encoded_cols: dict = field(default_factory=dict)

    def transform_new(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()
        
        # Log transform is not applied here as we expect raw input from the user 
        # (it will be transformed later if necessary)

        for col in self.dropped_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Target Encoding — replay the mapping learned at fit time.
        # Runs BEFORE imputation, exactly as in the training order (fit step 2, then step 3).
        # Unseen / missing categories fall back to the training global mean, so the model
        # never receives a raw string.
        for col, te in self.target_encoded_cols.items():
            if col in df.columns:
                df[col] = df[col].map(te["map"])

        for col, fill_val in self.impute_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_val)

        # Anything still unmapped (category unseen during training) -> safe fallback.
        for col, te in self.target_encoded_cols.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(te["default"])

        for col in self.labelenc_cols:
            if col in df.columns:
                le = self.encoders[col]
                df[col] = df[col].astype(str).map(
                    lambda v: le.transform([v])[0] if v in le.classes_ else -1
                )

        for orig_col, dummy_cols in self.onehot_cols.items():
            if orig_col in df.columns:
                dummies = pd.get_dummies(df[orig_col], prefix=orig_col, dtype=int)
                df = pd.concat([df.drop(columns=[orig_col]), dummies], axis=1)

        # Apply the exact same feature-name sanitation used at fit time,
        # otherwise reindex() below would not find the encoded columns.
        if self.renamed_cols:
            df = df.rename(columns=self.renamed_cols)

        return df.reindex(columns=self.feature_cols, fill_value=0)


class Preprocessor:
    def run(self, df: pd.DataFrame, fingerprint: DataFingerprint,
            train_index=None) -> PreprocessingResult:
        """train_index: row labels belonging to the training split. When provided, the
        Target Encoding means are computed from THOSE ROWS ONLY (no leakage from the test
        split) and then applied to every row. When None, behaviour is unchanged."""
        df = df.copy()
        log: list[str] = []
        dropped_cols: list[str] = []
        encoders: dict = {}
        impute_values: dict = {}
        onehot_cols: dict = {}
        labelenc_cols: list = []
        target_encoded_cols: dict = {}
        target_log_transformed = False

        target_col = fingerprint.target_col
        col_by_name = {c.name: c for c in fingerprint.columns}

        # --- [LOG] snapshot BEFORE cleaning (no logic change) ---
        rows_before = len(df)
        cols_before = df.shape[1]
        cols_before_names = list(df.columns)
        log.append(f"Rows before cleaning: {rows_before}")
        log.append(f"Columns before cleaning: {cols_before}")

        # Rows any fit-time statistic must be computed from (training split only).
        if train_index is not None:
            fit_idx = df.index.intersection(pd.Index(train_index))
        else:
            fit_idx = df.index
        if len(fit_idx) == 0:
            fit_idx = df.index

        # 0) Outlier removal — REGRESSION ONLY
        # Thresholds come from the TRAINING rows only (no information from the test split).
        # The same thresholds are then applied to every row.
        if fingerprint.task_type == "regression":
            before = len(df)
            q_low = df.loc[fit_idx, target_col].quantile(0.01)
            q_hi = df.loc[fit_idx, target_col].quantile(0.99)
            keep = (df[target_col] > q_low) & (df[target_col] < q_hi)
            # The held-out rows are NEVER dropped: the hardest cases must stay in the data
            # the model is scored on, otherwise the metrics flatter the model.
            keep.loc[keep.index.difference(fit_idx)] = True
            n_train_dropped = int((~keep.loc[fit_idx]).sum())
            df = df[keep]
            log.append(
                f"Outlier thresholds fitted on {len(fit_idx)} training rows only: "
                f"({q_low:,.4g}, {q_hi:,.4g})."
            )
            if before != len(df):
                log.append(
                    f"Outlier removal: {n_train_dropped} training rows dropped "
                    f"({before} -> {len(df)}). Test rows dropped: 0 (held-out set kept intact)."
                )
        else:
            log.append("Outlier removal: skipped (task type is classification).")

        # Training rows that actually survived outlier removal — every later fit-time
        # statistic (imputation, target encoding) must be computed from THESE rows only.
        fit_idx = df.index.intersection(fit_idx)

        # 1) Log transformation for target column — REGRESSION ONLY
        if fingerprint.task_type == "regression":
            df[target_col] = np.log1p(df[target_col])
            target_log_transformed = True
            log.append("Applied log transformation to target column.")
        else:
            target_log_transformed = False
            log.append("Log transformation: skipped (task type is classification).")

        # 2) Datetime and smart encoding
        # Target Encoding below averages the target per category, which requires a numeric
        # target. For classification the target is text (">50K" / "<=50K"), so we build a
        # TEMPORARY numeric view of it, used ONLY to compute those means. The real target
        # column is never modified here.
        if fingerprint.task_type == "classification":
            te_target = pd.Series(
                LabelEncoder().fit_transform(df[target_col].astype(str)),
                index=df.index,
            )
        else:
            te_target = df[target_col]

        # Rows the Target Encoding is FITTED on. Restricted to the training split so the
        # test rows never contribute to the category means (leakage fix).
        if train_index is not None:
            te_fit_idx = df.index.intersection(pd.Index(train_index))
        else:
            te_fit_idx = df.index
        if len(te_fit_idx) == 0:
            te_fit_idx = df.index

        for col, prof in col_by_name.items():
            if col == target_col or col not in df.columns:
                continue
            
            if prof.role == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}__year"] = df[col].dt.year
                df[f"{col}__month"] = df[col].dt.month
                df[f"{col}__dow"] = df[col].dt.dayofweek
                df.drop(columns=[col], inplace=True)
                log.append(f"Decomposed datetime column '{col}' into year/month/day.")
            
            elif prof.role in ("id_like", "text"):
                # Target Encoding for text columns of reasonable cardinality, otherwise drop.
                # The means are fitted on the TRAINING rows only (te_fit_idx) — see the
                # leakage fix. The mapping is stored so transform_new() can replay it.
                if prof.cardinality < 200:
                    te_fit_target = te_target.loc[te_fit_idx]
                    means = te_fit_target.groupby(df.loc[te_fit_idx, col]).mean()
                    default = float(te_fit_target.mean())
                    df[col] = df[col].map(means)
                    target_encoded_cols[col] = {
                        "map": {k: float(v) for k, v in means.items()},
                        "default": default,
                    }
                    log.append(
                        f"Applied Target Encoding to column '{col}' instead of dropping "
                        f"(fitted on {len(te_fit_idx)} training rows only)."
                    )
                else:
                    df.drop(columns=[col], inplace=True)
                    dropped_cols.append(col)
                    log.append(f"Dropped column '{col}' — lacks predictive signal.")

        # 3) Impute missing values
        # The fill value (median / mode) is computed from the TRAINING rows only, then
        # applied to every row — the test split must not influence its own imputation.
        for col in df.columns:
            if col == target_col: continue
            if df[col].isna().any():
                fit_col = df.loc[fit_idx, col]
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_val = fit_col.median()
                    if pd.isna(fill_val):          # every training value was missing
                        fill_val = 0
                    df[col] = df[col].fillna(fill_val)
                    impute_values[col] = fill_val
                else:
                    mode = fit_col.mode(dropna=True)
                    fill_val = mode.iloc[0] if not mode.empty else "unknown"
                    df[col] = df[col].fillna(fill_val)
                    impute_values[col] = fill_val
                log.append(
                    f"Imputed missing values in '{col}' with {impute_values[col]!r} "
                    f"(computed from {len(fit_idx)} training rows only)."
                )

        # 4) Encoding remaining categories
        for col in df.columns:
            if col == target_col or col in dropped_cols: continue
            if _is_textlike(df[col]):
                nunique = df[col].nunique()
                if nunique <= 15:
                    dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                    onehot_cols[col] = list(dummies.columns)
                    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                else:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    encoders[col] = le
                    labelenc_cols.append(col)

        # 5) Sanitize column names produced by encoding (XGBoost/LightGBM compatibility).
        # The target column name is left untouched (it is never passed as a feature).
        rename_map = _build_rename_map(list(df.columns), protected=(target_col,))
        changed = {k: v for k, v in rename_map.items() if k != v}
        if changed:
            df = df.rename(columns=changed)
            preview = ", ".join(f"'{k}' -> '{v}'" for k, v in list(changed.items())[:5])
            more = f" (+{len(changed) - 5} more)" if len(changed) > 5 else ""
            log.append(f"Sanitized {len(changed)} column name(s) for model compatibility: {preview}{more}")
        onehot_cols = {
            orig: [changed.get(c, c) for c in dummy_cols]
            for orig, dummy_cols in onehot_cols.items()
        }

        feature_cols = [c for c in df.columns if c != target_col]

        # --- [LOG] snapshot AFTER cleaning (no logic change) ---
        removed_cols = [c for c in cols_before_names if c not in df.columns]
        transformed_cols = [c for c in removed_cols if c not in dropped_cols]
        log.append(f"Rows after cleaning: {len(df)}  (removed: {rows_before - len(df)})")
        log.append(f"Columns after cleaning: {df.shape[1]}  (was {cols_before})")
        log.append(
            "Dropped columns (removed entirely): "
            + (", ".join(dropped_cols) if dropped_cols else "none")
        )
        log.append(
            "Transformed columns (replaced by encoded/derived features): "
            + (", ".join(transformed_cols) if transformed_cols else "none")
        )

        return PreprocessingResult(
            df=df, feature_cols=feature_cols, target_col=target_col,
            encoders=encoders, log=log, dropped_cols=dropped_cols,
            impute_values=impute_values, onehot_cols=onehot_cols,
            labelenc_cols=labelenc_cols, target_log_transformed=target_log_transformed,
            renamed_cols=changed, target_encoded_cols=target_encoded_cols,
        )