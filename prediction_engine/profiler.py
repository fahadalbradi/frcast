"""
profiler.py
============
Stage 1 of the loop: Profiling
Statistically analyzes an arbitrary dataframe to extract a "Data Fingerprint"
without any manual/domain-specific assumptions (Domain-Agnostic principle).
"""
from __future__ import annotations
import re
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    role: str               # "numeric" | "categorical" | "datetime" | "text" | "id_like"
    missing_pct: float
    cardinality: int
    sample_values: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


@dataclass
class DataFingerprint:
    n_rows: int
    n_cols: int
    task_type: str          # "regression" | "classification"
    target_col: str
    columns: list[ColumnProfile]
    overall_missing_pct: float
    duplicate_rows: int
    correlated_with_target: dict
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "task_type": self.task_type,
            "target_col": self.target_col,
            "overall_missing_pct": round(self.overall_missing_pct, 2),
            "duplicate_rows": self.duplicate_rows,
            "correlated_with_target": self.correlated_with_target,
            "warnings": self.warnings,
            "columns": [
                {
                    "name": c.name, "dtype": c.dtype, "role": c.role,
                    "missing_pct": round(c.missing_pct, 2),
                    "cardinality": c.cardinality,
                    "stats": c.stats,
                }
                for c in self.columns
            ],
        }


class DataProfiler:
    """Treats any dataset as a pure mathematical/statistical problem (Domain-Agnostic)."""

    ID_LIKE_UNIQUE_RATIO = 0.98
    MAX_CATEGORICAL_CARDINALITY = 30

    def profile(self, df: pd.DataFrame, target_col: str | None = None) -> DataFingerprint:
        warnings: list[str] = []
        n_rows, n_cols = df.shape

        target_col = target_col or self._infer_target(df)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        task_type = self._infer_task_type(df[target_col])

        columns: list[ColumnProfile] = []
        for col in df.columns:
            columns.append(self._profile_column(df, col))

        overall_missing_pct = float(df.isna().mean().mean() * 100)
        duplicate_rows = int(df.duplicated().sum())

        correlated_with_target = self._correlations_with_target(df, target_col, task_type)

        if n_rows < 50:
            warnings.append("Dataset is too small (<50 rows) — confidence in results will be low.")
        if overall_missing_pct > 30:
            warnings.append("The percentage of missing values is very high (>30%).")
        if duplicate_rows > 0:
            warnings.append(f"There are {duplicate_rows} duplicate rows in the data.")

        return DataFingerprint(
            n_rows=n_rows, n_cols=n_cols, task_type=task_type, target_col=target_col,
            columns=columns, overall_missing_pct=overall_missing_pct,
            duplicate_rows=duplicate_rows,
            correlated_with_target=correlated_with_target, warnings=warnings,
        )

    # ---------------------------------------------------------------- #

    def _infer_target(self, df: pd.DataFrame) -> str:
        """If no target given, pick the last numeric column as a sane default."""
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        return numeric_cols[-1] if numeric_cols else df.columns[-1]

    def _infer_task_type(self, series: pd.Series) -> str:
        if _is_textlike(series):
            return "classification"
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        if series.nunique(dropna=True) <= 15 and unique_ratio < 0.05:
            return "classification"
        return "regression"

    def _profile_column(self, df: pd.DataFrame, col: str) -> ColumnProfile:
        s = df[col]
        missing_pct = float(s.isna().mean() * 100)
        cardinality = int(s.nunique(dropna=True))
        unique_ratio = cardinality / max(len(s), 1)

        if pd.api.types.is_datetime64_any_dtype(s):
            role = "datetime"
        elif pd.api.types.is_float_dtype(s):
            # continuous measurements are never id-like, regardless of uniqueness
            role = "numeric"
        elif unique_ratio >= self.ID_LIKE_UNIQUE_RATIO and cardinality > 20:
            role = "id_like"
        elif pd.api.types.is_numeric_dtype(s):
            role = "numeric"
        elif cardinality <= self.MAX_CATEGORICAL_CARDINALITY:
            role = "categorical"
        else:
            role = "text"

        stats: dict[str, Any] = {}
        if role == "numeric":
            desc = s.describe()
            stats = {
                "mean": _safe_round(desc.get("mean")),
                "std": _safe_round(desc.get("std")),
                "min": _safe_round(desc.get("min")),
                "max": _safe_round(desc.get("max")),
                "skew": _safe_round(s.skew()),
            }
        elif role == "categorical":
            vc = s.value_counts(normalize=True).head(5)
            stats = {"top_categories": {str(k): round(float(v) * 100, 1) for k, v in vc.items()}}

        return ColumnProfile(
            name=col, dtype=str(s.dtype), role=role, missing_pct=missing_pct,
            cardinality=cardinality,
            sample_values=[str(v) for v in s.dropna().unique()[:3]],
            stats=stats,
        )

    def _correlations_with_target(self, df: pd.DataFrame, target_col: str, task_type: str) -> dict:
        if task_type != "regression":
            return {}
        numeric_df = df.select_dtypes(include=np.number)
        if target_col not in numeric_df.columns or numeric_df.shape[1] < 2:
            return {}
        corr = numeric_df.corr(numeric_only=True)[target_col].drop(target_col, errors="ignore")
        corr = corr.dropna().sort_values(key=lambda x: x.abs(), ascending=False)
        return {k: round(float(v), 3) for k, v in corr.head(8).items()}


def _is_textlike(s: pd.Series) -> bool:
    """pandas >=3.0 defaults string columns to a 'str' dtype instead of 'object',
    so we can't rely on `dtype == object` alone anymore."""
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )


def _safe_round(v, ndigits=4):
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return None


# --- Numeric values stored as text -------------------------------------------
# Some datasets store numbers as formatted strings: "$10,300", "38,005", "51,000 mi.".
# Left as-is they are profiled as text and the task is misread as classification.
# We coerce them back to numeric — but ONLY when the value really reads as a number:
# an optional currency symbol, then digits (with optional , separators / decimals),
# then at most a short trailing unit. This deliberately rejects IDs such as "P123",
# codes like "A1", and categories like "<1H OCEAN", which stay text.
_NUMERIC_LIKE = re.compile(
    r"^\s*[+-]?\s*[$€£¥₹﷼]?\s*[+-]?\d[\d,]*(?:\.\d+)?\s*[a-zA-Z.%/]{0,6}\s*$"
)
_NUM_EXTRACT = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")
_COERCE_MIN_SUCCESS_RATIO = 0.9   # convert only if ~all values read as numbers


def _to_number(v):
    m = _NUM_EXTRACT.search(str(v))
    if not m:
        return np.nan
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return np.nan


def _coerce_numeric_like(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (df, coerced_column_names). Columns whose text values are really numbers
    (currency / thousand-separated) are converted to numeric. Genuine categorical text
    and ID-like codes fail the check and are left untouched."""
    df = df.copy()
    coerced: list[str] = []

    for col in df.columns:
        s = df[col]
        if not _is_textlike(s):
            continue

        non_null = s.dropna()
        if non_null.empty:
            continue

        looks_numeric = non_null.astype(str).str.match(_NUMERIC_LIKE)
        if float(looks_numeric.mean()) < _COERCE_MIN_SUCCESS_RATIO:
            continue

        # float (not int): the profiler's id_like rule exempts float columns, so a
        # near-unique price column stays "numeric" instead of being read as an ID.
        df[col] = s.map(lambda v: np.nan if pd.isna(v) else _to_number(v)).astype("float64")
        coerced.append(col)

    return df, coerced