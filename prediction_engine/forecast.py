"""
forecast.py — Forecast Engine
=============================
STEP 1 ONLY: time-series detection + ForecastResult skeleton.

This file contains NO model, NO FLAML, NO forecasting, and NO LLM. It:
  1. finds a usable time axis in the raw dataframe,
  2. validates the target,
  3. regularizes the observed series onto a regular frequency,
  4. returns a ForecastResult holding ONLY the history (forecast fields stay empty).

Design decision #3 (approved): date detection inspects the ORIGINAL raw columns and never
calls the profiler's numeric coercion — a date string must never be turned into a number
before we get a chance to parse it as a date.

Later steps will fill: baseline, backtest, model, forecast, intervals, confidence.
This file imports no LLM library, by design (ordering principle 0.0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Result container — the single source of every number (decision #5)
# --------------------------------------------------------------------------- #

@dataclass
class ForecastResult:
    success: bool
    time_col: str | None = None
    target: str | None = None
    frequency: str | None = None
    horizon: int = 0

    # observed, regularized series: list of {"ds": Timestamp, "y": float}
    history: list = field(default_factory=list)

    # everything below is filled by LATER steps; empty in Step 1
    forecast: list = field(default_factory=list)      # {ds, yhat, lower, upper}
    backtest: dict = field(default_factory=dict)
    baseline: dict = field(default_factory=dict)
    model: str | None = None
    quality_gate: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)

    events: list = field(default_factory=list)
    run_log: list = field(default_factory=list)
    error: str | None = None
    warnings: list = field(default_factory=list)

    # candidate time axes found during detection (for the UI to offer alternatives)
    time_candidates: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "success": self.success,
            "time_col": self.time_col,
            "target": self.target,
            "frequency": self.frequency,
            "n_history": len(self.history),
            "horizon": self.horizon,
            "error": self.error,
            "warnings": self.warnings,
            "time_candidates": self.time_candidates,
        }


# --------------------------------------------------------------------------- #
# Time-axis detection — inspects RAW columns, no numeric coercion (decision #3)
# --------------------------------------------------------------------------- #

@dataclass
class TimeCandidate:
    name: str
    kind: str            # "datetime" | "parsed_datetime" | "ordered_int"
    parse_rate: float    # share of non-null values that parse as dates (1.0 for ordered_int)
    regularity: float    # 0..1, how evenly spaced the values are
    inferred_freq: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind,
            "parse_rate": round(self.parse_rate, 3),
            "regularity": round(self.regularity, 3),
            "inferred_freq": self.inferred_freq,
        }


_MIN_PARSE_RATE = 0.95


def _try_parse_dates(s: pd.Series) -> tuple[pd.Series, float]:
    """Parse a raw column as datetimes WITHOUT mutating it. Returns (parsed, parse_rate)."""
    non_null = s.dropna()
    if non_null.empty:
        return pd.Series([], dtype="datetime64[ns]"), 0.0
    parsed = pd.to_datetime(non_null, errors="coerce")
    rate = float(parsed.notna().mean())
    return parsed, rate


def _regularity_and_freq(times: pd.Series) -> tuple[float, str | None]:
    """Given sorted unique datetimes, estimate how regular the spacing is and guess a freq."""
    t = pd.Series(pd.to_datetime(times)).dropna().sort_values().drop_duplicates()
    if len(t) < 3:
        return 0.0, None

    # pandas can sometimes name the frequency directly
    inferred = pd.infer_freq(t) if len(t) >= 3 else None

    deltas = t.diff().dropna().dt.total_seconds().to_numpy()
    if len(deltas) == 0 or deltas.max() == 0:
        return 0.0, inferred
    # regularity = 1 - normalized spread of the gaps (1.0 = perfectly even spacing)
    regularity = 1.0 - min(1.0, float(np.std(deltas) / (np.mean(deltas) + 1e-9)))

    if inferred is None:
        inferred = _guess_freq_from_seconds(float(np.median(deltas)))
    return max(0.0, regularity), inferred


def _guess_freq_from_seconds(sec: float) -> str | None:
    day = 86400.0
    table = [
        (day * 0.5, "H"), (day * 1.5, "D"), (day * 8, "W"),
        (day * 45, "M"), (day * 135, "Q"), (day * 400, "Y"),
    ]
    for threshold, code in table:
        if sec <= threshold:
            return code
    return "Y"


def detect_time_candidates(df: pd.DataFrame, target_col: str) -> list[TimeCandidate]:
    """Find every usable time axis, ranked best-first. Inspects raw columns only."""
    candidates: list[TimeCandidate] = []

    for col in df.columns:
        if col == target_col:
            continue
        s = df[col]

        # 1) already a real datetime
        if pd.api.types.is_datetime64_any_dtype(s):
            reg, freq = _regularity_and_freq(s)
            candidates.append(TimeCandidate(col, "datetime", 1.0, reg, freq))
            continue

        # 2) text/object that parses as dates (checked on the ORIGINAL column)
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            parsed, rate = _try_parse_dates(s)
            if rate >= _MIN_PARSE_RATE:
                reg, freq = _regularity_and_freq(parsed)
                candidates.append(TimeCandidate(col, "parsed_datetime", rate, reg, freq))
            continue

        # 3) a monotonically increasing integer column -> usable ordered index
        if pd.api.types.is_integer_dtype(s):
            v = s.dropna()
            if len(v) >= 3 and v.is_monotonic_increasing and v.is_unique:
                candidates.append(TimeCandidate(col, "ordered_int", 1.0, 1.0, None))
            continue

    # rank: real datetimes first, then by parse rate, then regularity
    kind_rank = {"datetime": 0, "parsed_datetime": 1, "ordered_int": 2}
    candidates.sort(key=lambda c: (kind_rank[c.kind], -c.parse_rate, -c.regularity))
    return candidates


# --------------------------------------------------------------------------- #
# Step 1 entry point
# --------------------------------------------------------------------------- #

def prepare_series(df: pd.DataFrame, target_col: str,
                   time_col: str | None = None,
                   frequency: str | None = None,
                   horizon: int | None = None) -> ForecastResult:
    """STEP 1: detect the time axis, validate target, build the regularized history.
    NO model is trained and NO forecast is produced here."""
    run_log: list[str] = []
    warnings: list[str] = []

    # ---- validate target ----
    if target_col not in df.columns:
        return ForecastResult(success=False, error=f"Target '{target_col}' not in data.")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        # the target must be numeric to forecast; we do NOT coerce silently here
        return ForecastResult(
            success=False, target=target_col,
            error=f"Target '{target_col}' is not numeric — forecasting needs a numeric measure.")

    # ---- detect time axis (raw columns, no coercion) ----
    candidates = detect_time_candidates(df, target_col)
    cand_dicts = [c.to_dict() for c in candidates]
    if not candidates:
        return ForecastResult(
            success=False, target=target_col, time_candidates=cand_dicts,
            error="No usable time axis found. This dataset does not look like a time series, "
                  "so no forecast is produced.")

    chosen = next((c for c in candidates if c.name == time_col), candidates[0])
    if time_col and chosen.name != time_col:
        warnings.append(f"Requested time column '{time_col}' is not usable; "
                        f"using '{chosen.name}' instead.")
    run_log.append(f"[Detection] time axis: '{chosen.name}' ({chosen.kind}, "
                   f"parse_rate={chosen.parse_rate:.2f}, regularity={chosen.regularity:.2f})")

    freq = frequency or chosen.inferred_freq
    run_log.append(f"[Detection] frequency: {freq or 'unknown'}")

    # ---- build the observed series ----
    work = df[[chosen.name, target_col]].copy()
    if chosen.kind == "ordered_int":
        work = work.sort_values(chosen.name)
        history = [{"ds": int(t), "y": float(y)}
                   for t, y in zip(work[chosen.name], work[target_col]) if pd.notna(y)]
        run_log.append(f"[Series] ordered integer index, {len(history)} points "
                       "(no calendar regularization).")
    else:
        work[chosen.name] = pd.to_datetime(work[chosen.name], errors="coerce")
        work = work.dropna(subset=[chosen.name]).sort_values(chosen.name)
        # collapse duplicate timestamps by mean, then regularize onto the frequency grid
        series = work.groupby(chosen.name)[target_col].mean()
        n_before = len(series)
        if freq:
            full_idx = pd.date_range(series.index.min(), series.index.max(), freq=freq)
            regular = series.reindex(full_idx)
            n_gaps = int(regular.isna().sum())
            if n_gaps:
                warnings.append(f"{n_gaps} gap(s) on the {freq} grid — left as missing for "
                                "a later step to handle (not filled in Step 1).")
            history = [{"ds": ts, "y": (None if pd.isna(v) else float(v))}
                       for ts, v in regular.items()]
            run_log.append(f"[Series] {n_before} timestamps regularized to {len(history)} "
                           f"points on '{freq}' grid ({n_gaps} gaps).")
        else:
            history = [{"ds": ts, "y": float(v)} for ts, v in series.items()]
            warnings.append("Frequency could not be inferred; series left at its raw timestamps.")
            run_log.append(f"[Series] {len(history)} points, irregular spacing.")

    observed = [h for h in history if h["y"] is not None]
    if len(observed) < 3:
        return ForecastResult(
            success=False, time_col=chosen.name, target=target_col, frequency=freq,
            time_candidates=cand_dicts, run_log=run_log,
            error=f"Only {len(observed)} observed point(s) — too short to forecast.")

    default_h = max(1, int(len(observed) * 0.1))
    h = horizon if horizon is not None else default_h
    if h > len(observed) / 3:
        warnings.append(f"Horizon {h} is large relative to history ({len(observed)} points); "
                        "forecasts far out will be unreliable.")

    return ForecastResult(
        success=True, time_col=chosen.name, target=target_col, frequency=freq,
        horizon=h, history=history, time_candidates=cand_dicts,
        run_log=run_log, warnings=warnings,
    )
