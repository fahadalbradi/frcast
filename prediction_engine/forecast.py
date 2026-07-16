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


# =========================================================================== #
# STEP 2 — Temporal preparation
# Regularize the time grid, detect missing periods, apply an EXPLICIT gap
# strategy. Still NO model, NO backtest, NO forecast, NO LLM.
# =========================================================================== #

# Gap-handling strategies, chosen explicitly by the caller (never silently):
#   "none"        : leave gaps as missing (Step 1 behaviour)
#   "ffill"       : carry the last observation forward (common, safe default for levels)
#   "linear"      : linear interpolation between neighbours (smooth series)
#   "zero"        : fill with 0 (only valid when absence truly means zero, e.g. counts)
#   "mean"        : fill with the series mean (weak; offered for completeness)
_GAP_STRATEGIES = ("none", "ffill", "linear", "zero", "mean")


@dataclass
class TemporalPrep:
    """Result of Step 2. Carries the regularized, gap-handled series plus a full audit
    of what was done — so the later backtest and the report can be honest about it."""
    success: bool
    time_col: str | None = None
    target: str | None = None
    frequency: str | None = None
    strategy: str = "none"

    series: list = field(default_factory=list)          # {ds, y} — regular grid, gaps handled
    n_periods: int = 0                                   # length of the regular grid
    n_observed: int = 0                                  # real observations
    n_filled: int = 0                                    # values created by the strategy
    gap_index: list = field(default_factory=list)        # positions that were originally missing
    gap_runs: list = field(default_factory=list)         # [{start, end, length}] consecutive gaps
    leading_trailing_trimmed: int = 0                    # edge gaps dropped, never invented

    run_log: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    error: str | None = None

    def summary(self) -> dict:
        return {
            "success": self.success, "time_col": self.time_col, "target": self.target,
            "frequency": self.frequency, "strategy": self.strategy,
            "n_periods": self.n_periods, "n_observed": self.n_observed,
            "n_filled": self.n_filled, "n_gap_runs": len(self.gap_runs),
            "leading_trailing_trimmed": self.leading_trailing_trimmed,
            "warnings": self.warnings, "error": self.error,
        }


def _find_gap_runs(mask_missing: list[bool]) -> list[dict]:
    """Group consecutive missing positions into runs."""
    runs, start = [], None
    for i, m in enumerate(mask_missing):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append({"start": start, "end": i - 1, "length": i - start})
            start = None
    if start is not None:
        runs.append({"start": start, "end": len(mask_missing) - 1,
                     "length": len(mask_missing) - start})
    return runs


def prepare_temporal(forecast_result: ForecastResult,
                     strategy: str) -> TemporalPrep:
    """STEP 2: take the ForecastResult history from Step 1 and produce a clean, regular,
    gap-handled series ready for a model.

    `strategy` is REQUIRED and explicit — there is deliberately no default. Choosing how to
    fill gaps is a semantic decision (a level series wants ffill/linear, a count series wants
    zero) that depends on what the target MEANS, which this step cannot know. Deferring that
    logic to a later stage means the caller must state the strategy every time. Pass "none" to
    regularize and audit the gaps without filling them.

    Interior gaps are FILLED per the chosen strategy; leading/trailing gaps are TRIMMED, never
    invented (you cannot back-cast history you never observed)."""
    if not forecast_result.success:
        return TemporalPrep(success=False, error="Step 1 did not succeed; nothing to prepare.")
    if strategy not in _GAP_STRATEGIES:
        return TemporalPrep(success=False, error=f"Unknown gap strategy '{strategy}'. "
                            f"Choose one of {_GAP_STRATEGIES}.")

    history = forecast_result.history
    run_log: list[str] = []
    warnings: list[str] = []

    y_raw = [h["y"] for h in history]
    ds = [h["ds"] for h in history]
    missing = [v is None or (isinstance(v, float) and pd.isna(v)) for v in y_raw]

    # --- trim leading / trailing gaps (never fabricate edge history) ---
    first = next((i for i, m in enumerate(missing) if not m), None)
    last = next((i for i in range(len(missing) - 1, -1, -1) if not missing[i]), None)
    if first is None:
        return TemporalPrep(success=False, time_col=forecast_result.time_col,
                            target=forecast_result.target, frequency=forecast_result.frequency,
                            error="No observed values in the series.")
    trimmed = (first) + (len(missing) - 1 - last)
    ds = ds[first:last + 1]
    y_raw = y_raw[first:last + 1]
    missing = missing[first:last + 1]
    if trimmed:
        run_log.append(f"[Temporal] Trimmed {trimmed} leading/trailing empty period(s) "
                       "(edge history is never fabricated).")

    gap_index = [i for i, m in enumerate(missing) if m]
    gap_runs = _find_gap_runs(missing)
    n_observed = len(missing) - len(gap_index)

    if gap_runs:
        longest = max(r["length"] for r in gap_runs)
        run_log.append(f"[Temporal] {len(gap_index)} missing interior period(s) in "
                       f"{len(gap_runs)} run(s); longest run = {longest}.")
        if longest >= max(3, int(0.1 * len(missing))):
            warnings.append(f"A gap run of {longest} periods is long relative to the series; "
                            f"values filled there by '{strategy}' are low-confidence.")
    else:
        run_log.append("[Temporal] No interior gaps on the regular grid.")

    # --- apply the explicit strategy to interior gaps ---
    s = pd.Series([np.nan if m else float(v) for v, m in zip(y_raw, missing)])
    if not gap_index or strategy == "none":
        filled = s
        applied = "none"
    elif strategy == "ffill":
        filled = s.ffill()
        applied = "ffill"
    elif strategy == "linear":
        filled = s.interpolate(method="linear", limit_direction="both")
        applied = "linear"
    elif strategy == "zero":
        filled = s.fillna(0.0)
        applied = "zero"
    else:  # mean
        filled = s.fillna(float(s.mean()))
        applied = "mean"

    n_filled = int(s.isna().sum() - filled.isna().sum())
    if n_filled:
        run_log.append(f"[Temporal] Filled {n_filled} interior gap(s) using '{applied}'.")

    still_missing = int(filled.isna().sum())
    if still_missing:
        warnings.append(f"{still_missing} value(s) still missing after '{applied}'. "
                        "A gap strategy must be chosen before modelling, or the model step "
                        "will have to handle missing values itself.")

    series = [{"ds": d, "y": (None if pd.isna(v) else float(v))}
              for d, v in zip(ds, filled)]

    return TemporalPrep(
        success=True,
        time_col=forecast_result.time_col, target=forecast_result.target,
        frequency=forecast_result.frequency, strategy=applied,
        series=series, n_periods=len(series), n_observed=n_observed,
        n_filled=n_filled, gap_index=gap_index, gap_runs=gap_runs,
        leading_trailing_trimmed=trimmed, run_log=run_log, warnings=warnings,
    )


# =========================================================================== #
# STEP 3a — Backtest harness + naive baseline (NO ML, fully testable)
# Rolling-origin backtest. A "forecaster" is any callable:
#     fn(history_values: list[float], horizon: int) -> list[float]
# The harness knows nothing about FLAML; the baseline and (later) the FLAML model
# both plug into it through this signature.
# =========================================================================== #

def _mae(actual, pred):
    a = np.asarray(actual, float); p = np.asarray(pred, float)
    return float(np.mean(np.abs(a - p)))


def _rmse(actual, pred):
    a = np.asarray(actual, float); p = np.asarray(pred, float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def _mape(actual, pred):
    a = np.asarray(actual, float); p = np.asarray(pred, float)
    mask = a != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


# ---- baseline forecasters (the mandatory naive, decision #4) ----

def naive_forecaster(history: list[float], horizon: int) -> list[float]:
    """Repeat the last observed value h times. The floor every model must beat."""
    return [float(history[-1])] * horizon


def seasonal_naive_forecaster(period: int):
    """Repeat the value from `period` steps ago. Returned as a closure so the harness
    can call it with the same (history, horizon) signature."""
    def _fn(history: list[float], horizon: int) -> list[float]:
        if len(history) < period:
            return naive_forecaster(history, horizon)
        # take the last full season, then repeat it forward
        last_season = history[-period:]
        return [float(last_season[i % period]) for i in range(horizon)]
    return _fn


def backtest(series_values: list[float], forecaster, horizon: int,
             n_folds: int = 5, min_train: int | None = None) -> dict:
    """Rolling-origin backtest. Trains on a growing prefix, forecasts `horizon` ahead,
    scores against the held-out actuals. TIME-ORDERED — never shuffles (decision: temporal
    leakage). Returns per-fold and overall MAE/RMSE/MAPE plus the residuals (for later
    interval construction)."""
    y = [float(v) for v in series_values]
    n = len(y)
    if min_train is None:
        min_train = max(horizon, n // 2)

    # fold origins: last training index for each fold, spaced across the tail
    last_origin = n - horizon
    if last_origin <= min_train:
        return {"error": f"series too short for backtest: need > {min_train + horizon} points, "
                         f"have {n}.", "folds": [], "overall": {}}

    origins = sorted(set(
        int(round(min_train + i * (last_origin - min_train) / max(1, n_folds - 1)))
        for i in range(n_folds)
    ))
    origins = [o for o in origins if min_train <= o <= last_origin]

    folds, all_actual, all_pred = [], [], []
    residuals_by_step: dict[int, list] = {}
    for origin in origins:
        train = y[:origin]
        actual = y[origin:origin + horizon]
        pred = forecaster(train, horizon)[:horizon]
        if len(pred) < len(actual):                       # forecaster returned short
            pred = list(pred) + [pred[-1]] * (len(actual) - len(pred))
        folds.append({
            "train_end": origin,
            "mae": _mae(actual, pred),
            "rmse": _rmse(actual, pred),
            "mape": _mape(actual, pred),
        })
        all_actual.extend(actual)
        all_pred.extend(pred)
        # residual per horizon step (step 1 = first period ahead, ...)
        for step, (a, p) in enumerate(zip(actual, pred), start=1):
            residuals_by_step.setdefault(step, []).append(float(a - p))

    residuals = [float(a - p) for a, p in zip(all_actual, all_pred)]
    overall = {
        "mae": _mae(all_actual, all_pred),
        "rmse": _rmse(all_actual, all_pred),
        "mape": _mape(all_actual, all_pred),
        "n_folds": len(folds),
        "n_points_scored": len(all_actual),
    }
    return {"folds": folds, "overall": overall, "residuals": residuals,
            "residuals_by_step": residuals_by_step}


# =========================================================================== #
# STEP 3b — FLAML ts_forecast wrapper
# ---------------------------------------------------------------------------
# !!! NOT EXECUTED in the build environment: flaml[ts_forecast] is not installed
#     and the network is offline here. The code below is written against FLAML's
#     documented ts_forecast API and MUST be validated on a machine with FLAML
#     before it is trusted. Everything ABOVE this line (harness + baselines) IS
#     tested and verified. This wrapper is deliberately isolated behind the same
#     forecaster signature so the tested harness is what actually scores it.
# =========================================================================== #

def make_flaml_forecaster(frequency: str, time_budget: int = 30):
    """Return a forecaster closure fn(history_values, horizon) backed by FLAML ts_forecast.

    FLAML's time-series API expects a dataframe with a time column and a target column, so the
    closure rebuilds a minimal synthetic time index from the history length. The frequency is
    used to advance the index for the forecast horizon.

    UNTESTED here — see the banner above.
    """
    def _fn(history: list[float], horizon: int) -> list[float]:
        from flaml import AutoML                       # lazy: keeps the module importable

        freq = frequency or "D"
        idx = pd.date_range("2000-01-01", periods=len(history), freq=freq)
        train_df = pd.DataFrame({"ds": idx, "y": [float(v) for v in history]})

        automl = AutoML()
        automl.fit(
            dataframe=train_df, label="y",
            task="ts_forecast", time_budget=time_budget,
            period=horizon, eval_method="holdout", verbose=0,
        )
        future_idx = pd.date_range(idx[-1], periods=horizon + 1, freq=freq)[1:]
        pred = automl.predict(pd.DataFrame({"ds": future_idx}))
        return [float(v) for v in np.asarray(pred).ravel()[:horizon]]

    return _fn


def run_forecast_evaluation(temporal_prep, horizon: int,
                            frequency: str,
                            seasonal_period: int | None = None,
                            use_flaml: bool = True,
                            n_folds: int = 5,
                            time_budget: int = 30) -> dict:
    """Glue for Step 3: backtest the mandatory naive baseline, optionally backtest the FLAML
    model on the SAME harness, and compare. Decision #4: the naive baseline is always run.
    This returns a plain dict; wiring it into ForecastResult + the quality gate is Step 4.

    NOTE: the FLAML branch depends on make_flaml_forecaster, which is UNTESTED in this
    environment. If FLAML is absent or errors, the baseline result is still returned and the
    failure is reported — never a fabricated model score.
    """
    values = [h["y"] for h in temporal_prep.series if h["y"] is not None]
    out = {"horizon": horizon, "frequency": frequency, "baseline": {}, "model": {},
           "comparison": {}, "notes": []}

    # --- mandatory naive baseline (always) ---
    out["baseline"]["naive"] = backtest(values, naive_forecaster, horizon, n_folds)
    if seasonal_period:
        out["baseline"]["seasonal_naive"] = backtest(
            values, seasonal_naive_forecaster(seasonal_period), horizon, n_folds)

    # --- FLAML model (optional, isolated, may fail) ---
    if use_flaml:
        try:
            fc = make_flaml_forecaster(frequency, time_budget)
            out["model"]["flaml"] = backtest(values, fc, horizon, n_folds)
        except Exception as e:
            out["notes"].append(f"FLAML forecast unavailable: {e}")
            out["model"]["flaml"] = {"error": str(e), "folds": [], "overall": {}}

    # --- comparison (only if both scored) ---
    base_mae = out["baseline"]["naive"].get("overall", {}).get("mae")
    model_mae = out["model"].get("flaml", {}).get("overall", {}).get("mae")
    if base_mae is not None and model_mae is not None:
        out["comparison"] = {
            "naive_mae": base_mae, "model_mae": model_mae,
            "model_beats_naive": model_mae < base_mae,
            "improvement_pct": round((base_mae - model_mae) / base_mae * 100, 2) if base_mae else None,
        }
    return out


# =========================================================================== #
# STEP 4 — Prediction intervals + forecast quality gate + ForecastResult wiring
# Intervals are EMPIRICAL (from backtest residuals), never a distributional
# assumption. The gate enforces decision #4 (must beat naive). No LLM.
# =========================================================================== #

def build_intervals(point_forecast: list[float],
                    residuals_by_step: dict,
                    level: float = 0.80) -> list[dict]:
    """Empirical prediction intervals. For each horizon step, take the quantiles of the
    backtest residuals AT THAT STEP, so the band widens with the horizon. Falls back to the
    pooled residuals for steps that were never scored.

    Returns [{"yhat", "lower", "upper"}] aligned with point_forecast."""
    lo_q = (1 - level) / 2
    hi_q = 1 - lo_q
    pooled = [r for rs in residuals_by_step.values() for r in rs]

    # Cumulative pooling: the band at step h uses residuals from steps 1..h. Later horizons
    # therefore include the (larger) errors of far-ahead forecasts, so the band is
    # non-decreasing and built on a bigger, less noisy sample than a single step alone.
    out = []
    for i, yhat in enumerate(point_forecast, start=1):
        res = [r for step, rs in residuals_by_step.items() if step <= i for r in rs] or pooled
        if res:
            lo = float(np.quantile(res, lo_q))
            hi = float(np.quantile(res, hi_q))
        else:
            lo = hi = 0.0
        out.append({"yhat": float(yhat),
                    "lower": float(yhat + lo),   # residual = actual - pred, so add quantiles
                    "upper": float(yhat + hi)})
    return out


def _interval_coverage(backtest_result: dict, level: float = 0.80) -> float | None:
    """Back-check: across backtest folds, what fraction of actuals fell inside the empirical
    band built from the SAME residuals? Reported honestly even when far from nominal."""
    rbs = backtest_result.get("residuals_by_step") or {}
    pooled = [r for rs in rbs.values() for r in rs]
    if not pooled:
        return None
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    inside = 0
    total = 0
    for step, res in rbs.items():
        lo, hi = np.quantile(pooled, lo_q), np.quantile(pooled, hi_q)
        for r in res:                      # r is (actual - pred); inside if lo <= r <= hi
            inside += int(lo <= r <= hi)
            total += 1
    return float(inside / total) if total else None


# quality-gate thresholds for forecasting (distinct from the tabular Evaluator)
FORECAST_MIN_IMPROVEMENT = 0.0      # must beat naive (decision #4): improvement > 0
COVERAGE_TOLERANCE = 0.15           # warn if |coverage - level| exceeds this


def forecast_quality_gate(evaluation: dict, coverage: float | None,
                          level: float = 0.80) -> dict:
    """Decision #4 enforced here: reject if the model does not beat the naive baseline.
    Also warns (does not reject) on poor interval coverage and short history."""
    reasons, warnings = [], []

    comp = evaluation.get("comparison", {})
    model_mae = comp.get("model_mae")
    naive_mae = comp.get("naive_mae")

    if model_mae is None:
        reasons.append("No model score available (FLAML did not produce a forecast).")
    elif naive_mae is not None and not (model_mae < naive_mae):
        reasons.append(f"Model MAE ({model_mae:.4g}) does not beat naive baseline "
                       f"({naive_mae:.4g}).")

    if coverage is not None and abs(coverage - level) > COVERAGE_TOLERANCE:
        warnings.append(f"Interval coverage {coverage:.0%} is far from the nominal {level:.0%} — "
                        "prediction bands may be mis-calibrated.")

    return {"passed": len(reasons) == 0, "reasons": reasons, "warnings": warnings}


def _forecast_confidence(evaluation: dict, coverage: float | None,
                         n_history: int, level: float = 0.80) -> dict:
    """Forecast-specific confidence (decision #2: SEPARATE from the tabular ConfidenceEngine).
    Blends: how much the model beats naive, interval calibration, and history length."""
    comp = evaluation.get("comparison", {})
    imp = comp.get("improvement_pct")
    skill = 0.0 if imp is None else max(0.0, min(1.0, imp / 50.0))     # 50% improvement -> full
    calib = 0.5 if coverage is None else max(0.0, 1 - abs(coverage - level) / level)
    adequacy = max(0.0, min(1.0, n_history / 100.0))                   # saturates at 100 points

    score = round(0.5 * skill + 0.3 * calib + 0.2 * adequacy, 3)
    label = "High" if score >= 0.7 else "Medium" if score >= 0.4 else "Low"
    return {"score": score, "label": label,
            "breakdown": {"skill_vs_naive": round(skill, 3),
                          "interval_calibration": round(calib, 3),
                          "history_adequacy": round(adequacy, 3)}}


def assemble_forecast_result(temporal_prep, evaluation: dict,
                             point_forecast: list[float],
                             future_ds: list,
                             level: float = 0.80) -> ForecastResult:
    """STEP 4 wiring: turn the Step-3 evaluation + a point forecast into a complete
    ForecastResult — the single source of every number (decision #5). Intervals, gate,
    baseline, confidence all filled here. Still NO LLM."""
    model_bt = evaluation.get("model", {}).get("flaml", {})
    residuals_by_step = model_bt.get("residuals_by_step", {}) or {}

    intervals = build_intervals(point_forecast, residuals_by_step, level)
    forecast = [{"ds": ds, **iv} for ds, iv in zip(future_ds, intervals)]

    coverage = _interval_coverage(model_bt, level)
    gate = forecast_quality_gate(evaluation, coverage, level)
    n_hist = len([h for h in temporal_prep.series if h["y"] is not None])
    confidence = _forecast_confidence(evaluation, coverage, n_hist, level)

    baseline = {"naive": evaluation.get("baseline", {}).get("naive", {}).get("overall", {})}
    if "seasonal_naive" in evaluation.get("baseline", {}):
        baseline["seasonal_naive"] = evaluation["baseline"]["seasonal_naive"]["overall"]

    run_log = [
        f"[Forecast] horizon={len(point_forecast)} freq={temporal_prep.frequency}",
        f"[Forecast] naive MAE={baseline['naive'].get('mae')}, "
        f"model MAE={model_bt.get('overall', {}).get('mae')}",
        f"[Forecast] interval level={level:.0%}, empirical coverage="
        f"{None if coverage is None else round(coverage, 3)}",
        f"[Forecast] quality gate: {'PASSED' if gate['passed'] else 'REJECTED'}",
    ]

    return ForecastResult(
        success=gate["passed"],
        time_col=temporal_prep.time_col, target=temporal_prep.target,
        frequency=temporal_prep.frequency, horizon=len(point_forecast),
        history=temporal_prep.series,
        forecast=forecast,
        backtest={"model": model_bt.get("overall", {}),
                  "folds": model_bt.get("folds", []),
                  "interval_coverage": coverage},
        baseline=baseline,
        model=evaluation.get("model", {}).get("name", "flaml_ts_forecast"),
        quality_gate=gate,
        confidence=confidence,
        run_log=run_log,
        warnings=gate["warnings"],
        error=None if gate["passed"] else "; ".join(gate["reasons"]),
    )
