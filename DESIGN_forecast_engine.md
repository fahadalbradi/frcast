# Design Document — Forecast Engine

Status: **DESIGN ONLY. No forecast code exists yet.** This document defines the architecture,
inputs, outputs, visuals and report before any implementation.

Scope of this document: replace the current `ForecastTool` placeholder with a real
time-series forecasting path. It does NOT change the ML core, the router, the events layer,
or the smart form.

---

## 0. Governing principles (carried from the whole benchmark sprint)

### FINALIZED DECISIONS (approved)

1. **Forecast v1 uses FLAML `ts_forecast` only.** No Prophet for now.
2. **Forecast confidence is a SEPARATE score**, not the tabular `ConfidenceEngine`.
3. **Date detection runs BEFORE any numeric coercion** inside the forecast path. The detector
   inspects the ORIGINAL raw column and never calls `_coerce_numeric_like`.
4. **The naive baseline is mandatory and cannot be skipped.** A model that does not beat it is
   rejected by the forecast quality gate.
5. **`ForecastResult` is the single source of every number.**
6. **Order is locked:** ForecastResult -> Visuals -> Structured Report -> LLM Narrative.

### LLM provider (finalized)

The user currently holds an OpenAI key and NO Anthropic key. Therefore:
- Any future LLM layer (`forecast_report.py` narrative, the router's intent classifier) must
  work with OpenAI.
- The forecast/report engines must NOT be hardwired to Claude/Anthropic.
- Target design is provider-agnostic (a thin LLM-client interface); if one provider must be the
  default now, it is OpenAI.
- Design intent recorded for later steps only. NO LLM layer is built in Step 1, and the existing
  `llm_agent.py` (Anthropic, for the old tabular tab) is left untouched and unused here.

### 0.0 STRICT ORDERING — non-negotiable

The LLM runs LAST, and only after every number already exists. No exceptions.

```
ForecastResult   (all numbers computed here — deterministic, no LLM)
      |
      v
   Visuals        (each chart bound to a value in ForecastResult)
      |
      v
   Report         (structured sections filled from ForecastResult)
      |
      v
   LLM Narrative  (prose only, receives the finished result, invents nothing)
```

Enforcement, not just intention:
- The LLM layer is a SEPARATE module (`forecast_report.py`) whose single entry point takes a
  COMPLETED `ForecastResult` as input. It is architecturally incapable of running earlier: it
  has no access to the raw data or the model, only to the finished result.
- The forecast core (`forecast.py`) imports NO LLM library. If someone tries to make it call an
  LLM, the dependency graph makes it obvious.
- If the LLM is unavailable (no key, call fails), the pipeline still produces the full
  ForecastResult, all visuals, and the structured report. Only the narrative section is omitted.
  The numbers never depend on the LLM being present.
- The number-grounding validator (section 5) runs AFTER the narrative and can only remove or
  flag hallucinated figures — it can never add a number the core did not compute.

This is the direct lesson from the current "LLM Forecast": there, prose was generated around
metrics with no ordering guarantee and no verification. Here the order is inverted and locked.

### Other principles

1. **Numbers come from deterministic code. The LLM only narrates.** Every figure in the
   report is computed by the forecast core and passed to the LLM as input. A validator checks
   that every number appearing in the generated text exists in the structured result. This is
   the fix for the old "LLM Forecast" which invented prose around metrics with no verification.
2. **Temporal leakage is not the same as tabular leakage.** There is NO random
   `train_test_split` here. The split is by time: train on the past, test on the most recent
   window. Backtesting walks forward. Shuffling would let the model see the future.
3. **Honest placeholders over fake output.** If a dataset has no usable time axis, the engine
   says so and stops. It never fabricates a trend — the same stance the current placeholder
   already takes.
4. **New file(s) only.** `forecast.py` (core) and a `forecast_report.py` (report/validator).
   The ML core stays frozen.

---

## 1. Architecture

Layered, mirroring the existing prediction path so the two feel consistent.

```
Router (forecast intent)  ->  already exists
        |
        v
[1] Time-series detection + axis selection      forecast.py
        |
        v
[2] Temporal preparation (sort, regularize, time-split)
        |
        v
[3] Deterministic forecast core                 forecast.py
        |   FLAML task="ts_forecast"
        |   backtesting (rolling-origin)
        |   naive/seasonal baseline for comparison
        |   prediction intervals from backtest residuals
        v
    ForecastResult  (structured, like EngineResult)
        |
        +--> [4] Visuals            app.py (Forecast tab)
        +--> [5] Report             forecast_report.py  (LLM narrates, validator checks)
        |
        v
    Structured events (existing EventLog)
```

---

## 2. Inputs

| input | source | notes |
|---|---|---|
| dataframe | uploaded CSV | same df the rest of the app uses |
| **time column (X)** | auto-detected, user confirms | a real datetime, or a parseable string, or an ordered integer index |
| **target (Y)** | user selection | the quantity to forecast; must be numeric |
| frequency | inferred (`D`/`W`/`M`/`Q`/`Y`) | user can override |
| horizon `h` | user input | how many periods ahead; default = 10% of history, capped |
| group column (optional) | user | forecast per store / per SKU (phase 2, not v1) |

### Time-axis detection rules (deterministic, no LLM)
1. Any column already `datetime64` -> candidate.
2. Any text column that `pd.to_datetime` parses at >=95% success -> candidate.
3. A monotonically increasing integer column -> candidate ordered index.
4. If several candidates: rank by parse rate and regularity, present the top one, let the user
   switch. If none: **stop with an honest "no time axis found" — do not forecast.**

> Caution carried from the coercion ticket: `_coerce_numeric_like` can turn a date-like string
> into a number before detection runs. The forecast path must inspect the ORIGINAL column for
> datetime parsing before any numeric coercion is considered.

---

## 3. Outputs — `ForecastResult` (structured, deterministic)

```
ForecastResult
  success            : bool
  time_col, target   : str
  frequency          : str
  horizon            : int
  history            : [{ds, y}]                 the observed series (regularized)
  forecast           : [{ds, yhat, lower, upper}]  point + interval per future period
  backtest           : {
      folds          : [{train_end, mae, mape, rmse}]
      overall        : {mae, mape, rmse, coverage}   coverage = % of held-out inside the band
  }
  baseline           : {method: "naive"|"seasonal_naive", mae, mape}  for honest comparison
  model              : str        which estimator FLAML chose
  quality_gate       : {passed: bool, reasons: [...]}   e.g. worse than naive -> reject
  confidence         : {score, label, breakdown}        reuse the confidence idea, time-adapted
  events             : [...]      structured events
  run_log            : [...]      human-readable
```

**Prediction intervals** are derived from the empirical distribution of backtest residuals, not
from a model assumption. If backtest coverage of the 80% band is far from 80%, that is reported
as a warning rather than hidden.

**Quality gate for forecasting** (distinct from the regression gate):
- reject if the model does not beat the naive baseline on backtest MAE;
- warn if interval coverage deviates strongly from nominal;
- warn if history is too short for the requested horizon (e.g. `h > len(history)/3`).

---

## 4. Visuals (the part the current implementation completely lacks)

All rendered in the Forecast tab. Four charts, each tied to a number in `ForecastResult`:

1. **History + forecast** — the observed line, then the point forecast, with the shaded
   prediction band. One clear vertical marker at "today".
2. **Backtest fit** — for the held-out window, actual vs predicted overlaid. This is what earns
   trust: it shows the model on data it did not train on.
3. **Residual check** — backtest residuals over time; flags drift or widening error.
4. **Model vs baseline** — a small bar of forecast MAE vs naive MAE, so the user sees whether
   the model is actually adding value.

Design rules: no chart shows a number that is not in `ForecastResult`; the band is always drawn
(never a bare line implying false certainty); the "today" boundary between history and forecast
is always explicit.

---

## 5. Report design

A structured report, not a paragraph. Sections, each populated from `ForecastResult`:

1. **Summary** — target, horizon, frequency, the headline forecast (e.g. next-period value
   with its interval).
2. **Reliability** — backtest MAE/MAPE, band coverage, comparison to baseline, confidence label.
3. **Risks & caveats** — short history, wide intervals, detected drift, gaps that were filled.
4. **Narrative** — the ONLY LLM-written section. It explains what the numbers mean in plain
   language. It receives the structured result as input and is forbidden to introduce figures.

### Number-grounding validator (mandatory)
After the LLM writes the narrative, a validator extracts every numeric token from the text and
checks each against the values in `ForecastResult` (within rounding tolerance). Any number not
found is flagged, and the report marks the narrative as "unverified" rather than shipping a
hallucinated figure. This is the concrete mechanism behind principle 0.

Report is exportable (markdown first; docx/pdf later if wanted).

---

## 6. Explicitly OUT of scope for v1

- Per-group forecasting (per store / SKU) — phase 2.
- Exogenous regressors (weather, promotions) — phase 2.
- Multivariate / hierarchical forecasting.
- Real-time / streaming updates.
- Automatic anomaly detection on the history.

---

## 7. Open questions to resolve BEFORE coding

1. **Does FLAML `ts_forecast` meet the need, or do we add Prophet/statsmodels?** Needs a spike
   on one real series. FLAML keeps the "one engine" story; Prophet gives intervals and
   seasonality for free. Decide by measuring, not by preference.
2. **Interval method:** empirical backtest residuals (model-agnostic, preferred) vs a
   distributional assumption. Leaning empirical.
3. **How short is too short?** A minimum history length below which we refuse to forecast.
4. **Confidence for forecasts:** reuse the tabular `ConfidenceEngine` idea, but its inputs
   (CV std, row count) do not map cleanly to time series. Likely a new, small time-aware score.
5. **Coercion guard:** ensure date detection runs on the original column, ahead of any numeric
   coercion (ties into the open Numeric Coercion False Positives ticket).

---

## 8. Build order (when we start — not now)

```
[1] time-axis detection + ForecastResult skeleton  (no model yet, returns history only)
[2] naive baseline + backtest harness              (measurable, honest, no ML)
[3] FLAML ts_forecast plugged into the harness
[4] prediction intervals from residuals
   --- ForecastResult is now COMPLETE: every number exists ---
[5] visuals in the Forecast tab                    (read ForecastResult only)
[6] structured report                              (read ForecastResult only, no LLM)
   --- everything works and is shippable WITHOUT any LLM ---
[7] LLM narrative + number-grounding validator     (LAST; consumes the finished result)
```

The order is deliberate and matches principle 0.0: steps [1]-[6] produce a fully working,
fully honest forecast with charts and a report and NO LLM anywhere. Step [7] is added last and
is optional at runtime — if it is removed or the LLM is unavailable, everything above still
works. Step [2] gives a working naive forecast before any model exists, so the tab never shows
fabricated numbers at any point during the build.
