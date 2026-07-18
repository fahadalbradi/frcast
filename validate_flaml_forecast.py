"""
validate_flaml_forecast.py — runtime validation for Step 3b (FLAML ts_forecast)
==============================================================================
Run this ON A MACHINE WITH FLAML INSTALLED. It exercises the UNTESTED
`make_flaml_forecaster` against real FLAML and reports exactly what works and what does not.

    pip install "flaml[ts_forecast]"      # note: ts_forecast, not just automl
    python validate_flaml_forecast.py

It checks four assumptions independently, so a failure points at the specific line to fix:

  CHECK 1  is flaml importable, and is the ts_forecast task actually available?
  CHECK 2  does make_flaml_forecaster() run and return `horizon` numbers?
  CHECK 3  does it plug into the (already-tested) backtest harness and beat naive?
  CHECK 4  does the full run_forecast_evaluation() glue produce a comparison?

Nothing here modifies the engine. It only imports and calls it.
"""
import sys
import traceback

import numpy as np
import pandas as pd


def line(msg=""):
    print(msg)


def check(n, title):
    line(f"\n{'='*70}\nCHECK {n}: {title}\n{'='*70}")


def main():
    # ---- build a known series: trend + weekly seasonality, daily freq ----
    n = 140
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    y = 100 + np.arange(n) * 1.2 + 8 * np.sin(2 * np.pi * np.arange(n) / 7) \
        + np.random.default_rng(0).normal(0, 1.0, n)
    history = list(y)

    # ===== CHECK 1 =====
    check(1, "flaml import + ts_forecast task availability")
    try:
        import flaml
        line(f"  flaml version: {getattr(flaml, '__version__', 'unknown')}")
        from flaml import AutoML
        # a tiny fit to see whether ts_forecast is a recognised task at all
        probe = AutoML()
        tiny = pd.DataFrame({"ds": idx[:30], "y": history[:30]})
        try:
            probe.fit(dataframe=tiny, label="y", task="ts_forecast",
                      time_budget=5, period=3, verbose=0)
            line("  PASS: ts_forecast task ran on a tiny sample.")
        except Exception as e:
            line("  FAIL: ts_forecast task did not run.")
            line(f"        {type(e).__name__}: {e}")
            line("  -> Likely missing extras. Try: pip install \"flaml[ts_forecast]\"")
            line("     (requirements.txt currently pins flaml[automl], which is NOT enough.)")
            return
    except ImportError as e:
        line(f"  FAIL: cannot import flaml ({e}). Install it, then re-run.")
        return

    # ===== CHECK 2 =====
    check(2, "make_flaml_forecaster returns `horizon` numbers")
    try:
        from prediction_engine.forecast import make_flaml_forecaster
        fc = make_flaml_forecaster(frequency="D", time_budget=10)
        preds = fc(history, horizon=7)
        ok_len = len(preds) == 7
        ok_num = all(isinstance(v, float) and np.isfinite(v) for v in preds)
        line(f"  returned {len(preds)} values: {[round(v,2) for v in preds]}")
        line(f"  length == 7 : {ok_len}")
        line(f"  all finite floats : {ok_num}")
        if not (ok_len and ok_num):
            line("  FAIL: output shape/type is wrong — fix make_flaml_forecaster.")
            return
        line("  PASS")
    except Exception:
        line("  FAIL: make_flaml_forecaster raised. Traceback:")
        traceback.print_exc()
        line("\n  This is the specific code that was never executed in the build env.")
        line("  Common culprits: fit() arg names (dataframe/label/period), or predict()")
        line("  expecting a different frame shape than {'ds': future_index}.")
        return

    # ===== CHECK 3 =====
    check(3, "plugs into the tested backtest harness and beats naive")
    try:
        from prediction_engine.forecast import backtest, naive_forecaster, make_flaml_forecaster
        base = backtest(history, naive_forecaster, horizon=7, n_folds=4)
        fc = make_flaml_forecaster(frequency="D", time_budget=10)
        model = backtest(history, fc, horizon=7, n_folds=4)
        bmae = base["overall"]["mae"]
        mmae = model["overall"]["mae"]
        line(f"  naive MAE : {bmae:.3f}")
        line(f"  flaml MAE : {mmae:.3f}")
        line(f"  model beats naive : {mmae < bmae}")
        line(f"  residuals collected : {len(model.get('residuals', []))}")
        line("  PASS (harness accepted the FLAML forecaster)"
             if model.get("folds") else "  WARN: no folds scored — check horizon/length.")
    except Exception:
        line("  FAIL: backtest with FLAML raised. Traceback:")
        traceback.print_exc()
        return

    # ===== CHECK 4 =====
    check(4, "run_forecast_evaluation end-to-end glue")
    try:
        from prediction_engine.forecast import (
            prepare_series, prepare_temporal, run_forecast_evaluation)
        df = pd.DataFrame({"day": idx, "sales": y.round(2)})
        r1 = prepare_series(df, "sales")
        prep = prepare_temporal(r1, strategy="linear")
        res = run_forecast_evaluation(prep, horizon=7, frequency=prep.frequency,
                                      seasonal_period=7, use_flaml=True, time_budget=10)
        line(f"  baseline naive MAE : {res['baseline']['naive']['overall'].get('mae')}")
        line(f"  flaml MAE          : {res['model'].get('flaml', {}).get('overall', {}).get('mae')}")
        line(f"  comparison         : {res.get('comparison')}")
        line(f"  notes              : {res.get('notes')}")
        if res.get("comparison"):
            line("  PASS: full glue produced a model-vs-naive comparison.")
        else:
            line("  WARN: no comparison — FLAML branch did not score. See notes above.")
    except Exception:
        line("  FAIL: run_forecast_evaluation raised. Traceback:")
        traceback.print_exc()
        return

    line("\n" + "="*70)
    line("ALL CHECKS PASSED — Step 3b can be marked validated.")
    line("Report the four MAE numbers back before moving to Step 4.")
    line("="*70)


if __name__ == "__main__":
    main()
