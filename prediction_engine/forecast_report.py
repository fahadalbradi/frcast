"""
forecast_report.py — Structured Forecast Report (Step 6)
========================================================
Builds a STRUCTURED report from a ForecastResult. No LLM here — this is the deterministic
report that must work with no API key at all (ordering principle 0.0). The LLM narrative is a
SEPARATE, later step (7) that will consume this same result; it is not imported or called here.

Every figure in the report is read from ForecastResult (decision #5). Nothing is computed or
invented. Sections:
    summary · reliability · coverage · risks · quality_gate

`build_report()` returns a dict (for the UI / JSON export).
`report_to_markdown()` renders that dict as text (for download). Both are pure formatting.
"""
from __future__ import annotations


def build_report(result) -> dict:
    """Assemble the structured report dict from a ForecastResult. Read-only."""
    if result is None:
        return {"available": False, "reason": "No forecast result."}

    fc = result.forecast or []
    naive = (result.baseline or {}).get("naive", {}) or {}
    model_bt = (result.backtest or {}).get("model", {}) or {}
    coverage = (result.backtest or {}).get("interval_coverage")
    gate = result.quality_gate or {}
    conf = result.confidence or {}

    # ---- summary ----
    headline = None
    if fc:
        first = fc[0]
        headline = {
            "period": _fmt_ds(first.get("ds")),
            "yhat": _r(first.get("yhat")),
            "lower": _r(first.get("lower")),
            "upper": _r(first.get("upper")),
        }
    summary = {
        "target": result.target,
        "time_col": result.time_col,
        "frequency": result.frequency,
        "horizon": result.horizon,
        "n_history": len([h for h in (result.history or []) if h.get("y") is not None]),
        "model": result.model,
        "next_period_forecast": headline,
    }

    # ---- reliability ----
    naive_mae = naive.get("mae")
    model_mae = model_bt.get("mae")
    improvement = None
    if naive_mae not in (None, 0) and model_mae is not None:
        improvement = round((naive_mae - model_mae) / naive_mae * 100, 2)
    reliability = {
        "model_mae": _r(model_mae),
        "model_rmse": _r(model_bt.get("rmse")),
        "model_mape": _r(model_bt.get("mape")),
        "naive_mae": _r(naive_mae),
        "improvement_vs_naive_pct": improvement,
        "beats_naive": (model_mae is not None and naive_mae is not None and model_mae < naive_mae),
        "confidence": {"score": conf.get("score"), "label": conf.get("label"),
                       "breakdown": conf.get("breakdown", {})},
    }

    # ---- coverage ----
    coverage_section = {
        "empirical_interval_coverage": _r(coverage) if coverage is not None else None,
        "note": ("Share of held-out backtest points that fell inside the empirical interval. "
                 "Close to the nominal level means the band is well-calibrated."),
    }

    # ---- risks / caveats (read from warnings + structural signals) ----
    risks = list(result.warnings or [])
    n_hist = summary["n_history"]
    if n_hist and result.horizon and result.horizon > n_hist / 3:
        risks.append(f"Horizon ({result.horizon}) is large relative to history "
                     f"({n_hist} points); far-ahead forecasts are less reliable.")
    if coverage is not None and abs(coverage - 0.80) > 0.15:
        risks.append(f"Interval coverage ({coverage:.0%}) is far from nominal (80%); "
                     "bands may be mis-calibrated.")
    if not risks:
        risks.append("No structural risks flagged.")

    # ---- quality gate ----
    quality_gate = {
        "passed": gate.get("passed"),
        "reasons": gate.get("reasons", []),
        "warnings": gate.get("warnings", []),
    }

    return {
        "available": True,
        "success": result.success,
        "summary": summary,
        "reliability": reliability,
        "coverage": coverage_section,
        "risks": risks,
        "quality_gate": quality_gate,
    }


def report_to_markdown(report: dict) -> str:
    """Render the structured report as markdown. Pure formatting, no new numbers."""
    if not report.get("available"):
        return f"# Forecast Report\n\n_{report.get('reason', 'No result.')}_"

    s = report["summary"]
    r = report["reliability"]
    cov = report["coverage"]
    gate = report["quality_gate"]

    lines = ["# Forecast Report", ""]

    status = "PASSED" if report.get("success") else "REJECTED"
    lines += [f"**Status:** {status}", ""]

    # summary
    lines += ["## Summary", ""]
    lines += [f"- Target: **{s['target']}**  ·  time axis: `{s['time_col']}`  ·  "
              f"frequency: `{s['frequency']}`",
              f"- History: {s['n_history']} periods  ·  horizon: {s['horizon']}  ·  "
              f"model: `{s['model']}`"]
    hp = s.get("next_period_forecast")
    if hp:
        lines += [f"- Next period (**{hp['period']}**): **{hp['yhat']}** "
                  f"(interval {hp['lower']} – {hp['upper']})"]
    lines += [""]

    # reliability
    lines += ["## Reliability", ""]
    lines += [f"- Model MAE: **{r['model_mae']}**  ·  RMSE: {r['model_rmse']}  ·  "
              f"MAPE: {r['model_mape']}",
              f"- Naive baseline MAE: {r['naive_mae']}",
              f"- Improvement vs naive: **{r['improvement_vs_naive_pct']}%**  ·  "
              f"beats naive: **{r['beats_naive']}**"]
    c = r.get("confidence", {})
    if c.get("score") is not None:
        lines += [f"- Forecast confidence: **{c['score']} ({c['label']})**"]
    lines += [""]

    # coverage
    lines += ["## Interval Coverage", ""]
    ec = cov.get("empirical_interval_coverage")
    lines += [f"- Empirical coverage: **{_pct(ec)}**", f"- {cov['note']}", ""]

    # risks
    lines += ["## Risks & Caveats", ""]
    lines += [f"- {risk}" for risk in report["risks"]]
    lines += [""]

    # quality gate
    lines += ["## Quality Gate", ""]
    lines += [f"- Passed: **{gate['passed']}**"]
    for reason in gate.get("reasons", []):
        lines += [f"- Rejected because: {reason}"]
    for w in gate.get("warnings", []):
        lines += [f"- Warning: {w}"]
    lines += [""]

    lines += ["_All figures above are read directly from the forecast result; "
              "no value is generated._"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def _r(v, nd=4):
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def _pct(v):
    return "n/a" if v is None else f"{float(v)*100:.0f}%"


def _fmt_ds(ds):
    try:
        return ds.strftime("%Y-%m-%d")
    except AttributeError:
        return str(ds)
