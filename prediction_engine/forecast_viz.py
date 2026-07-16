"""
forecast_viz.py — Forecast Visuals (Step 5)
===========================================
Builds Plotly figures from a ForecastResult. It READS the result and nothing else:
every point plotted is already a number in ForecastResult (decision #5). It computes no
metric, fits no model, calls no LLM.

Four figures, each tied to specific fields:
  1. history_vs_forecast  -> result.history + result.forecast (yhat, lower, upper)
  2. confidence_band       -> the same band, isolated, with the "today" boundary
  3. backtest_chart        -> result.backtest["folds"]  (per-fold MAE)
  4. model_vs_naive        -> result.baseline["naive"] vs result.backtest["model"]

Each function returns a plotly Figure. If the relevant fields are missing (e.g. a rejected
forecast), it returns a figure with an explicit "no data" annotation rather than inventing any.
"""
from __future__ import annotations

import plotly.graph_objects as go


_HIST = "#2563eb"
_FCST = "#dc2626"
_BAND = "rgba(220,38,38,0.15)"
_NAIVE = "#9ca3af"


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False,
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(height=320, xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def history_vs_forecast(result) -> go.Figure:
    """Observed series, then the point forecast with its shaded interval, split by a 'today'
    marker. Reads result.history and result.forecast only."""
    hist = [h for h in (result.history or []) if h.get("y") is not None]
    fc = result.forecast or []
    if not hist:
        return _empty("No history to plot.")

    fig = go.Figure()

    hx = [h["ds"] for h in hist]
    hy = [h["y"] for h in hist]
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines", name="History",
                             line={"color": _HIST}))

    if fc:
        fx = [f["ds"] for f in fc]
        fy = [f["yhat"] for f in fc]
        lo = [f["lower"] for f in fc]
        hi = [f["upper"] for f in fc]

        # connect the last observed point to the first forecast point
        fx = [hx[-1]] + fx
        fy = [hy[-1]] + fy
        lo = [hy[-1]] + lo
        hi = [hy[-1]] + hi

        fig.add_trace(go.Scatter(x=fx + fx[::-1], y=hi + lo[::-1], fill="toself",
                                 fillcolor=_BAND, line={"color": "rgba(0,0,0,0)"},
                                 name="Interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines+markers", name="Forecast",
                                 line={"color": _FCST, "dash": "dash"}))
        # 'today' boundary
        fig.add_vline(x=hx[-1], line_width=1, line_dash="dot", line_color="#6b7280")

    fig.update_layout(height=380, title="History vs Forecast",
                      legend={"orientation": "h"}, margin={"t": 40})
    return fig


def confidence_band(result, level_label: str = "") -> go.Figure:
    """The forecast interval on its own, so the band width over the horizon is clearly visible.
    Reads result.forecast only."""
    fc = result.forecast or []
    if not fc:
        return _empty("No forecast interval (forecast was not produced or was rejected).")

    fx = [f["ds"] for f in fc]
    fy = [f["yhat"] for f in fc]
    lo = [f["lower"] for f in fc]
    hi = [f["upper"] for f in fc]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fx + fx[::-1], y=hi + lo[::-1], fill="toself",
                             fillcolor=_BAND, line={"color": "rgba(0,0,0,0)"},
                             name="Interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines+markers", name="Point forecast",
                             line={"color": _FCST}))
    title = "Forecast Confidence Band"
    cov = (result.backtest or {}).get("interval_coverage")
    if cov is not None:
        title += f"  ·  empirical coverage {cov:.0%}"
    fig.update_layout(height=340, title=title, legend={"orientation": "h"}, margin={"t": 40})
    return fig


def backtest_chart(result) -> go.Figure:
    """Per-fold backtest MAE — how the model did on data it did not train on.
    Reads result.backtest['folds']."""
    folds = (result.backtest or {}).get("folds", [])
    if not folds:
        return _empty("No backtest folds available.")

    labels = [f"train_end={f['train_end']}" for f in folds]
    maes = [f["mae"] for f in folds]
    fig = go.Figure(go.Bar(x=labels, y=maes, marker_color=_HIST, name="MAE"))
    overall = (result.backtest or {}).get("model", {}).get("mae")
    if overall is not None:
        fig.add_hline(y=overall, line_dash="dash", line_color=_FCST,
                      annotation_text=f"overall MAE {overall:.3g}")
    fig.update_layout(height=320, title="Backtest MAE per Fold (out-of-sample)",
                      margin={"t": 40})
    return fig


def model_vs_naive(result) -> go.Figure:
    """Model MAE vs the mandatory naive baseline — is the model actually adding value?
    Reads result.baseline['naive'] and result.backtest['model']."""
    naive = (result.baseline or {}).get("naive", {}).get("mae")
    model = (result.backtest or {}).get("model", {}).get("mae")
    if naive is None or model is None:
        return _empty("Baseline or model MAE missing.")

    colors = [_NAIVE, _FCST if model < naive else _NAIVE]
    fig = go.Figure(go.Bar(x=["Naive baseline", "Model"], y=[naive, model],
                           marker_color=colors, text=[f"{naive:.3g}", f"{model:.3g}"],
                           textposition="outside"))
    verdict = "model beats naive" if model < naive else "model does NOT beat naive"
    fig.update_layout(height=320, title=f"Model vs Naive (MAE)  ·  {verdict}",
                      margin={"t": 40})
    return fig


def all_figures(result) -> dict:
    """Convenience: every figure keyed by name, for the UI to drop into tabs/columns."""
    return {
        "history_vs_forecast": history_vs_forecast(result),
        "confidence_band": confidence_band(result),
        "backtest_chart": backtest_chart(result),
        "model_vs_naive": model_vs_naive(result),
    }
