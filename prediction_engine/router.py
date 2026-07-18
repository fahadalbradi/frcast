"""
router.py
=========
ReAct Router — v1 (routing only, no reasoning loop).

Sits ON TOP of the deterministic core. It decides WHICH tool to call. It never touches
preprocessing, encoding, imputation, FLAML or evaluation — those stay exactly as frozen
in Benchmark-Stable-v1.

    user message ──► IntentDetector ──► Intent ──► Tool ──► ToolResult
                          │                                     │
                          └────────── EventLog ◄────────────────┘

Intent detection order:
    1. explicit_intent passed by the caller (a UI button) — always wins, no LLM call
    2. LLM classifier (OpenAI) — only if OPENAI_API_KEY is set AND `openai` is installed
    3. keyword rules — always available, always the fallback

The LLM is optional by design. If it is missing, misconfigured or fails, the router
degrades to rules and records WHY in the event log. It never raises.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import pandas as pd

from .events import EventLog
from .orchestrator import PredictionEngine, EngineResult


class Intent(str, Enum):
    PREDICTION = "prediction"
    FORECAST = "forecast"
    EXPLANATION = "explanation"
    UNKNOWN = "unknown"


@dataclass
class ToolResult:
    tool: str
    status: str                       # "ok" | "error" | "not_implemented"
    output: Any = None
    message: str = ""
    events: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Intent detection
# --------------------------------------------------------------------------- #

_RULES: dict[Intent, list[str]] = {
    Intent.FORECAST: [
        r"\bforecast\b", r"\btime series\b", r"\btrend\b", r"\bover time\b",
        r"\bnext (month|year|week|quarter|day|\d+)\b", r"\bfuture\b", r"\bprojection\b",
        r"\bcoming (months?|years?|weeks?)\b",
        # Arabic: the verb "توقع" alone is AMBIGUOUS (it means both predict and forecast),
        # so it must never appear here on its own. Only temporal markers qualify.
        "فوركاست", "سلسلة زمنية", "مستقبلي", "المستقبل",
        "القادم", "القادمة", "الجاي", "الأشهر", "الاشهر",
        r"اتجاه",
    ],
    Intent.EXPLANATION: [
        r"\bwhy\b", r"\bexplain\b", r"\bhow does\b", r"\breport\b", r"\bsummar",
        r"\bwhat (drives|affects|influences)\b", r"\bimportant features?\b",
        r"\binterpret\b", r"\banaly[sz]e\b", r"\bwhich (factors?|features?)\b",
        "اشرح", "لماذا", "ليش", "وش يأثر", "تقرير", "تحليل", "تفسير",
        "أهم العوامل", "اهم العوامل", "ملخص",
    ],
    Intent.PREDICTION: [
        r"\bpredict\b", r"\bestimate\b", r"\bhow much\b", r"\bclassif",
        # covers "what is / what's / whats the price|value|cost"
        r"\bwhat'?s? (is |will be )?the (price|value|cost)\b",
        r"\b(price|value|cost) of\b", r"\bwill (it|this|they|he|she)\b", r"\bscore\b",
        "توقع", "تنبؤ", "كم سعر", "كم راح", "احسب", "قدر", "بريديكت", "تصنيف", "سعر",
    ],
}

# forecast keywords are checked FIRST: "predict the next 6 months" is a forecast,
# not a row-level prediction, even though it contains the word "predict".
_PRIORITY = [Intent.FORECAST, Intent.EXPLANATION, Intent.PREDICTION]

_LLM_SYSTEM = (
    "You are an intent classifier for a tabular ML tool. "
    "Classify the user's message into exactly one of: prediction, forecast, explanation.\n"
    "- prediction: they want a value/class for ONE new record (a house's price, will a "
    "customer churn).\n"
    "- forecast: they want FUTURE values over time (next quarter's sales, the trend).\n"
    "- explanation: they want to understand results, drivers, or a report.\n"
    'Answer with JSON only: {"intent": "...", "reason": "..."} — no other text.'
)


class IntentDetector:
    """Rules always work. The LLM is an optional upgrade, never a dependency."""

    def __init__(self, model: str = "gpt-4o-mini", use_llm: bool = True):
        self.model = model
        self.use_llm = use_llm

    # -- rules ------------------------------------------------------------- #
    def _by_rules(self, message: str) -> tuple[Intent, str]:
        text = (message or "").lower()
        for intent in _PRIORITY:
            for pat in _RULES[intent]:
                if re.search(pat, text):
                    return intent, f"keyword rule matched: {pat!r}"
        return Intent.UNKNOWN, "no keyword rule matched"

    # -- llm --------------------------------------------------------------- #
    def _by_llm(self, message: str) -> tuple[Intent, str] | None:
        """Returns None when the LLM is unavailable or unusable — never raises."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI          # lazy: absence must not break the package
        except ImportError:
            return None

        try:
            import json
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user", "content": message},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content)
            intent = Intent(parsed["intent"].strip().lower())
            return intent, parsed.get("reason", "classified by LLM")
        except Exception:
            return None

    # -- public ------------------------------------------------------------ #
    def detect(self, message: str, log: EventLog) -> Intent:
        if self.use_llm:
            llm = self._by_llm(message)
            if llm is not None:
                intent, reason = llm
                log.add("intent_detection", f"{intent.value}_selected",
                        reason, {"source": "llm", "model": self.model, "message": message})
                return intent
            log.add("intent_detection", "llm_unavailable",
                    "no OPENAI_API_KEY, openai not installed, or the call failed — "
                    "falling back to keyword rules",
                    {"source": "llm"}, status="skipped")

        intent, reason = self._by_rules(message)
        status = "ok" if intent is not Intent.UNKNOWN else "error"
        log.add("intent_detection", f"{intent.value}_selected", reason,
                {"source": "rules", "message": message}, status=status)
        return intent


# --------------------------------------------------------------------------- #
# Tools — thin wrappers. They CALL the core; they never modify it.
# --------------------------------------------------------------------------- #

class PredictionTool:
    """Delegates to the frozen PredictionEngine. Zero new ML logic."""

    name = "prediction"

    def __init__(self, engine: PredictionEngine | None = None):
        self.engine = engine or PredictionEngine()

    def run(self, df: pd.DataFrame, target_col: str, log: EventLog,
            new_rows: pd.DataFrame | None = None,
            engine_result: EngineResult | None = None) -> ToolResult:

        # Reuse an already-trained model when the caller has one (the Predict tab does).
        # Retraining on every button press would be wasteful and would change the numbers.
        if engine_result is not None and engine_result.success:
            log.add("tool_execution", "reusing_trained_model",
                    "a trained model already exists for this dataset — not retraining",
                    {"best_estimator": engine_result.training.best_estimator})
            result = engine_result
        else:
            log.add("tool_execution", "prediction_pipeline_invoked",
                    "routing to the existing deterministic pipeline",
                    {"rows": len(df), "target_col": target_col})
            result = self.engine.run(df, target_col=target_col)

        if not result.success:
            log.add("tool_execution", "prediction_pipeline_failed", result.error or "",
                    {"engine_run_log": result.run_log}, status="error")
            # Same output shape as the success path, so callers can always read
            # output["engine_result"] regardless of status (contract: output is a dict).
            return ToolResult(self.name, "error",
                              {"engine_result": result, "predictions": None},
                              result.error or "pipeline failed")

        log.add("tool_execution", "prediction_pipeline_completed",
                f"best estimator: {result.training.best_estimator}",
                {"task_type": result.fingerprint.task_type,
                 "statistical": result.evaluation.statistical,
                 "stability": {k: v for k, v in result.evaluation.stability.items()
                               if k in ("cv_mean", "cv_std", "n_splits")},
                 "passed_quality_gate": result.evaluation.passed_quality_gate,
                 "confidence": result.confidence.score,
                 "attempts": result.attempts,
                 "engine_run_log": result.run_log})

        predictions = None
        if new_rows is not None:
            try:
                predictions = self.engine.predict(result, new_rows)
                log.add("tool_execution", "prediction_computed", "",
                        {"n_rows": len(new_rows), "predictions": predictions})
            except Exception as e:
                log.add("tool_execution", "prediction_failed", str(e), status="error")

        return ToolResult(self.name, "ok",
                          {"engine_result": result, "predictions": predictions},
                          "prediction pipeline completed")


class ForecastTool:
    """Runs the full forecast pipeline (Steps 1-6) and returns a ForecastResult plus a
    structured report. Output is a dict (contract: always a dict), so a rejected or failed
    forecast never crashes the caller.

    Gap strategy is REQUIRED (no default, per the Step-2 decision): the caller must pass
    `strategy` in kwargs. Time axis, horizon and seasonal period are optional and auto-detected
    when omitted."""

    name = "forecast"

    def run(self, df: pd.DataFrame, target_col: str, log: EventLog,
            strategy: str | None = None, time_col: str | None = None,
            horizon: int | None = None, seasonal_period: int | None = None,
            frequency: str | None = None, use_flaml: bool = True,
            level: float = 0.80, time_budget: int = 30, **kwargs) -> ToolResult:
        # lazy import: forecast pipeline depends on plotly/flaml only at call time
        from .forecast import (prepare_series, prepare_temporal,
                               run_forecast_evaluation, assemble_forecast_result)
        from .forecast_report import build_report
        import pandas as _pd

        if strategy is None:
            log.add("tool_execution", "forecast_needs_gap_strategy",
                    "gap strategy is required and has no default — ask the user",
                    {"choices": ["none", "ffill", "linear", "zero", "mean"]},
                    status="error")
            return ToolResult(self.name, "error",
                              {"forecast_result": None, "report": None},
                              "A gap-handling strategy must be chosen before forecasting "
                              "(none / ffill / linear / zero / mean).")

        # Step 1 — detect time axis + build history
        log.add("tool_execution", "forecast_step1_detection", "detecting time axis",
                {"target_col": target_col, "time_col": time_col})
        r1 = prepare_series(df, target_col, time_col=time_col,
                            frequency=frequency, horizon=horizon)
        if not r1.success:
            log.add("tool_execution", "forecast_step1_failed", r1.error or "", status="error")
            return ToolResult(self.name, "error",
                              {"forecast_result": r1, "report": build_report(r1)}, r1.error or "")

        # Step 2 — temporal preparation with the explicit strategy
        log.add("tool_execution", "forecast_step2_temporal",
                f"regularizing on '{r1.frequency}' grid, gap strategy='{strategy}'",
                {"strategy": strategy})
        prep = prepare_temporal(r1, strategy=strategy)
        if not prep.success:
            log.add("tool_execution", "forecast_step2_failed", prep.error or "", status="error")
            return ToolResult(self.name, "error",
                              {"forecast_result": None, "report": None}, prep.error or "")

        # Steps 3+4 — backtest (naive + FLAML), then build the point forecast
        h = r1.horizon
        log.add("tool_execution", "forecast_step3_backtest",
                f"backtesting naive baseline + {'FLAML' if use_flaml else 'no'} model",
                {"horizon": h, "use_flaml": use_flaml})
        evaluation = run_forecast_evaluation(prep, horizon=h, frequency=prep.frequency,
                                             seasonal_period=seasonal_period,
                                             use_flaml=use_flaml, time_budget=time_budget)
        for note in evaluation.get("notes", []):
            log.add("tool_execution", "forecast_note", note, status="skipped")

        # point forecast for the future horizon (refit on the full history)
        values = [s["y"] for s in prep.series if s["y"] is not None]
        point = None
        if use_flaml:
            try:
                from .forecast import make_flaml_forecaster
                point = make_flaml_forecaster(prep.frequency, time_budget)(values, h)
            except Exception as e:
                log.add("tool_execution", "forecast_model_forecast_failed", str(e),
                        status="error")
        if point is None:                      # fall back to naive so the tab still shows a forecast
            from .forecast import naive_forecaster
            point = naive_forecaster(values, h)
            log.add("tool_execution", "forecast_point_fallback",
                    "model forecast unavailable — showing the naive baseline forecast",
                    status="skipped")

        # future timestamps
        last = prep.series[-1]["ds"]
        if isinstance(last, (int,)):
            future_ds = [last + i for i in range(1, h + 1)]
        else:
            future_ds = list(_pd.date_range(last, periods=h + 1, freq=prep.frequency)[1:])

        # Step 4 — assemble the ForecastResult
        result = assemble_forecast_result(prep, evaluation, point, future_ds, level=level)
        log.add("tool_execution", "forecast_completed",
                f"gate {'passed' if result.success else 'REJECTED'}",
                {"model_mae": (result.backtest or {}).get("model", {}).get("mae"),
                 "naive_mae": (result.baseline or {}).get("naive", {}).get("mae"),
                 "confidence": result.confidence.get("score")},
                status="ok" if result.success else "error")

        # Step 6 — structured report (Step 5 visuals are built in the UI from this result)
        report = build_report(result)
        return ToolResult(self.name, "ok" if result.success else "error",
                          {"forecast_result": result, "report": report},
                          "forecast completed" if result.success
                          else (result.error or "forecast rejected by quality gate"))


class ExplanationTool:
    """Builds a report from results the core ALREADY computed. Deterministic: every number
    it prints is read from the EngineResult, none is generated."""

    name = "explanation"

    def run(self, df: pd.DataFrame, target_col: str, log: EventLog,
            engine_result: EngineResult | None = None,
            engine: PredictionEngine | None = None) -> ToolResult:

        if engine_result is None:
            log.add("tool_execution", "explanation_needs_a_trained_model",
                    "no prior result supplied — running the pipeline first",
                    {"target_col": target_col})
            engine_result = (engine or PredictionEngine()).run(df, target_col=target_col)

        if not engine_result.success:
            log.add("tool_execution", "explanation_tool_failed",
                    engine_result.error or "", status="error")
            return ToolResult(self.name, "error", None, engine_result.error or "no model")

        fp, ev, conf = engine_result.fingerprint, engine_result.evaluation, engine_result.confidence
        report = {
            "dataset": {"rows": fp.n_rows, "cols": fp.n_cols,
                        "task_type": fp.task_type, "target": fp.target_col},
            "model": engine_result.training.best_estimator,
            "metrics": ev.statistical,
            "stability": {k: v for k, v in ev.stability.items()
                          if k in ("cv_mean", "cv_std", "n_splits")},
            "top_drivers": ev.explainability.get("top_features", [])[:5],
            "passed_quality_gate": ev.passed_quality_gate,
            "rejection_reasons": ev.rejection_reasons,
            "confidence": {"score": conf.score, "label": conf.label,
                           "breakdown": conf.breakdown},
            "recommendation": engine_result.recommendation,
        }

        log.add("tool_execution", "explanation_report_built",
                "every figure below was read from the trained model — none was generated",
                report)
        return ToolResult(self.name, "ok", report, "report built from the existing model")


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

class ReActRouter:
    """v1: detect intent -> pick tool -> call it -> record structured events.
    No reasoning loop, no self-correction, no plan revision. That comes later."""

    def __init__(self, engine: PredictionEngine | None = None,
                 detector: IntentDetector | None = None):
        self.engine = engine or PredictionEngine()
        self.detector = detector or IntentDetector()
        self.tools: dict[Intent, Any] = {
            Intent.PREDICTION: PredictionTool(self.engine),
            Intent.FORECAST: ForecastTool(),
            Intent.EXPLANATION: ExplanationTool(),
        }

    def route(self, message: str, df: pd.DataFrame, target_col: str,
              explicit_intent: Intent | str | None = None,
              new_rows: pd.DataFrame | None = None,
              engine_result: EngineResult | None = None,
              forecast_kwargs: dict | None = None) -> ToolResult:

        log = EventLog()
        log.add("session", "started", "", {"rows": len(df), "target_col": target_col,
                                           "message": message})

        # 1) intent
        if explicit_intent is not None:
            intent = Intent(explicit_intent)
            log.add("intent_detection", f"{intent.value}_selected",
                    "chosen explicitly by the user — no inference needed",
                    {"source": "explicit"})
        else:
            intent = self.detector.detect(message, log)

        # 2) routing
        if intent is Intent.UNKNOWN:
            log.add("tool_routing", "no_tool_selected",
                    "intent could not be determined — asking the user instead of guessing",
                    {"available": [i.value for i in self.tools]}, status="error")
            res = ToolResult("none", "error", None,
                             "Could not determine the intent. Please choose: "
                             "prediction, forecast, or explanation.")
            res.events = log.to_list()
            return res

        tool = self.tools[intent]
        log.add("tool_routing", f"{tool.name}_tool_selected",
                f"intent '{intent.value}' maps to the {tool.name} tool",
                {"tool": tool.name})

        # 3) execution
        if intent is Intent.PREDICTION:
            result = tool.run(df, target_col, log, new_rows=new_rows,
                              engine_result=engine_result)
        elif intent is Intent.EXPLANATION:
            result = tool.run(df, target_col, log,
                              engine_result=engine_result, engine=self.engine)
        else:
            result = tool.run(df, target_col, log, **(forecast_kwargs or {}))

        log.add("session", "finished", result.message, {"status": result.status})
        result.events = log.to_list()
        return result
