"""
llm_agent.py
============
The DSPy reasoning layer referenced in the charter, used ONLY for the
analytical "LLM Forecast" mode (never for the numeric Predict path — numbers
always come from the deterministic FLAML/sklearn model, per the charter's
"Code Interpreter instead of LLM-generated numbers" principle).

Requires an ANTHROPIC_API_KEY environment variable when run locally.
If it is missing, the module fails gracefully and the rest of the engine
(Profiling/Preprocessing/Training/Evaluation/Predict) keeps working normally.
"""
from __future__ import annotations
import os

try:
    import dspy
    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False


class ForecastSignature(dspy.Signature if _DSPY_AVAILABLE else object):
    """You are an expert data analyst. Based on the data fingerprint and evaluation results, 
    provide a concise analytical explanation (LLM Forecast) covering: key influential factors, 
    result reliability, and risks/caveats. Do not invent numbers that are not in the input data."""
    data_fingerprint: str = dspy.InputField(desc="JSON summary of data fingerprint")
    evaluation_summary: str = dspy.InputField(desc="JSON summary of triple-threat evaluation results")
    confidence: str = dspy.InputField(desc="Confidence score and interpretation")
    analytical_forecast: str = dspy.OutputField(desc="Professional analytical report, 4-6 sentences")


class LLMForecastAgent:
    """ReAct-orchestrated analytical agent (dspy.Predict is sufficient here since
    this is a single reasoning hop, not a multi-tool task)."""

    def __init__(self, model: str = "anthropic/claude-sonnet-4-6"):
        self.available = False
        self.error = None

        if not _DSPY_AVAILABLE:
            self.error = "The dspy library is not installed."
            return

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self.error = "ANTHROPIC_API_KEY environment variable not found — add it to enable LLM Forecast mode."
            return

        try:
            lm = dspy.LM(model, api_key=api_key, max_tokens=600)
            dspy.configure(lm=lm)
            self.predictor = dspy.Predict(ForecastSignature)
            self.available = True
        except Exception as e:
            self.error = f"Failed to initialize language model: {e}"

    def forecast(self, fingerprint_dict: dict, evaluation_dict: dict, confidence_dict: dict) -> str:
        if not self.available:
            return f"[LLM Forecast unavailable] {self.error}"
        try:
            result = self.predictor(
                data_fingerprint=str(fingerprint_dict),
                evaluation_summary=str(evaluation_dict),
                confidence=str(confidence_dict),
            )
            return result.analytical_forecast
        except Exception as e:
            return f"[Error during LLM Forecast generation] {e}"