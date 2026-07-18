"""
forecast_narrative.py — Step 7: LLM narrative + number-grounding validator
==========================================================================
The LLM runs LAST (ordering principle 0.0) and ONLY narrates. This module:

  1. VALIDATOR (no LLM): extract every number from a text and check each against the numbers
     that actually exist in the ForecastResult. Any number not found is flagged. This is the
     safety mechanism; it is pure and fully testable without any API.

  2. NARRATIVE (OpenAI, optional): given a COMPLETED ForecastResult + its structured report,
     ask the model for a plain-language explanation, then run the validator on the output. If
     the LLM is unavailable, the pipeline still returns the structured report — only the prose
     is missing. If the narrative contains an ungrounded number, it is marked UNVERIFIED rather
     than shipped as fact.

Provider: OpenAI (decision recorded in the design doc). No Anthropic anywhere.
"""
from __future__ import annotations

import os
import re


# --------------------------------------------------------------------------- #
# 1) Number-grounding validator — NO LLM, pure, testable
# --------------------------------------------------------------------------- #

_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")

# Date-like substrings are NOT claimed figures — they are time labels (2017-08-16, 08/16/2017,
# timestamps, "Aug 16 2017"). We remove them before pulling free numbers, so their components
# (2017, 16, ...) are never treated as ungrounded. This does NOT widen the grounded set — real
# claimed numbers are still checked exactly as before.
_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b"   # 2017-08-16, 2017-08-16 00:00:00
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"                     # 08/16/2017, 16-08-2017
    r"|\b\d{4}/\d{1,2}/\d{1,2}\b"                             # 2017/08/16
)


def _strip_dates(text: str) -> str:
    return _DATE_RE.sub(" ", text)


def _to_float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def collect_grounded_numbers(result) -> set[float]:
    """Every number the narrative is ALLOWED to state — read from the ForecastResult only."""
    vals: set[float] = set()

    def add(x):
        try:
            if x is not None:
                vals.add(round(float(x), 4))
        except (TypeError, ValueError):
            pass

    add(result.horizon)
    add(len([h for h in (result.history or []) if h.get("y") is not None]))

    for f in (result.forecast or []):
        for k in ("yhat", "lower", "upper"):
            add(f.get(k))

    bt = result.backtest or {}
    model = bt.get("model", {}) or {}
    for k in ("mae", "rmse", "mape"):
        add(model.get(k))
    add(bt.get("interval_coverage"))
    for fold in bt.get("folds", []):
        add(fold.get("mae"))

    naive = (result.baseline or {}).get("naive", {}) or {}
    for k in ("mae", "rmse", "mape"):
        add(naive.get(k))

    conf = result.confidence or {}
    add(conf.get("score"))
    for v in (conf.get("breakdown") or {}).values():
        add(v)

    # derived numbers the narrative may legitimately restate
    nm = naive.get("mae")
    mm = model.get("mae")
    if nm not in (None, 0) and mm is not None:
        add(round((nm - mm) / nm * 100, 2))          # improvement %
    cov = bt.get("interval_coverage")
    if cov is not None:
        add(round(cov * 100, 2))                     # coverage as a percentage
        add(round(cov * 100))
    add(80); add(0.80)                               # the nominal interval level
    return vals


def validate_numbers(text: str, result, tolerance: float = 0.02) -> dict:
    """Check every number in `text` against the grounded set. Returns which numbers are
    grounded and which are not. `tolerance` is relative, to allow rounding in prose.

    Date-like substrings (2017-08-16, 08/16/2017, timestamps) are excluded first: their digits
    are time labels, not claimed figures. They are reported in `dates_skipped` so the exclusion
    is visible rather than silent."""
    grounded = collect_grounded_numbers(result)
    dates_skipped = _DATE_RE.findall(text)
    scannable = _strip_dates(text)

    found = []
    ungrounded = []
    for tok in _NUM_RE.findall(scannable):
        val = _to_float(tok)
        if val is None:
            continue
        v = round(val, 4)
        ok = any(_close(v, g, tolerance) for g in grounded)
        found.append({"token": tok, "value": v, "grounded": ok})
        if not ok:
            ungrounded.append(tok)
    return {
        "all_grounded": len(ungrounded) == 0,
        "n_numbers": len(found),
        "n_ungrounded": len(ungrounded),
        "ungrounded_tokens": ungrounded,
        "dates_skipped": dates_skipped,
        "detail": found,
    }


def _close(a: float, b: float, rel: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / scale <= rel


# --------------------------------------------------------------------------- #
# 2) Narrative (OpenAI) — runs LAST, validated after generation
# --------------------------------------------------------------------------- #

_SYSTEM = (
    "You are a forecasting analyst. You will be given a JSON summary of a completed forecast "
    "(metrics, baseline comparison, interval coverage, confidence, risks). Write a short, plain "
    "explanation (4-6 sentences) of what the forecast says and how reliable it is.\n"
    "STRICT RULE: use ONLY numbers that appear in the JSON. Do not invent, extrapolate, or "
    "compute new figures. If you mention a number, it must be one from the JSON. Prefer words "
    "over numbers where possible."
)


def generate_narrative(result, report: dict, model: str = "gpt-4o-mini") -> dict:
    """Produce a validated narrative. Always returns a dict; the structured report is never
    blocked by the LLM being absent or by validation failing."""
    out = {"available": False, "narrative": None, "validation": None, "error": None}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        out["error"] = "OPENAI_API_KEY not set — narrative skipped; the structured report stands alone."
        return out
    try:
        from openai import OpenAI
    except ImportError:
        out["error"] = "openai package not installed — narrative skipped."
        return out

    import json
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": json.dumps(report, default=str)}],
        )
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        out["error"] = f"LLM call failed: {e}"
        return out

    validation = validate_numbers(text, result)
    out["available"] = True
    out["narrative"] = text
    out["validation"] = validation
    if not validation["all_grounded"]:
        # do NOT ship hallucinated figures as fact — flag the prose
        out["narrative_status"] = "UNVERIFIED"
        out["warning"] = ("The narrative contains number(s) not found in the forecast result: "
                          f"{validation['ungrounded_tokens']}. Treat the prose with caution; "
                          "the structured report above is the source of truth.")
    else:
        out["narrative_status"] = "VERIFIED"
    return out
