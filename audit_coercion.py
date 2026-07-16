"""
audit_coercion.py — TICKET: Numeric Coercion False Positives
============================================================
READ-ONLY. Touches no engine file. It monkey-patches the regex IN MEMORY for the
comparison run, then restores it.

What it answers
---------------
1. Which columns does the CURRENT pattern coerce to numeric?
2. Which of those are semantically corrupted (a category read as a number)?
3. What would the PROPOSED (unit-whitelist) pattern do instead?
4. What is the impact on the model: R2 / MAE / Accuracy, current vs proposed.

Usage
-----
    python audit_coercion.py --csv train.csv --target SalePrice
    python audit_coercion.py --csv used_cars.csv --target price
    python audit_coercion.py --csv adult.csv --target income
    python audit_coercion.py --csv train.csv --target SalePrice --no-model   # audit only, fast
"""
import argparse
import re

import pandas as pd

from prediction_engine import PredictionEngine
from prediction_engine import profiler as P


# The pattern under review (current, in production)
CURRENT = P._NUMERIC_LIKE

# Proposal: a trailing unit is allowed ONLY if it is a real unit of measure.
PROPOSED = re.compile(
    r"^\s*[+-]?\s*[$€£¥₹﷼]?\s*[+-]?\d[\d,]*(?:\.\d+)?\s*"
    r"(?:%|mi\.?|miles|km|kg|g|lbs?|ft|sqft|sqm|m2|m|hrs?|usd|sar|eur|gbp)?\s*$",
    re.IGNORECASE,
)

RATIO = P._COERCE_MIN_SUCCESS_RATIO   # 0.9


def _would_coerce(s: pd.Series, pattern) -> tuple[bool, float]:
    non_null = s.dropna()
    if non_null.empty:
        return False, 0.0
    share = float(non_null.astype(str).str.match(pattern).mean())
    return share >= RATIO, share


def audit(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        if not P._is_textlike(s):
            continue

        cur_hit, cur_share = _would_coerce(s, CURRENT)
        new_hit, new_share = _would_coerce(s, PROPOSED)

        if not cur_hit and not new_hit:
            continue   # never coerced under either pattern — not interesting

        non_null = s.dropna().astype(str)
        # values that do NOT read as numbers are the ones destroyed (they become NaN,
        # then get imputed with the median of the fake numbers)
        lost = sorted(set(non_null[~non_null.str.match(CURRENT)]))[:4] if cur_hit else []

        if cur_hit and not new_hit:
            verdict = "FALSE POSITIVE — proposal stops corrupting it"
        elif cur_hit and new_hit:
            verdict = "genuine number (both agree)"
        else:
            verdict = "proposal would coerce, current does not"

        rows.append({
            "column": col,
            "is_target": col == target,
            "current_coerces": cur_hit,
            "current_match_%": round(cur_share * 100, 1),
            "proposed_coerces": new_hit,
            "proposed_match_%": round(new_share * 100, 1),
            "n_unique": int(s.nunique(dropna=True)),
            "samples": ", ".join(non_null.unique()[:3]),
            "values_destroyed": ", ".join(lost) if lost else "",
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


def measure(df: pd.DataFrame, target: str, pattern, label: str) -> dict:
    """Run the full engine with `pattern` swapped in, then restore it."""
    saved = P._NUMERIC_LIKE
    P._NUMERIC_LIKE = pattern
    try:
        r = PredictionEngine().run(df.copy(), target_col=target)
        if not r.success:
            return {"variant": label, "status": "FAILED", "error": r.error}
        coerced = [l for l in r.run_log if "Coerced" in l]
        return {
            "variant": label,
            "task": r.fingerprint.task_type,
            **r.evaluation.statistical,
            "cv_mean": r.evaluation.stability.get("cv_mean"),
            "gate": r.evaluation.passed_quality_gate,
            "coerced_cols": coerced[0].split(":")[-1].strip() if coerced else "none",
        }
    finally:
        P._NUMERIC_LIKE = saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--no-model", action="store_true", help="skip the two engine runs")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"{args.csv}: {df.shape[0]} rows x {df.shape[1]} cols | target: {args.target}\n")

    report = audit(df, args.target)
    if report.empty:
        print("No text column is coerced under either pattern.")
    else:
        print("=" * 100)
        print("COERCION AUDIT")
        print("=" * 100)
        print(report.to_string(index=False))

        fp = report[report.verdict.str.startswith("FALSE POSITIVE")]
        print(f"\nFALSE POSITIVES (categories currently read as numbers): {len(fp)}")
        for _, r in fp.iterrows():
            print(f"  - {r['column']}: e.g. {r['samples']}  ->  destroys: {r['values_destroyed']}")

    if args.no_model:
        return

    print("\n" + "=" * 100)
    print("MODEL IMPACT")
    print("=" * 100)
    rows = [measure(df, args.target, CURRENT, "current"),
            measure(df, args.target, PROPOSED, "proposed")]
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nA drop in the metrics is NOT necessarily bad: a corrupted column can act as a\n"
          "leak-ish shortcut. Judge on whether the columns are being read for what they mean.")


if __name__ == "__main__":
    main()
