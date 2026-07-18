"""
benchmark_regression.py — read-only benchmark runner.

Runs the CURRENT engine, unchanged, on House Prices and California Housing and prints
R2 / MAE / CV Mean / CV Std side by side. It imports the engine; it does not modify it.

Usage
-----
    # House Prices (Kaggle "House Prices - Advanced Regression Techniques" -> train.csv)
    python benchmark_regression.py --house path/to/train.csv

    # California Housing (downloaded via scikit-learn, no file needed)
    python benchmark_regression.py --california

    # both
    python benchmark_regression.py --house path/to/train.csv --california
"""
import argparse
import pandas as pd

from prediction_engine import PredictionEngine


def run_one(name: str, df: pd.DataFrame, target_col: str) -> dict:
    print(f"\n{'='*70}\n{name}  —  {df.shape[0]} rows x {df.shape[1]} cols  |  target: {target_col}\n{'='*70}")

    result = PredictionEngine().run(df, target_col=target_col)

    if not result.success:
        print(f"FAILED: {result.error}")
        for line in result.run_log:
            print("   ", line)
        return {"dataset": name, "status": "FAILED", "error": result.error}

    ev, stab = result.evaluation, result.evaluation.stability

    print("\n--- run log ---")
    for line in result.run_log:
        print("   ", line)

    row = {
        "dataset": name,
        "task": result.fingerprint.task_type,
        "best_model": result.training.best_estimator,
        "attempts": result.attempts,
        "R2": ev.statistical.get("r2"),
        "MAE": ev.statistical.get("mae"),
        "CV_Mean": stab.get("cv_mean"),
        "CV_Std": stab.get("cv_std"),
        "n_splits": stab.get("n_splits"),
        "cv_error": stab.get("error"),
        "gate": ev.passed_quality_gate,
        "confidence": result.confidence.score,
    }
    print("\n--- metrics ---")
    for k, v in row.items():
        if v is not None:
            print(f"    {k:12} {v}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", help="path to House Prices train.csv")
    ap.add_argument("--california", action="store_true", help="fetch California Housing via sklearn")
    args = ap.parse_args()

    rows = []

    if args.house:
        df = pd.read_csv(args.house)
        rows.append(run_one("House Prices", df, "SalePrice"))

    if args.california:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing(as_frame=True)
        df = data.frame                      # features + MedHouseVal
        rows.append(run_one("California Housing", df, "MedHouseVal"))

    if not rows:
        ap.error("nothing to run — pass --house and/or --california")

    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    summary = pd.DataFrame(rows)[
        ["dataset", "R2", "MAE", "CV_Mean", "CV_Std", "best_model", "gate", "confidence"]
    ]
    print(summary.to_string(index=False))

    print("""
READ THIS BEFORE INTERPRETING THE NUMBERS
-----------------------------------------
R2 and CV_Mean are NOT measured the same way, so a gap between them is expected:

  CV_Mean : R2 on log1p(target), computed on the TRAINING rows (outliers removed).
  R2      : R2 on the ORIGINAL target scale (expm1 applied), computed on the HELD-OUT
            rows, which are kept intact — extreme values included.

A large CV_Mean with a small R2 means the model is fine on the bulk of the data but
cannot price the extremes (tree models cannot extrapolate beyond the training range,
and the outliers were removed from training). Cross-check with MAE: if MAE is small,
the model is healthy and only R2 is being destroyed by a handful of extreme rows.

California Housing has a hard cap at 5.00001 (500,001 USD) — a large block of identical
capped values. Expect that to affect both metrics.
""")


if __name__ == "__main__":
    main()
