# KNOWN ISSUES — deferred (system frozen for the Regression benchmark)

Status: **frozen**. No feature work, no refactor, no logic changes.
Scope of the current phase: **Regression benchmark only.**

---

## 1. Target Encoding → possible Data Leakage  (DEFERRED)

**Where:** `prediction_engine/preprocessor.py`, step 2 (`role in ("id_like", "text")`).

```python
means = df.groupby(col)[target_col].mean()
df[col] = df[col].map(means)
```

**Issue:** the mapping is fitted on the **full dataframe**, before the train/test split
performed in `orchestrator.py`. The encoded column therefore carries information from rows
that later end up in the held-out test set.

**Consequence for the benchmark:** for any dataset that contains a `text` / `id_like` column
with cardinality < 200, the held-out R² may still be **optimistic**. Datasets without such a
column are unaffected.

**Planned fix (later):** fit the mapping on the training split only, or use out-of-fold
target encoding.

---

## 2. Classification path crashes  (DEFERRED — after the regression tests)

**Where:** `prediction_engine/preprocessor.py`, step 0 (outlier removal).

```python
q_low = df[target_col].quantile(0.01)
```

Runs on a **text target**, raising `TypeError: unsupported operand type(s) for -: 'str' and 'str'`.
The `log1p` transform on the next line has the same problem. Classification therefore fails
before training starts.

**Not in scope now.** To be addressed once the regression benchmark is complete.

---

## 3. `llm_agent.py` import error when `dspy` is absent  (NOT AN ISSUE IN PRACTICE)

`ForecastSignature` references `dspy.InputField` at class-body evaluation time, so importing
the package fails with `NameError` if `dspy` is not installed. `dspy` is listed in
`requirements.txt`, so this does not occur in the normal environment. Left untouched.

---

# WHAT WAS ACTUALLY CHANGED (this phase only)

| File | Change |
|---|---|
| `orchestrator.py` | Added `train_test_split` (test_size=0.2, stratified for classification). Training now runs on the train split; final metrics come from the held-out test split. Added logs. |
| `evaluator.py` | `evaluate()` accepts optional `X_cv` / `y_cv`. Statistical metrics + SHAP use the held-out data; Cross-Validation stability runs on the training split. |
| `preprocessor.py` | Logging only (rows/columns before and after cleaning, dropped columns, transformed columns) + this note as a comment. **No cleaning logic was modified.** |

Untouched: `profiler.py`, `trainer.py`, `confidence.py`, `llm_agent.py`, `__init__.py`,
`app.py`, `requirements.txt`, DSPy, Forecast, Agents, ReAct, UI, Confidence Engine,
Feature Engineering.

---

# BASELINE (verified run, synthetic regression, 600 rows)

```
[Preprocessing] Rows before cleaning: 600
[Preprocessing] Columns before cleaning: 6
[Preprocessing] Rows after cleaning: 588  (removed: 12)
[Preprocessing] Columns after cleaning: 8  (was 6)
[Preprocessing] Dropped columns (removed entirely): id
[Preprocessing] Transformed columns (replaced by encoded/derived features): district
[Preprocessing] Training Rows: 588
[Preprocessing] Training Columns: 7
[Split] Train: 470 rows | Test (held-out): 118 rows | test_size=0.2
[Training] FLAML final selected model: <estimator>
[Evaluation] Metrics computed on 118 held-out rows (unseen during training)
             | CV run on 470 training rows.
```

Note: this baseline was produced with stubbed `flaml` / `shap` (no network in the build
environment), so the estimator name and the metric values will differ on a real run. The
pipeline wiring — split, log lines, metric source, `predict()` — is what was verified.

---

# FIX LOG — feature-name sanitation (California Housing)

**Symptom:** `feature_names must be string, and may not contain [, ] or <`

**Source:** One-Hot Encoding in `preprocessor.py` builds column names as `{col}_{category}`.
The California Housing category value `<1H OCEAN` produced the column
`ocean_proximity_<1H OCEAN`, which XGBoost/LightGBM reject.

**Fix (scope: column names only):**
- `_sanitize_name()` / `_build_rename_map()` added to `preprocessor.py`.
- Applied once, **after** encoding, to every column except the target.
- The map is stored on `PreprocessingResult.renamed_cols` and replayed inside
  `transform_new()` so predict-time columns match training columns exactly.

Nothing else changed: no training, evaluation, cleaning, forecast, agent or UI logic touched.

---

# FIX LOG — numeric values stored as text (Used Cars)

**Symptom:** target `price` holds `$10,300`, `$38,005`... -> profiled as text ->
`task_type` inferred as **classification** instead of regression (and the classification
path then crashes at outlier removal). Blocking.

**Fix (scope: column interpretation only):**
- `_coerce_numeric_like()` added to `profiler.py`. A text column is converted to `float64`
  only when >= 90% of its values match a strict "reads as a number" pattern:
  optional currency symbol -> digits (optional `,` separators / decimals) -> optional short
  trailing unit. Examples converted: `$10,300`, `38,005`, `51,000 mi.`
  Examples **left untouched**: `P123` (ID), `Toyota`, `<1H OCEAN`, `S/M/L`.
- Called once in `orchestrator.run()` **before profiling**, because both the profiler and
  the preprocessor consume the same dataframe.
- Output dtype is `float64` on purpose: the profiler's `id_like` rule exempts float columns,
  so a near-unique price column keeps the `numeric` role instead of being read as an ID.

Verified: task type is now `regression`; real categorical columns keep their role; the
real-estate and California-Housing benchmarks return byte-identical metrics (no regression).

Nothing else changed: FLAML, evaluation, forecast, agents and UI untouched.

---

# FIX LOG — making the Classification path runnable

Three blocking changes, no accuracy/threshold/metric work.

1. `preprocessor.py` — **Outlier removal** now runs only when `task_type == "regression"`.
   Classification logs: `Outlier removal: skipped (task type is classification).`
2. `preprocessor.py` — **log1p on the target** now runs only when `task_type == "regression"`.
   Classification logs: `Log transformation: skipped (task type is classification).`
   `target_log_transformed` is `False` for classification.
3. `evaluator.py` — **`expm1()` moved inside the regression branch.** It was applied to `y`
   and to the predictions *before* the task-type branch, so a text target ("yes"/"no")
   raised: `loop of ufunc does not support argument 0 of type str ...`, which the retry loop
   swallowed into `Training attempts exhausted`. The classification branch already used the
   raw `y` / `preds`, so it is unaffected; regression metrics are unchanged.

**Verified — classification benchmark now runs on all five datasets:**

| dataset | accuracy | f1_weighted | gate |
|---|---|---|---|
| Titanic | 0.8187 | 0.8172 | pass |
| Telco Churn | 0.6278 | 0.6225 | pass |
| Adult Income | 0.7350 | 0.7258 | pass |
| Bank Marketing | 0.7889 | 0.7890 | pass |
| HR Attrition | 0.9000 | 0.8827 | pass |

(Stubbed flaml/shap in the build environment — the numbers above prove the path executes,
they are not the real FLAML scores.)

**Regression benchmarks re-run: byte-identical metrics. No regression.**
Real estate R2 0.9813 | California 0.9740 | Used Cars 0.9494

---

# FIX LOG — Target Encoding on a text target (Adult Income)

**Symptom:** `TypeError: dtype 'str' does not support operation 'mean'`
Raised by `df.groupby(col)[target_col].mean()` in `preprocessor.py` step 2, which assumed a
numeric target. Triggered by any classification dataset holding a `text`-role column
(cardinality < 200), e.g. Adult Income's `native-country` (42 categories).

**Fix (scope: target encoding only):**
- For `task_type == "classification"` a TEMPORARY numeric view of the target is built with
  `LabelEncoder`, and used **only** to compute the per-category means.
- The real target column is never modified — it stays text and reaches FLAML unchanged.
- Regression is untouched: `te_target` is simply the target column itself, so the means are
  computed exactly as before.

Verified: Adult Income (with a 42-category text column) now runs; the target stays
`>50K` / `<=50K`; `native_country` becomes `float64`. All 3 regression and all 5
classification benchmarks return identical metrics — no regression.

## STILL OPEN (deferred, NOT fixed)

- **Leakage:** the means are still fitted on the full dataframe, before the train/test split.
- **`transform_new()` does not replay the target-encoding map.** At predict time a
  target-encoded column keeps its raw text value and is passed to the model as-is.
  This affects the Predict tab (not the benchmark) for any dataset with a `text`-role column.

---

# FIX LOG — Target Encoding replay at predict time

**Symptom:** `transform_new()` never replayed the target-encoding mapping learned during
training. A target-encoded column kept its raw text value and was handed to the model as-is.

**Fix (scope: predict-time mapping only):**
- `PreprocessingResult.target_encoded_cols` now stores, per column,
  `{"map": {category: mean}, "default": <training global mean>}`.
- `transform_new()` replays that map **before** imputation — the same order as at fit time
  (target encoding is step 2, imputation is step 3).
- Safe handling of new values: an unseen category or a missing value falls back to the
  training global mean, so no raw string can ever reach the model.

**Verified** (Adult Income, `native_country`, 42 categories):

| input | value after `transform_new()` |
|---|---|
| seen category `Country_0` | 0.2403 (from the map) |
| unseen category `Atlantis` | 0.2300 (fallback) |
| missing `NaN` | 0.2300 (fallback) |

No strings remain; predict-time columns match training columns exactly.
Training, evaluation and FLAML untouched. All 3 regression + 5 classification benchmarks
return identical metrics — no regression.

## NEXT BLOCKING BUG (found, NOT fixed — separate issue)

`orchestrator.predict()` ends with `return [float(p) for p in preds]`. A classification model
returns text labels, so Predict raises:
`ValueError: could not convert string to float: '<=50K'`
Unrelated to target encoding. Regression Predict is unaffected.

---

# FIX LOG — Predict path for Classification

Two changes, no new logic.

1. `orchestrator.py` — `predict()` ended with `return [float(p) for p in preds]`, which broke on
   text labels (`ValueError: could not convert string to float: '<=50K'`). It now branches on
   `fingerprint.task_type`: classification returns the labels as-is (no `expm1`, no `float()`
   cast); regression keeps the exact previous behaviour (inverse log + float).

2. `app.py` — display only. The Predict tab reached
   `f"{pred_value:,.2f}"` for classification (`ValueError: Unknown format code 'f' for object of
   type 'str'`), because its label-decoding branch required `encoders["__target__"]`, a key the
   preprocessor never sets. Now: classification -> show the label as-is; regression -> unchanged
   numeric formatting. The legacy label-encoder branch is kept in case the key ever exists.

**Verified:**

| dataset | predict() | Predict tab shows |
|---|---|---|
| Adult Income | `['<=50K']` | Predicted Class: **<=50K** |
| Telco Churn | `['Yes']` | Predicted Class: **Yes** |
| HR Attrition | `['Attrition']` | Predicted Class: **Attrition** |
| Regression | `[1482897.48...]` | Predicted Value: **1,482,897.48** |

Training, evaluation, FLAML, preprocessing and target encoding untouched.
All regression + classification benchmarks return identical metrics — no regression.

---

# FIX LOG — Target Encoding leakage removed  (item 1 of the leakage list)

**Was:** the per-category means were computed with `groupby(...).mean()` over the FULL
dataframe, before the train/test split. Test rows contributed to the encoding of their own
features. Held-out metrics were therefore optimistic for any dataset with a `text`-role column.

**Now:** the split is decided FIRST (on row labels of the raw dataframe), and the training
indices are passed into `Preprocessor.run(df, fingerprint, train_index=...)`. The means and the
fallback default are computed from **training rows only**, then applied to every row.

**Verified independently:** the stored map matches a hand-computed train-only groupby
(`True`) and does NOT match the old full-data groupby (`False`). Log line now reads:
`Applied Target Encoding to column 'native_country' ... (fitted on 800 training rows only).`

**Order change (necessary):** `train_test_split` moved from after preprocessing to before it,
and is now applied to row labels. Same `test_size=0.2`, same `random_state=42`, same stratify
rule. Rows removed later by outlier removal simply drop out of whichever side they were on.

**Not touched:** outlier removal and imputation are still fitted on the full data — items 2 and 3
of the leakage list, still open.

## Benchmark shift (expected — the numbers got HONEST, they did not get worse)

| dataset | before | after |
|---|---|---|
| Real estate (reg) | 0.9813 | 0.9830 |
| California (reg) | 0.9740 | 0.9726 |
| Used Cars (reg) | 0.9494 | 0.9547 |
| Titanic | 0.8187 | 0.8187 |
| Telco Churn | 0.6278 | 0.6500 |
| Adult Income (5-cat) | 0.7350 | 0.7400 |
| Bank Marketing | 0.7889 | 0.8000 |
| HR Attrition | 0.9000 | 0.9071 |
| **Adult Income (42-cat, the leaky one)** | **0.7500** | **0.7550** |

Small movements on the leak-free datasets come from the split now being drawn before outlier
removal, so train/test membership differs slightly. Not a behaviour change.

---

# FIX LOG — Classification Cross-Validation failure

**Symptom** (swallowed by the try/except in `_stability`, surfaced in the UI as
"Could not calculate stability"):

```
All the 5 fits failed ...
ValueError: Invalid classes inferred from unique values of `y`.
            Expected: [0 1], got ['<=50K' '>50K']
```

**Cause:** `cross_val_score` refits a *clone of the raw estimator*
(`trained_model.model.estimator`), which bypasses the label encoding FLAML applies
internally. FLAML trains fine, but the bare `XGBClassifier` rejects text class labels, so
every fold fails. Only classification with a text target is affected — hence Adult Income,
Telco Churn and HR Attrition.

**Fix (scope: `Evaluator._stability()` only):** the labels are `LabelEncoder`-encoded for
the CV call only. Accuracy is invariant to class renaming, so the score value is unchanged —
the encoding merely makes it computable. Regression takes the untouched `KFold` / `r2` path.

**Verified** (stub faithfully reproducing XGBoost's rejection of text labels):

| dataset | before | after |
|---|---|---|
| Adult Income | `cv_mean = None` (5 fits failed) | `cv_mean = 0.7513`, `cv_std = 0.0207` |
| Telco Churn | `cv_mean = None` | `cv_mean = 0.7014`, `cv_std = 0.0211` |
| HR Attrition | `cv_mean = None` | `cv_mean = 0.8339`, `cv_std = 0.0268` |

Regression CV unchanged (`cv_mean = 0.9731`). Statistical metrics, thresholds, the quality
gate, FLAML, preprocessing and target encoding all untouched.

**Side effect worth knowing:** with `cv_mean` no longer `None`, the Confidence Engine stops
falling back to its hardcoded `stability = 0.5` for classification, so confidence scores now
reflect real fold variance. The `MAX_CV_STD_RATIO` gate also becomes active for
classification for the first time — an unstable model can now legitimately be rejected.

---

# FIX LOG — Outlier removal fitted on training rows only  (item 2 of the leakage list)

**Was:** the 1% / 99% target quantiles were computed over the FULL dataframe, before the
split, so the test rows helped decide their own thresholds.

**Now:** the quantiles are computed from the training rows only (`train_index`, already
available from the target-encoding fix). The same thresholds are then applied to every row.

**Verified** (real-estate benchmark):

```
train-only quantiles : (4.468e+05, 2.262e+06)   <- what is used now
full-data quantiles  : (4.479e+05, 2.262e+06)   <- the old leaky ones
[Preprocessing] Outlier thresholds fitted on 480 training rows only: (4.468e+05, 2.262e+06).
[Preprocessing] Outlier removal: Data reduced from 600 to 587 rows (10 from train, 3 from test).
```

| dataset | before | after |
|---|---|---|
| Real estate | 0.9830 | 0.9818 |
| California | 0.9726 | 0.9705 |
| Used Cars | 0.9547 | 0.9555 |

Classification is unchanged (the step is skipped there). Imputation, evaluation and FLAML
untouched.

## STILL OPTIMISTIC — deliberate, per instruction (NOT a leak)

Rows outside the thresholds are still dropped **from the test split as well**
(`3 from test` above). No test information is used to build the thresholds, so there is no
leakage — but the hardest cases (cheapest / most expensive properties) are removed from the
data the model is scored on, so R2 stays flatteringly high.

The stricter alternative — keep the test split intact and drop outliers from the training
rows only — is a one-line change:

```python
keep = (df[target_col] > q_low) & (df[target_col] < q_hi)
keep.loc[keep.index.difference(fit_idx)] = True   # never drop a test row
```

Not applied: the current instruction is to apply the same decision to the test data.

---

# FIX LOG — Outliers dropped from the TRAINING rows only

**Change:** thresholds still come from the training rows, and now the removal itself is
restricted to those rows. The held-out split is never touched.

```python
keep = (df[target_col] > q_low) & (df[target_col] < q_hi)
keep.loc[keep.index.difference(fit_idx)] = True   # never drop a test row
```

**Verified** (real estate, 600 rows):

```
test rows in original split : 120
test rows surviving pipeline: 120
ALL test rows kept          : True
most extreme test rows still evaluated: True
train rows: 480 -> surviving: 470 (dropped 10)
[Preprocessing] Outlier removal: 10 training rows dropped (600 -> 590).
                Test rows dropped: 0 (held-out set kept intact).
```

| dataset | full-data thresholds (leaky) | train thresholds, test also trimmed | train thresholds, test intact |
|---|---|---|---|
| Real estate | 0.9830 | 0.9818 | 0.9819 |
| California | 0.9726 | 0.9705 | 0.9714 |
| Used Cars | 0.9547 | 0.9555 | 0.9569 |

Classification unchanged. Imputation, evaluation, FLAML, confidence untouched.

Note: on these synthetic benchmarks the metrics barely move, because the generated targets
have no heavy tails — the extreme rows are not genuinely hard. On real data with true
outliers the effect is larger, and the numbers should be read as *more honest*, not better.

## REMAINING LEAK — item 3

Imputation (median / mode) is still computed over the full dataframe, test rows included.

---

# FIX LOG — Imputation fitted on training rows only  (item 3 — LAST leak closed)

**Was:** `df[col].median()` / `df[col].mode()` were computed over the full dataframe, so the
test rows helped decide the value used to impute themselves.

**Now:** the fill value comes from the surviving training rows only, then is applied to every
row. `fit_idx` is re-intersected after outlier removal, so rows dropped as training outliers
no longer contribute. A numeric column whose training values are all missing falls back to 0
instead of producing NaN.

**Verified** with a deliberately divergent test distribution (`age = 200` for every test row):

```
impute value used : 15.0
train-only median : 15.0   -> match: True
full-data median  : 18.0   -> match: False   <- the old leaky value
[Preprocessing] Imputed missing values in 'age' with 15.0 (computed from 470 training rows only).
```

A leaky median would have been pulled toward 18.0. It was not.

Evaluation, FLAML, confidence, SHAP untouched. Predict path still correct
(target-encoding replay and classification labels both verified).

Titanic moved 0.8187 -> 0.8250: it is the only benchmark with missing values in a feature
(`Age`), so it is the only one whose imputed values changed. Expected, not a regression.

---

# LEAKAGE LIST — ALL THREE ITEMS NOW CLOSED

| # | item | status |
|---|---|---|
| 1 | Target Encoding fitted on full data | FIXED — training rows only |
| 2 | Outlier thresholds fitted on full data | FIXED — training rows only, test set kept intact |
| 3 | Imputation fitted on full data | FIXED — training rows only |

Every fit-time statistic in the preprocessor is now computed from the training split alone.
The held-out metrics are, to the best of the current design, honest.

## Still open (lower priority, not leaks)

- `trainer.py` logs `Best accuracy (R2) achieved: {automl.best_loss}` — that value is a LOSS,
  not R2. Mislabelled log line.
- The self-correction loop only raises the time budget; it does not correct anything. If all
  3 attempts fail the quality gate, the result is still returned with `success=True`.
- The quality gate reads the same held-out set that is reported, so retrying until the gate
  passes introduces a mild selection bias on the test split.
- `ConfidenceEngine` uses `fingerprint.n_rows` (pre-cleaning) for data adequacy, and ignores MAE.
- `_infer_task_type` requires `unique_ratio < 0.05`; a small dataset with a binary target can
  be misread as regression.
- `requirements.txt` pins versions that may not resolve (pandas>=3.0.2, numpy>=2.4.4, sklearn>=1.8.0).
- `llm_agent.py` raises NameError at import if dspy is absent.

---

# REGRESSION INTRODUCED BY ME, AND FIXED — categorical imputation

**What happened:** while adding the imputation log line, the two lines that compute the mode
were deleted by accident. The categorical branch became:

```python
else:
    df[col] = df[col].fillna(fill_val)      # fill_val left over from a previous NUMERIC column
    impute_values[col] = fill_val
```

**Effect:** a categorical column with missing values was imputed with the *median of whatever
numeric column happened to be processed before it* — silently, with no error. If a categorical
column had been the first one with missing values, it would have raised UnboundLocalError.

Demonstrated on a House-Prices-shaped frame:
`{'LotFrontage': 76.0, 'GarageType': 76.0}` — `GarageType` filled with a number.

**Why the tests missed it:** every benchmark used here has missing values only in NUMERIC
columns (Titanic `Age`, real-estate `age`). The categorical branch was never exercised.

**Scope of the damage:** any dataset with missing values in a categorical column. House Prices
is full of them (`Alley`, `MasVnrType`, `GarageType`, `BsmtQual`, ...). The House Prices and
California Housing benchmark numbers produced BEFORE this fix are not trustworthy and must be
re-run.

**Fix:** the mode computation is restored, fitted on training rows only.

```
GarageType: expected train-only mode = 'BuiltIn' | got = 'BuiltIn' | match = True
Alley:      expected train-only mode = 'Grvl'    | got = 'Grvl'    | match = True
```

All other benchmarks are unchanged (they never hit this branch).

---

# ReAct Router v1 — agent layer added ON TOP of Benchmark-Stable-v1

**Core untouched.** `orchestrator.py`, `preprocessor.py`, `evaluator.py`, `trainer.py`,
`profiler.py`, `confidence.py`, `llm_agent.py`, `app.py` are byte-identical. All benchmarks
return identical numbers.

| file | change |
|---|---|
| `events.py` | NEW — structured event log |
| `router.py` | NEW — intent detection + tool routing |
| `__init__.py` | exports only |

## 1. Structured events

```json
{"stage": "intent_detection", "action": "prediction_selected",
 "reason": "keyword rule matched", "data": {...}, "status": "ok", "ts": ...}
```

`EventLog.to_list()` / `.to_json()` for machines, `.pretty()` for the existing Run Log tab.
The engine's own `run_log` is preserved and attached as event payload — nothing was removed.

## 2. Intent detection

Order: explicit user choice (a UI button) -> LLM (OpenAI, optional) -> keyword rules (always).
The LLM never becomes a dependency: no key, no `openai` package, or a failed call all degrade
to rules, and the reason is recorded as an event. Lesson taken from the `dspy` import bug.

12/12 on the test cases, including the ambiguous Arabic verb: "توقع" means both *predict* and
*forecast*, so it cannot decide the intent on its own — the temporal marker does:

| message | intent |
|---|---|
| توقع السعر لشقة 4 غرف | prediction |
| توقع المبيعات للسنة القادمة | forecast |
| predict the next quarter revenue | forecast |

## 3. Tools

- **PredictionTool** — calls the frozen `PredictionEngine`. Zero new ML logic.
- **ForecastTool** — PLACEHOLDER. Returns `status="not_implemented"` and says so plainly.
  It does not fabricate a trend.
- **ExplanationTool** — builds a report from an existing `EngineResult`. Every figure is read
  from the trained model; none is generated.

Unknown intent -> no tool is called, and the user is asked. The router never guesses.

## NOT integrated into the UI

`app.py` was left untouched deliberately. The router is a library; wiring it into the Predict
tab is a separate, explicit step.

---

# Smart Prediction Form

New file `form.py`. `app.py` Predict tab rewritten to use it. `__init__.py` exports it.
ML core untouched (orchestrator, preprocessor, evaluator, trainer, profiler, confidence)
and so is the router.

## The real problem it solves

SHAP reports importances for POST-preprocessing features — `district_Olaya`,
`ocean_proximity__1H_OCEAN`, `signup__year` — while the user fills in RAW columns. The module
walks the preprocessing artefacts backwards (`onehot_cols`, `renamed_cols`, datetime
decomposition, target-encoded names) and folds every derived feature back onto the raw column
it came from, summing their importance.

Verified on the three awkward cases:

| case | post-preprocessing feature | mapped back to |
|---|---|---|
| one-hot | `district_Sulay` | `district` |
| renamed for XGBoost | `ocean_proximity__1H_OCEAN` | `ocean_proximity` |
| target-encoded | `native_country` | `native_country` |

Total importance captured on Adult Income: **1.0** — no feature is lost in the mapping.

## Behaviour

- **Required**: the top raw columns covering 85% of the summed importance (min 3, max 8).
- **Optional**: everything else, in a collapsed expander, pre-filled with the median
  (numeric) or the most common value (categorical).
- **Excluded**: columns the pipeline dropped (IDs, high-cardinality text) — never shown.
- **Fallbacks**: if SHAP failed, it falls back to the profiler's linear correlations; if those
  are empty too, it asks for every column rather than guessing.

Example (real-estate, 6 usable columns): the user fills **3** fields instead of 6; `listing_id`
is not asked at all; `noise1` and `district` default silently.

---

# FIX LOG — Smart Form crash: median on a string column

**Symptom:** `TypeError: Cannot perform reduction 'median' with string dtype`
at `form.py::_make_field` -> `float(s.median())`.

**Root cause:** `orchestrator.run()` calls `_coerce_numeric_like(df)` BEFORE profiling, so the
fingerprint's roles describe the COERCED frame. `build_form_spec()` was handed
`st.session_state.raw_df` — the RAW frame. A column could therefore be `role="numeric"` and
`dtype=string` at the same time.

```
LotArea      profiler role = numeric   | raw_df dtype = str
HouseStyle   profiler role = numeric   | raw_df dtype = str
```

**Fix (form.py only):**
1. `build_form_spec()` now views the data through the engine's own `_coerce_numeric_like()`,
   so the form sees exactly the frame the profiler saw.
2. `_make_field()` no longer trusts `role` on its own: it verifies the dtype, coerces with
   `pd.to_numeric(errors="coerce")`, and degrades to a categorical field instead of raising if
   nothing converts.

ML core untouched. All benchmarks identical.

## SEPARATE, MORE SERIOUS ISSUE FOUND — in the ML core, NOT fixed

`HouseStyle` is being silently converted to a number by `_coerce_numeric_like`.

Its values are `1Story`, `2Story`, `1.5Fin`, `2.5Unf`, `SFoyer`, `SLvl`. The coercion pattern
accepts "digits + a short trailing unit" (built for `51,000 mi.`), so:

| raw value | becomes |
|---|---|
| `1Story` | 1.0 |
| `2Story` | 2.0 |
| `1.5Fin` | 1.5 |
| `SFoyer`, `SLvl` | NaN -> imputed with the median |

About 92% of the values match the pattern — above the 0.9 threshold — so the whole column is
converted. The model is training on `HouseStyle` as a NUMBER, and the non-numeric styles are
erased. No error is raised.

This is semantic data corruption in `profiler._coerce_numeric_like`, introduced by the
Used-Cars currency fix. Any categorical column whose labels start with digits is at risk
(`1Story`, `2Fam`, `1st Class`, grades like `3A`).

Deferred: the fix belongs in the ML core, which is out of scope for this task.

---

# TICKET — Numeric Coercion False Positives  (OPEN, analysis only, no fix applied)

## The pattern under review

`profiler._NUMERIC_LIKE`:

```python
r"^\s*[+-]?\s*[$€£¥₹﷼]?\s*[+-]?\d[\d,]*(?:\.\d+)?\s*[a-zA-Z.%/]{0,6}\s*$"
                                                    ^^^^^^^^^^^^^^^^^^
```

The trailing group accepts ANY 6 letters. It was written for `51,000 mi.` (Used Cars) but the
only real condition it imposes is "starts with a digit".

## What gets corrupted

A text column is converted to float when >=90% of its values match. Non-matching values become
NaN and are then imputed with the median of the fake numbers — silently, with no error.

| column | values | current | result |
|---|---|---|---|
| `HouseStyle` | 1Story, 2Story, 1.5Fin, SFoyer, SLvl | 95.8% match -> COERCED | 1Story->1.0, SFoyer/SLvl destroyed |
| `BldgType` | 1Fam, 2fmCon, Duplex, TwnhsE | 86% match -> survives | narrowly under the 0.9 threshold |
| `LotArea` | "5,144" | 100% | genuinely a number — correct |
| `Distance` | "16 mi." | 100% | genuinely a number — correct |

`BldgType` is the warning: it is safe only by luck. Any column whose labels start with digits is
at risk — `1st Class`, `2Fam`, grade codes like `3A`, `1.5Baths`.

## Proposed pattern — unit whitelist

```python
r"^\s*[+-]?\s*[$€£¥₹﷼]?\s*[+-]?\d[\d,]*(?:\.\d+)?\s*"
r"(?:%|mi\.?|miles|km|kg|g|lbs?|ft|sqft|sqm|m2|m|hrs?|usd|sar|eur|gbp)?\s*$"
```

A trailing unit is allowed only if it is a REAL unit of measure — otherwise no suffix at all.
`51,000 mi.` and `$10,300` still convert. `1Story`, `1.5Fin`, `1.6L` no longer do.

## Measured impact (Ames-shaped synthetic, stubbed FLAML)

| variant | R2 | MAE | CV mean | coerced columns |
|---|---|---|---|---|
| current | 0.8930 | 20,482 | 0.8740 | HouseStyle, LotArea, Distance |
| proposed | 0.8928 | 20,585 | 0.8744 | LotArea, Distance |

The metrics barely move — `HouseStyle` carries little signal here. **The point is not the score,
it is that a categorical column stops being read as a number and its rarest categories stop
being erased.**

## Tool

`audit_coercion.py` — read-only. Monkey-patches the regex in memory, never edits an engine file.

```
python audit_coercion.py --csv train.csv --target SalePrice
python audit_coercion.py --csv used_cars.csv --target price
python audit_coercion.py --csv adult.csv --target income
```

Prints, per column: match % under both patterns, sample values, the values that would be
destroyed, a verdict, and then R2/MAE/Accuracy under each pattern.

## Residual risk of the proposal (to weigh before deciding)

The whitelist is finite. A legitimate unit that is missing from it (`bhp`, `cc`, `L`, `pcs`,
`bar`, `°C`) stops being coerced and the column stays text — the exact opposite failure. Two
mitigations, both deferrable: extend the list, or drop suffix support entirely and require a
column to be a plain number.

STATUS: no code changed. Awaiting the numbers from the three real datasets.

---

# Forecast Engine — STEP 1 built (detection + skeleton). No model, no LLM.

New file `forecast.py`. `__init__.py` exports it. Everything else (ML core, router, events,
form) is byte-identical.

## What Step 1 does
- `detect_time_candidates(df, target)` — finds usable time axes by inspecting the RAW columns:
  real datetime -> parseable string (>=95%) -> monotonic integer index. Ranked best-first.
- `prepare_series(...)` — validates the target is numeric, picks the time axis, regularizes the
  observed series onto the inferred frequency grid, and returns a `ForecastResult` holding ONLY
  the history. `forecast`, `backtest`, `baseline`, `model`, `confidence` are intentionally empty.

## Decisions honoured
- **#3 (date before coercion):** detection parses the ORIGINAL column with `pd.to_datetime`;
  it never calls `_coerce_numeric_like`. A date stored as a string stays a Timestamp, verified
  in tests (it is NOT turned into a number).
- **#5 (single source of numbers):** `ForecastResult` is the only container.
- **Honest refusal:** no time axis / non-numeric target / <3 points -> `success=False` with a
  clear reason. No fabricated series.
- **No LLM import** in `forecast.py` (ordering principle 0.0).

## Verified (7 cases)
string-dates ok · daily-with-gaps ok (gaps reported, not filled) · no-time-axis refused ·
ordered-int ok · non-numeric-target refused · too-short refused · multi-candidate ranked
(real datetime before integer index).

## NOT built yet (awaiting Step 1 review)
FLAML forecasting, baseline, backtest, intervals, visuals, report, LLM narrative.

---

# Anthropic dependency removed + Forecast STEP 2

## Anthropic -> OpenAI (provider cleanup)
Full grep before/after. Every reference switched:

| file | was | now |
|---|---|---|
| llm_agent.py | model "anthropic/claude-sonnet-4-6", ANTHROPIC_API_KEY | "openai/gpt-4o-mini", OPENAI_API_KEY |
| app.py | ANTHROPIC_API_KEY (3 spots) | OPENAI_API_KEY |
| requirements.txt | anthropic>=0.116.0 | openai>=1.40.0 |
| README.md | "DSPy + Claude", sk-ant- | "DSPy + OpenAI", sk- |

Verified: zero `anthropic` / `ANTHROPIC_API_KEY` references remain. The agent still degrades
gracefully with no key (`available=False`, error names OPENAI). ML core untouched.

## Forecast STEP 2 — Temporal preparation (no model, no LLM)
Added `prepare_temporal()` + `TemporalPrep` to `forecast.py`.

- **Regularize:** consumes the Step-1 regular grid.
- **Detect missing periods:** interior gaps grouped into runs (`gap_runs`), long runs warned.
- **Explicit gap strategy:** one of `none | ffill | linear | zero | mean`, chosen by the caller,
  never silent. Unknown strategy is rejected.
- **Edge policy:** leading/trailing gaps are TRIMMED, never filled — edge history is not
  fabricated. Interior gaps are filled per strategy.
- Full audit on the result: n_observed, n_filled, gap_runs, leading_trailing_trimmed.

Verified: 4 edge periods trimmed, 7 interior gaps filled under each strategy, long-gap warning
fires, bad strategy rejected, no interior None left after fill.

NOT built: FLAML forecasting, backtest, intervals, visuals, report, LLM narrative.

---

# Step 2 adjustment — gap strategy is now REQUIRED (no default)

Per review: `ffill` must not be the silent default. Changed `prepare_temporal(fr, strategy)` so
`strategy` is a required positional argument — there is no default at all. Calling without it
raises `TypeError`, which forces every call site to state the strategy explicitly until the
gap-handling logic is defined in a later stage.

- `strategy="none"` regularizes and audits gaps WITHOUT filling, and warns that a strategy must
  be chosen before modelling.
- `ffill | linear | zero | mean` unchanged.
- unknown strategy still rejected.

No default fill is applied anywhere. The decision stays with the caller.

---

# FIX LOG — 'EngineResult' object is not subscriptable (app.py:85)

**Diagnosis (answering the three questions): none of the three.** The contract MATCHED on the
success path — PredictionTool returns `output = {"engine_result":..., "predictions":...}`, which
is what app.py reads. The mismatch was on the ERROR path only:

```python
# router.py, failure branch — BEFORE
return ToolResult(self.name, "error", result, ...)   # result is a raw EngineResult, not a dict
```

`app.py:85` read `tool_result.output["engine_result"]` WITHOUT checking status, so a failed
pipeline (e.g. quality-gate rejection on weak data) made output a raw EngineResult and the
subscription blew up. It was a latent bug that only surfaced the first time a run failed —
app.py had always assumed success.

**Fix — unify the contract (option B): `output` is ALWAYS a dict.**
- `router.py`: the failure branch now returns `{"engine_result": result, "predictions": None}`,
  same shape as success.
- `app.py`: reads defensively via `out.get("engine_result") if isinstance(out, dict) else None`
  (line 85) and the same guard for `predictions` in the Predict tab (was also unguarded).

Verified: failing pipeline -> no crash, result still extracted; success unchanged (R2 0.9852);
predict on a failed model -> clean error, no crash; forecast placeholder (output=None) and
explanation both fine. Frozen ML core byte-identical.

---

# Forecast STEP 3 — baseline + backtest harness (TESTED) + FLAML wrapper (UNTESTED)

Added to `forecast.py`. No ML-core change.

## 3a — Backtest harness + naive baseline — FULLY TESTED
- `backtest(values, forecaster, horizon, n_folds)` — rolling-origin, TIME-ORDERED, never
  shuffles. Returns per-fold + overall MAE/RMSE/MAPE and the residuals (for Step-4 intervals).
- `naive_forecaster` (mandatory, decision #4) and `seasonal_naive_forecaster(period)`.
- A `forecaster` is any `fn(history_values, horizon) -> list`. The harness knows nothing about
  FLAML; baseline and model plug in through this one signature.

Verified:
- **No future leakage** — a spy confirms the forecaster only ever sees the growing prefix;
  train lengths [60,73,86,100,113] all < n=120.
- An oracle (linear extrapolation) beats naive; seasonal-naive crushes naive on purely seasonal
  data (1.02 vs 15.84) and correctly loses on trend-dominated data (that is real behaviour, not
  a bug).
- Too-short series -> honest error, no fabricated score.

## 3b — FLAML ts_forecast wrapper — **NOT EXECUTED HERE**
`make_flaml_forecaster()` and `run_forecast_evaluation()` are written against FLAML's documented
ts_forecast API but flaml[ts_forecast] is NOT installed and the network is offline in the build
environment, so the FLAML branch has NEVER RUN. It is isolated behind the tested harness:
- the naive baseline is always backtested (decision #4);
- if FLAML is absent or errors, the baseline result is still returned and the failure is
  recorded in `notes` — never a fabricated model score;
- a stand-in forecaster was injected to prove the glue + comparison path works
  (model beats naive by 2.33% in that test).

**Must be validated on a machine with FLAML before the FLAML numbers are trusted.**
The internal logic of make_flaml_forecaster (dataframe build, automl.fit args, predict) is the
specific untested part.

NOT built: prediction intervals (Step 4), forecast quality gate, ForecastResult wiring, visuals,
report, LLM narrative.

---

# Step 3b validation — tooling shipped, awaiting user runtime confirmation

I cannot validate FLAML here (not installed, network offline). Shipped
`validate_flaml_forecast.py` for the user to run on a machine with FLAML. It checks four
assumptions independently so a failure points at the exact line:

1. flaml imports AND the ts_forecast task actually runs (catches the missing-extras case);
2. make_flaml_forecaster returns exactly `horizon` finite floats;
3. it plugs into the tested backtest harness and collects residuals;
4. run_forecast_evaluation produces a naive-vs-model comparison.

The validator fails gracefully when FLAML is absent and points to the fix.

**requirements.txt corrected:** was `flaml[automl]` (does NOT pull the time-series backends),
now `flaml[automl,ts_forecast]`. This was a real install blocker, not a guess — the automl
extra alone cannot run task="ts_forecast".

`make_flaml_forecaster` itself is UNCHANGED — it has never run, so any edit from me would be a
guess. The validator runs it as-is; if a CHECK fails, that output tells us the real API shape to
correct against.

STATUS: Step 3a approved. Step 3b pending the four MAE numbers from the user's run.

---

# Forecast STEP 4 — intervals + quality gate + ForecastResult wiring (tested)

Added to `forecast.py`. No ML-core change. FLAML not needed for these tests (verified with a
stand-in forecaster on the tested harness).

## Prediction intervals — EMPIRICAL, not a distribution
- `backtest()` now tags each residual with its horizon step (`residuals_by_step`), a small
  backward-compatible addition; 3a still passes (35 residuals, steps 1..7).
- `build_intervals(point, residuals_by_step, level)` uses CUMULATIVE residuals up to step h:
  the band at horizon h is built from the errors of steps 1..h, so it rests on a larger, less
  noisy sample and grows with the horizon. Bands come from residual quantiles, never a normal
  assumption.
- Note: bands are not forced to be strictly monotone. On a model that is very accurate
  short-term, near-horizon residuals are genuinely small; forcing monotonicity would fake the
  numbers. We report what the residuals say.

## Forecast quality gate — enforces decision #4
- `forecast_quality_gate()` REJECTS if the model does not beat the naive baseline (or if no
  model score exists). Warns (does not reject) on poor interval coverage.
- Verified: good stand-in model -> passed (coverage 77% vs nominal 80%); bad model that predicts
  the first value -> REJECTED with reason "Model MAE (125.6) does not beat naive (9.048)".

## Forecast confidence — SEPARATE score (decision #2)
- `_forecast_confidence()` blends skill-vs-naive, interval calibration, and history adequacy.
  Independent of the tabular ConfidenceEngine.

## Wiring (decision #5)
- `assemble_forecast_result()` produces a complete `ForecastResult`: history, forecast with
  intervals, backtest, baseline, model, quality_gate, confidence, run_log. `success` is tied to
  the gate — a model that loses to naive yields `success=False` with the reason in `error`.

NOT built: visuals (Step 5), structured report + LLM narrative (Step 6/7). No LLM anywhere yet.

---

# Forecast STEP 5 — Visuals (tested). No LLM, no computation.

New file `forecast_viz.py`. Reads a ForecastResult and returns Plotly figures. It computes no
metric, fits no model, calls no LLM — every plotted value is already a number in ForecastResult
(decision #5). `forecast.py` itself is UNTOUCHED; the ML core is byte-identical.

## Four figures, each bound to specific fields
1. `history_vs_forecast` -> history + forecast (yhat/lower/upper), with a "today" divider.
2. `confidence_band` -> the interval alone; title shows empirical coverage from backtest.
3. `backtest_chart` -> per-fold out-of-sample MAE, with the overall MAE line.
4. `model_vs_naive` -> model MAE vs the mandatory naive baseline; colour flips if it loses.

`all_figures(result)` returns all four keyed by name for the UI.

## Verified — every plotted value traces back to ForecastResult
Built a real ForecastResult through Steps 1->4 (stand-in model) and checked:
- forecast points plotted == result.forecast yhat (exact);
- naive/model bars == result.baseline / result.backtest (exact);
- backtest bars == fold MAEs (exact);
- confidence band edges == result.forecast lower/upper (exact);
- an empty/rejected result -> figure with an explicit annotation and NO traces (no fabrication).

Tested with a lightweight plotly stand-in because plotly is offline in the build env; on the
user's machine plotly is already a dependency (the app uses it). The logic verified is
value-provenance, which is stand-in-independent.

## NOT wired into the UI yet
`app.py` Forecast tab is still the placeholder. Rendering these figures there is a separate,
explicit step (it needs a real ForecastResult in the app, which requires the router's
ForecastTool to move beyond placeholder — a decision for the user).

NOT built: structured report (Step 6), LLM narrative + validator (Step 7).

---

# Forecast STEP 6 — Structured Report (tested). NO LLM.

New file `forecast_report.py`. Builds a structured report from a ForecastResult. This is the
DETERMINISTIC report that works with no API key (ordering principle 0.0). The LLM narrative is
the SEPARATE Step 7 — not imported or called here. `forecast.py`, `forecast_viz.py` and the ML
core are all byte-identical.

## Sections (all read from ForecastResult, decision #5)
- **summary**: target, time axis, frequency, horizon, history length, model, next-period
  forecast with interval.
- **reliability**: model MAE/RMSE/MAPE, naive MAE, improvement vs naive, beats_naive,
  forecast confidence.
- **coverage**: empirical interval coverage from backtest.
- **risks**: warnings + structural flags (horizon vs history, mis-calibrated coverage).
- **quality_gate**: passed, reasons, warnings.

`build_report(result)` -> dict (JSON/UI). `report_to_markdown(report)` -> text (download).
Both are pure formatting; no metric is computed, no number invented.

## Verified — every value traces to ForecastResult
naive_mae, model_mae, headline yhat, coverage, gate.passed, confidence — all exact matches to
the source result. Rejected forecast -> report shows success=False with the gate reason and
beats_naive=False. Markdown renders cleanly.

Tested with the plotly stand-in (package import pulls forecast_viz). Report logic itself has no
plotly dependency.

## Ready to wire (next, with user approval)
Steps 5 (visuals) + 6 (report) can now be dropped into the app's Forecast tab together, driven
by one real ForecastResult — which requires moving the router's ForecastTool beyond placeholder.

NOT built: Step 7 (LLM narrative + number-grounding validator, OpenAI, last of all).

---

# Forecast WIRED INTO THE UI (Steps 1-6 live). Two files changed: router.py, app.py.

ML core + all six forecast layers (forecast.py, forecast_viz.py, forecast_report.py) are
byte-identical. Only the router's ForecastTool and the app's Forecast tab changed.

## router.py — ForecastTool is no longer a placeholder
Runs the full pipeline: prepare_series (1) -> prepare_temporal (2) -> run_forecast_evaluation
(3) -> assemble_forecast_result (4) -> build_report (6). Output is ALWAYS a dict
{forecast_result, report}, so a rejection or failure never crashes the caller (same contract
fix as the prediction path).
- Gap strategy is REQUIRED (Step-2 decision): if the caller omits it, the tool returns an error
  asking for one — it does not assume a default.
- If FLAML is unavailable, the point forecast falls back to the naive baseline and says so in
  the events; no fabricated model numbers.
- `route()` gained a `forecast_kwargs` param to pass strategy/time_col/horizon/season through.

## app.py — Forecast tab (renamed from "LLM Forecast")
- User picks time column (X), target (Y), gap strategy (required, no default), horizon, season.
- "Run Forecast" routes through ForecastTool, then renders the four forecast_viz figures and the
  forecast_report markdown, with a download button.
- The legacy tabular LLM analysis is kept below, clearly separated.
- Forecast state guarded via isinstance(output, dict); a missing strategy shows an error, no crash.

## Verified (stand-in FLAML, plotly stand-in)
Full tab flow: route -> ok -> 4 figures built -> report markdown (828 chars, download-ready).
Missing strategy -> error, dict output, no crash. Prediction (R2 0.9852) and explanation still
work — no regression.

## Requires on the user's machine
- FLAML installed (validated in 3b) for real model forecasts; otherwise the tab shows the naive
  forecast and labels it as such.
- plotly (already a dependency) for the charts.

NOT built: Step 7 (LLM narrative + number-grounding validator, OpenAI, last).

---

# Forecast made an INDEPENDENT path (UX fix). One file changed: app.py.

## Problem
The whole UI (all tabs, including Forecast) was hidden behind `if result is not None:` — i.e.
a SUCCESSFUL tabular Prediction run. A plain `date,sales` file cannot be used for tabular
prediction (the date column is dropped, training becomes impossible), so the user could never
reach Forecast for exactly the kind of data forecasting is for.

## Fix (app.py only — no engine/router/forecast-layer change)
- Top-level tabs added: **"Prediction Engine"** and **"Forecast (independent)"**.
- `render_forecast(df)` extracted as a standalone function that needs ONLY the uploaded df and
  the router. It creates its own router if none exists and never touches `engine_result`.
- The Forecast tab renders as soon as data is uploaded — no prediction run required.
- The tabular prediction display stays under the Prediction Engine tab, still gated on `result`.
- The old inner "LLM Forecast" sub-tab keeps only the legacy tabular LLM explanation, with a
  note pointing to the top-level Forecast tab.

## Verified
- app.py executes top-to-bottom under a Streamlit mock (df present, result=None) — no structural
  or indentation error after the re-nesting.
- `date,sales` forecasts successfully with result=None (history 120, gate passed, model beats
  naive) — the exact case that was previously unreachable.
- Tabular prediction unchanged (R2 0.9819, predict works). Every non-app file byte-identical.

All ML core + forecast layers (forecast.py, forecast_viz.py, forecast_report.py) and router.py
are UNTOUCHED. Only app.py changed.

STATUS: Forecast is now an independent, usable path. Step 7 (LLM narrative + validator) still
not built.

---

# Forecast STEP 7 — LLM narrative + number-grounding validator (OpenAI). LAST step.

New file `forecast_narrative.py`. `app.py` gets a narrative section; `__init__.py` exports it.
ML core + all other forecast layers + router are byte-identical.

## Ordering principle 0.0 — enforced
The narrative runs AFTER the full ForecastResult and report exist, and only narrates. The
forecast core has no LLM import. With no OPENAI_API_KEY the pipeline still yields the complete
ForecastResult, visuals and structured report — only the prose is skipped. Verified.

## Number-grounding validator (NO LLM, pure, tested)
- `collect_grounded_numbers(result)` gathers every number the narrative may state (forecast
  points, MAE/RMSE/MAPE, naive, coverage, confidence, fold MAEs, derived improvement %, nominal
  level).
- `validate_numbers(text, result)` extracts every number from the prose and checks each against
  that set (relative tolerance for rounding). Returns which are grounded and which are not.
- Verified: clean narrative -> all_grounded True; hallucinated ("23.7%", "5000") -> flagged;
  "46.18%" stated as "46%" -> accepted (tolerance).
- Fixed a regex bug where "5000" split into "500","0"; whole numbers and comma-grouped numbers
  (59,365 / 1,234,567) now parse correctly.

## Narrative (OpenAI, optional)
- `generate_narrative(result, report)`: OpenAI system prompt forbids inventing numbers; the
  output is validated; if any number is ungrounded the narrative is marked UNVERIFIED and shown
  with a warning rather than as fact. Never blocks the structured report.
- Provider is OpenAI only; no Anthropic anywhere.

## UI
Forecast tab shows the narrative section after the report: a key field, a Generate button, and
either a "verified" success or an "unverified" error with the ungrounded tokens listed.

ALL SEVEN STEPS OF THE FORECAST ENGINE ARE NOW COMPLETE.
Steps 1-6 validated end-to-end with real FLAML (46.18% improvement over naive on 1688 periods).
Step 7 validator tested without an API; the OpenAI call itself runs on the user's machine.
