"""
app.py — "Smart Prediction Engine" Interface
=====================================
Local run:
    pip install -r requirements.txt
    streamlit run app.py

To enable analytical LLM Forecast mode (optional):
    export OPENAI_API_KEY="sk-..."   # macOS/Linux
    setx OPENAI_API_KEY "sk-..."     # Windows
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from prediction_engine import PredictionEngine, LLMForecastAgent, ReActRouter, Intent
from prediction_engine.form import build_form_spec

st.set_page_config(page_title="Smart Prediction Engine", layout="wide")

# ---------------------------------------------------------------- CSS (RTL) --
st.markdown("""
<style>
html, body, [class*="css"] { direction: ltr; text-align: left; }
[data-testid="stMetricLabel"] { direction: ltr; }
.stDataFrame { direction: ltr; }
</style>
""", unsafe_allow_html=True)

if "engine_result" not in st.session_state:
    st.session_state.engine_result = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "router" not in st.session_state:
    st.session_state.router = ReActRouter()
if "events" not in st.session_state:
    st.session_state.events = []          # structured events, newest run appended
if "agent_reply" not in st.session_state:
    st.session_state.agent_reply = None
for _k in ("forecast_result","forecast_report","forecast_msg"):
    if _k not in st.session_state:
        st.session_state[_k] = None

st.title("Smart Prediction Engine")
st.caption("General Multi-Agent Prediction System — Profiling → Preprocessing → FLAML Training → Triple-Threat Evaluation → Confidence Score")

# ============================================================== SIDEBAR =====
with st.sidebar:
    st.header("1. Data")
    source = st.radio("Data Source", ["Use Sample Data (Real Estate)", "Upload CSV File"])

    if source == "Upload CSV File":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            st.session_state.raw_df = pd.read_csv(uploaded)
    else:
        st.session_state.raw_df = pd.read_csv("sample_data_real_estate.csv")
        st.info("Sample data: Real estate prices in Riyadh (600 rows).")

    df = st.session_state.raw_df

    target_col = None
    if df is not None:
        st.header("2. Target Column")
        default_idx = len(df.columns) - 1
        target_col = st.selectbox("Select Target Column", options=list(df.columns), index=default_idx)

        st.header("3. Execution")
        run_clicked = st.button("Run Prediction Engine", type="primary", use_container_width=True)
    else:
        run_clicked = False
        st.warning("Upload your data or use sample data to start.")

# ============================================================== RUN ENGINE ==
if df is not None and run_clicked:
    with st.spinner("Executing full pipeline: Profiling → Preprocessing → Training → Evaluation ..."):
        engine = PredictionEngine()
        router = ReActRouter(engine=engine)
        tool_result = router.route(
            message="run the prediction pipeline",
            df=df, target_col=target_col,
            explicit_intent=Intent.PREDICTION,      # the button IS the intent
        )
        st.session_state.events = tool_result.events
        # output is a dict on both success and error paths; guard anyway in case a future
        # tool returns a different shape, so the app never crashes on a failed run.
        out = tool_result.output
        result = out.get("engine_result") if isinstance(out, dict) else None
        st.session_state.engine_result = result
        st.session_state.engine = engine
        st.session_state.router = router

result = st.session_state.engine_result

# ============================================================== FORECAST (independent) ==
def render_forecast(raw_df):
    """Time-series forecast — a standalone path. Needs only the uploaded data and the router;
    it does NOT require the tabular Prediction Engine to have run. This is what lets a plain
    date,sales file be forecast even though it cannot be used for tabular prediction."""
    st.markdown("### Time-Series Forecast")
    st.caption("Independent of the Prediction Engine. Upload any date + measure series and "
               "forecast it directly. Numbers come from the model; charts and report read them.")

    if "router" not in st.session_state or st.session_state.router is None:
        st.session_state.router = ReActRouter()

    fcol1, fcol2, fcol3 = st.columns(3)
    time_choice = fcol1.selectbox("Time column (X)", options=list(raw_df.columns))
    y_choice = fcol2.selectbox("Target to forecast (Y)",
                               options=[c for c in raw_df.columns if c != time_choice])
    strategy = fcol3.selectbox("Gap strategy",
                               options=["none", "ffill", "linear", "zero", "mean"], index=1,
                               help="How to fill missing periods on the regular grid.")
    hcol1, hcol2 = st.columns(2)
    horizon_in = hcol1.number_input("Horizon (periods ahead, 0 = auto)",
                                    min_value=0, value=0, step=1)
    season_in = hcol2.number_input("Seasonal period (0 = none)",
                                   min_value=0, value=0, step=1)

    if st.button("Run Forecast", type="primary"):
        fk = {"strategy": strategy, "time_col": time_choice,
              "horizon": (int(horizon_in) or None),
              "seasonal_period": (int(season_in) or None)}
        with st.spinner("Forecasting..."):
            ftr = st.session_state.router.route(
                message="forecast", df=raw_df, target_col=y_choice,
                explicit_intent=Intent.FORECAST, forecast_kwargs=fk)
        st.session_state.events = ftr.events
        st.session_state.forecast_result = ftr.output.get("forecast_result") \
            if isinstance(ftr.output, dict) else None
        st.session_state.forecast_report = ftr.output.get("report") \
            if isinstance(ftr.output, dict) else None
        st.session_state.forecast_msg = (ftr.status, ftr.message)

    fr = st.session_state.get("forecast_result")
    fmsg = st.session_state.get("forecast_msg")
    if fmsg and fmsg[0] != "ok":
        st.error(f"Forecast: {fmsg[1]}")
    if fr is not None:
        from prediction_engine import forecast_viz as _viz
        from prediction_engine.forecast_report import report_to_markdown
        figs = _viz.all_figures(fr)
        st.plotly_chart(figs["history_vs_forecast"], use_container_width=True)
        g1, g2 = st.columns(2)
        g1.plotly_chart(figs["confidence_band"], use_container_width=True)
        g2.plotly_chart(figs["model_vs_naive"], use_container_width=True)
        st.plotly_chart(figs["backtest_chart"], use_container_width=True)
        rep = st.session_state.get("forecast_report")
        if rep:
            st.markdown("#### Structured Report")
            md = report_to_markdown(rep)
            st.markdown(md)
            st.download_button("Download report (markdown)", md, file_name="forecast_report.md")


# ============================================================== TOP-LEVEL MODE ==
if df is not None:
    _tab_pred, _tab_fcst = st.tabs(["Prediction Engine", "Forecast (independent)"])
    with _tab_fcst:
        render_forecast(df)
else:
    _tab_pred = None

# ============================================================== DISPLAY =====
if df is not None and result is None:
    with _tab_pred:
        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.info("Click 'Run Prediction Engine' from the sidebar to begin. "
                "For a plain date + measure series, use the Forecast tab instead — "
                "it does not need the Prediction Engine.")

if result is not None:
    _pred_ctx = _tab_pred if _tab_pred is not None else st.container()
    with _pred_ctx:
        if not result.success:
            st.error(f"Execution failed: {result.error}")
            with st.expander("Full Run Log"):
                for l in result.run_log:
                    st.text(l)
            st.stop()

        fp = result.fingerprint
        ev = result.evaluation
        conf = result.confidence

        # ---- Top banner ----
        gate_status = "Passed Quality Gate" if ev.passed_quality_gate else "Rejected — Weak Statistical Correlation"
        st.subheader(f"Result: {gate_status}")
        st.write(f"**Recommendation:** {result.recommendation}")

        # ---- Top metrics row ----
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Task Type", "Classification" if fp.task_type == "classification" else "Regression")
        c2.metric("Best Model", result.training.best_estimator)
        c3.metric("Self-Correction Attempts", result.attempts)
        if fp.task_type == "regression":
            c4.metric("R² / MAE", f"{ev.statistical['r2']} / {ev.statistical['mae']:,.0f}")
        else:
            c4.metric("Accuracy / F1", f"{ev.statistical['accuracy']} / {ev.statistical['f1_weighted']}")

        st.divider()

        tab_fp, tab_prep, tab_eval, tab_predict, tab_forecast, tab_log = st.tabs(
            ["Data Fingerprint", "Preprocessing", "Triple-Threat Evaluation", "Predict", "Forecast", "Run Log"]
        )

        # ---------------- Data Fingerprint tab ----------------
        with tab_fp:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Rows", fp.n_rows)
                st.metric("Columns", fp.n_cols)
                st.metric("Overall Missing Ratio", f"{fp.overall_missing_pct:.2f}%")
                st.metric("Duplicate Rows", fp.duplicate_rows)
            with col2:
                if fp.correlated_with_target:
                    corr_df = pd.DataFrame(
                        {"feature": list(fp.correlated_with_target.keys()),
                         "correlation": list(fp.correlated_with_target.values())}
                    )
                    fig = px.bar(corr_df, x="correlation", y="feature", orientation="h",
                                 title="Linear Correlation with Target", color="correlation",
                                 color_continuous_scale="RdBu", range_color=[-1, 1])
                    st.plotly_chart(fig, use_container_width=True)

            if fp.warnings:
                for w in fp.warnings:
                    st.warning(w)

            st.markdown("**Column Details:**")
            cols_rows = []
            for c in fp.columns:
                cols_rows.append({
                    "Column": c.name, "Type": c.dtype, "Role": c.role,
                    "% Missing": round(c.missing_pct, 2), "Cardinality": c.cardinality,
                })
            st.dataframe(pd.DataFrame(cols_rows), use_container_width=True)

        # ---------------- Preprocessing tab ----------------
        with tab_prep:
            st.markdown("**Executed Preprocessing Steps:**")
            for step in result.preprocessing.log:
                st.write(f"- {step}")
            if result.preprocessing.dropped_cols:
                st.info(f"Dropped columns: {', '.join(result.preprocessing.dropped_cols)}")
            st.markdown("**Sample of Preprocessed Data:**")
            st.dataframe(result.preprocessing.df.head(10), use_container_width=True)

        # ---------------- Triple-Threat Evaluation tab ----------------
        with tab_eval:
            e1, e2, e3 = st.columns(3)

            with e1:
                st.markdown("### 1. Statistical")
                for k, v in ev.statistical.items():
                    st.metric(k.upper(), v)

            with e2:
                st.markdown("### 2. Stability (CV)")
                if ev.stability.get("cv_mean") is not None:
                    st.metric("CV Mean Score", ev.stability["cv_mean"])
                    st.metric("Std Deviation", ev.stability["cv_std"])
                    fold_df = pd.DataFrame({
                        "Fold": [f"Fold {i+1}" for i in range(len(ev.stability["fold_scores"]))],
                        "Score": ev.stability["fold_scores"],
                    })
                    st.plotly_chart(px.bar(fold_df, x="Fold", y="Score", title="Cross-Validation Performance"),
                                    use_container_width=True)
                else:
                    st.warning(f"Could not calculate stability: {ev.stability.get('error', 'Unknown')}")

            with e3:
                st.markdown("### 3. Explainability (SHAP)")
                top_feats = ev.explainability.get("top_features", [])
                if top_feats:
                    shap_df = pd.DataFrame(top_feats)
                    fig = px.bar(shap_df.sort_values("importance"), x="importance", y="feature",
                                 orientation="h", title="Top Factors (|SHAP|)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Could not calculate SHAP: {ev.explainability.get('error', 'Unknown')}")

            st.divider()
            st.markdown("### Confidence Score")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=conf.score * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2563eb"},
                    "steps": [
                        {"range": [0, 50], "color": "#fecaca"},
                        {"range": [50, 75], "color": "#fef08a"},
                        {"range": [75, 100], "color": "#bbf7d0"},
                    ],
                },
                title={"text": f"Confidence: {conf.label}"},
            ))
            fig_gauge.update_layout(height=280)
            st.plotly_chart(fig_gauge, use_container_width=True)
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Model Quality", conf.breakdown["quality"])
            bc2.metric("Stability", conf.breakdown["stability"])
            bc3.metric("Data Adequacy", conf.breakdown["data_adequacy"])

            if not ev.passed_quality_gate:
                st.error("Rejection Reasons:\n" + "\n".join(f"- {r}" for r in ev.rejection_reasons))

        # ---------------- Predict tab ----------------
        with tab_predict:
            st.markdown("**Predict** — Fill the fields that matter. The rest are pre-filled.")
            raw_df = st.session_state.raw_df

            spec = build_form_spec(result, raw_df)

            st.caption(
                f"The model's signal is concentrated in {len(spec.required)} of "
                f"{len(spec.required) + len(spec.optional)} usable columns "
                f"({spec.coverage:.0%} of total SHAP importance). "
                "The rest default to the median / most common value."
            )
            if spec.excluded:
                st.caption(f"Not asked (dropped by the pipeline): {', '.join(spec.excluded)}")

            with st.form("predict_form"):
                input_values = dict(spec.defaults())   # start from the defaults

                st.markdown("##### Required")
                cols = st.columns(3)
                for i, f in enumerate(spec.required):
                    w = cols[i % 3]
                    if f.role == "numeric":
                        input_values[f.name] = w.number_input(
                            f"{f.name}  ·  {f.importance:.3f}", value=float(f.default))
                    else:
                        opts = f.options or [f.default]
                        idx = opts.index(f.default) if f.default in opts else 0
                        input_values[f.name] = w.selectbox(
                            f"{f.name}  ·  {f.importance:.3f}", options=opts, index=idx)

                if spec.optional:
                    with st.expander(f"Optional — {len(spec.optional)} fields (already filled in)"):
                        ocols = st.columns(3)
                        for i, f in enumerate(spec.optional):
                            w = ocols[i % 3]
                            if f.role == "numeric":
                                input_values[f.name] = w.number_input(
                                    f"{f.name}  ·  {f.importance:.3f}",
                                    value=float(f.default), key=f"opt_{f.name}")
                            else:
                                opts = f.options or [f.default]
                                idx = opts.index(f.default) if f.default in opts else 0
                                input_values[f.name] = w.selectbox(
                                    f"{f.name}  ·  {f.importance:.3f}", options=opts,
                                    index=idx, key=f"opt_{f.name}")

                submitted = st.form_submit_button("Calculate Prediction", type="primary")

            if submitted:
                new_row = pd.DataFrame([input_values])
                try:
                    tr = st.session_state.router.route(
                        message="predict this record",
                        df=st.session_state.raw_df, target_col=fp.target_col,
                        explicit_intent=Intent.PREDICTION,
                        new_rows=new_row,
                        engine_result=result,           # reuse the trained model, do not retrain
                    )
                    st.session_state.events = tr.events
                    preds = tr.output.get("predictions") if isinstance(tr.output, dict) else None
                    if tr.status != "ok" or not preds:
                        st.error(f"Could not calculate prediction: {tr.message}")
                    else:
                        pred_value = preds[0]
                        if fp.task_type == "classification":
                            if "__target__" in result.preprocessing.encoders:
                                le = result.preprocessing.encoders["__target__"]
                                pred_value = le.inverse_transform([int(round(pred_value))])[0]
                            st.success(f"Predicted Class: **{pred_value}**")
                        else:
                            st.success(f"Predicted Value for **{fp.target_col}**: **{pred_value:,.2f}**")
                        st.caption(f"Based on model general confidence: {conf.score} ({conf.label})")
                        st.caption("Routed through the agent — see the Run Log tab for the events.")
                except Exception as e:
                    st.error(f"Could not calculate prediction: {e}")

        # ---------------- Forecast tab ----------------
        with tab_forecast:
            st.info("The time-series forecast lives in the top-level **Forecast (independent)** "
                    "tab — it does not need the Prediction Engine. This section is the tabular "
                    "LLM explanation of the model trained above.")

            # ---------------- Legacy tabular LLM analysis ----------------
            st.markdown("### LLM Forecast (tabular explanation)")
            st.caption("Descriptive analysis of the tabular model above — separate from the "
                       "time-series forecast.")
            api_key_input = st.text_input("OPENAI_API_KEY (optional, not saved)", type="password",
                                          value=os.environ.get("OPENAI_API_KEY", ""))
            if st.button("Generate LLM Forecast Analysis"):
                if api_key_input:
                    os.environ["OPENAI_API_KEY"] = api_key_input
                agent = LLMForecastAgent()
                if not agent.available:
                    st.warning(agent.error)
                else:
                    with st.spinner("Generating analysis..."):
                        text = agent.forecast(fp.to_dict(), ev.__dict__ if hasattr(ev, "__dict__") else {}, conf.__dict__)
                    st.write(text)

        # ---------------- Run log tab ----------------
        with tab_log:
            st.markdown("### Ask the Agent")
            st.caption("Type what you want. The router detects the intent and picks the tool. "
                       "If it can't tell, it asks instead of guessing.")
            user_msg = st.text_input("Your request", placeholder="e.g. what's the price of a 300sqm house?")
            if st.button("Route Request"):
                tr = st.session_state.router.route(
                    message=user_msg,
                    df=st.session_state.raw_df, target_col=fp.target_col,
                    engine_result=result,          # never retrains
                )
                st.session_state.events = tr.events
                st.session_state.agent_reply = tr

            tr = st.session_state.agent_reply
            if tr is not None:
                if tr.status == "error" and tr.tool == "none":
                    st.warning(tr.message)          # unclear intent -> ask the user
                    c1, c2, c3 = st.columns(3)
                    if c1.button("Prediction"):
                        st.session_state.agent_reply = st.session_state.router.route(
                            message=user_msg, df=st.session_state.raw_df, target_col=fp.target_col,
                            explicit_intent=Intent.PREDICTION, engine_result=result)
                        st.session_state.events = st.session_state.agent_reply.events
                    if c2.button("Forecast"):
                        st.session_state.agent_reply = st.session_state.router.route(
                            message=user_msg, df=st.session_state.raw_df, target_col=fp.target_col,
                            explicit_intent=Intent.FORECAST)
                        st.session_state.events = st.session_state.agent_reply.events
                    if c3.button("Explanation"):
                        st.session_state.agent_reply = st.session_state.router.route(
                            message=user_msg, df=st.session_state.raw_df, target_col=fp.target_col,
                            explicit_intent=Intent.EXPLANATION, engine_result=result)
                        st.session_state.events = st.session_state.agent_reply.events
                elif tr.status == "not_implemented":
                    st.info(tr.message)             # forecast placeholder — honest, no fake numbers
                elif tr.status == "ok" and tr.tool == "explanation":
                    st.json(tr.output)
                elif tr.status == "ok" and tr.tool == "prediction":
                    st.success("Routed to the prediction pipeline. Use the Predict tab to enter values.")
                else:
                    st.error(tr.message)

            st.divider()
            st.markdown("### Structured Events")
            if st.session_state.events:
                ev_rows = [{"stage": e["stage"], "action": e["action"],
                            "status": e["status"], "reason": e["reason"]}
                           for e in st.session_state.events]
                st.dataframe(pd.DataFrame(ev_rows), use_container_width=True)
                with st.expander("Raw events (JSON)"):
                    st.json(st.session_state.events)
            else:
                st.caption("No events yet.")

            st.divider()
            st.markdown("### Engine Run Log (deterministic core)")
            for l in result.run_log:
                st.text(l)