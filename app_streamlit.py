# Streamlit app — updated to fetch latest ingestion from Hopsworks and run predictions in-process
# Includes:
#  - Plotly interactive charts (hover) with forecast (next 3 days) for pollutants (history plotting removed)
#  - SHAP summarizer & table + optional beeswarm (runs only when shap and model are available and user opts in)
#  - Safe fallbacks to synthetic history and synthetic AQI forecast when Hopsworks/predictor unavailable

import os
import datetime
import glob
import traceback
from typing import Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Plotly (preferred for interactive charts)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# Hopsworks and dotenv (optional)
try:
    import hopsworks
except Exception:
    hopsworks = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Try to import the predictor module (prefer in-process)
try:
    import predict as predictor
except Exception:
    predictor = None

# Try to use fetcher & compute functions for live current AQI
try:
    from fetcher import fetch_openweather_current
except Exception:
    fetch_openweather_current = None

try:
    from compute_aqi import compute_aqi_row, aqi_category
except Exception:
    compute_aqi_row = None
    aqi_category = None

st.set_page_config(page_title="Karachi AQI Forecast", page_icon="🌬️", layout="wide")


# ---------- compatibility helpers ----------
def safe_rerun():
    try:
        if hasattr(st, "experimental_rerun"):
            try:
                st.experimental_rerun()
                return
            except Exception:
                pass
    except Exception:
        pass
    try:
        try:
            from streamlit.runtime.scriptrunner.script_runner import RerunException  # type: ignore
        except Exception:
            from streamlit.script_runner import RerunException  # type: ignore
        raise RerunException()
    except Exception:
        pass
    try:
        st.session_state["_copilot_force_rerun_flag"] = not st.session_state.get("_copilot_force_rerun_flag", False)
    except Exception:
        pass
    try:
        st.stop()
    except Exception:
        raise RuntimeError("Could not trigger Streamlit rerun; please reload the page manually.")


# ---------- small helpers ----------
def find_first_local(pattern):
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def find_local_model(models_dir: str = "models") -> Tuple[Optional[object], Optional[str]]:
    patterns = [
        os.path.join(models_dir, "aqi_predictor.pkl"),
        os.path.join(models_dir, "rf_aqi_predictor_*.pkl"),
        os.path.join(models_dir, "xgb_aqi_predictor_*.pkl"),
        os.path.join(models_dir, "*aqi_predictor*.pkl"),
        os.path.join(models_dir, "*.pkl"),
    ]
    seen = []
    for pat in patterns:
        for fp in glob.glob(pat):
            if fp not in seen:
                seen.append(fp)
    for fp in seen:
        try:
            mdl = joblib.load(fp)
            return mdl, fp
        except Exception:
            continue
    return None, None


# ---------- Hopsworks caching helpers (load once per server) ----------
@st.cache_resource
def get_hopsworks_project():
    try:
        if hopsworks is None:
            return None
        if not os.getenv("HOPSWORKS_API_KEY"):
            return None
        try:
            proj = hopsworks.login()
            return proj
        except Exception as e:
            print("Hopsworks login failed:", e)
            return None
    except Exception as e:
        print("Unexpected error in get_hopsworks_project:", e)
        return None


@st.cache_resource
def load_model_and_artifacts_hopsworks(model_name: Optional[str] = None, model_version: Optional[int] = None, allowed_patterns=None):
    # Fixed: use datetime.datetime.now() (no erroneous datetime.datetime.datetime)
    start = datetime.datetime.now()
    project = get_hopsworks_project()
    if project is None:
        return None, {}, None, None, 0.0

    model_name = model_name or os.getenv("HOPSWORKS_MODEL_NAME", "aqi_predictor")
    model_version = int(model_version or os.getenv("HOPSWORKS_MODEL_VERSION", "1"))
    if allowed_patterns is None:
        allowed_patterns = ["feature_cols", "feature_medians", "scaler", "feature_names", "rf_feature_cols", "pipeline", "medians"]

    try:
        mr = project.get_model_registry()
        entry = mr.get_model(model_name, version=model_version)
        if entry is None:
            print(f"Hopsworks: model {model_name} v{model_version} not found in registry")
            return None, {}, None, None, 0.0

        try:
            folder = entry.download()
        except Exception:
            print("Hopsworks model download failed or timed out")
            return None, {}, None, None, 0.0

        download_time = (datetime.datetime.now() - start).total_seconds()
        model_obj = None
        model_path = None
        artifacts = {}
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".pkl") or f.endswith(".joblib"):
                    p = os.path.join(root, f)
                    name_lower = f.lower()
                    try:
                        obj = joblib.load(p)
                        if model_obj is None and (hasattr(obj, "predict") or hasattr(obj, "predict_proba") or hasattr(obj, "transform")):
                            model_obj = obj
                            model_path = p
                            artifacts[f] = obj
                            continue
                        for pat in allowed_patterns:
                            if pat in name_lower:
                                artifacts[f] = obj
                                break
                        if f not in artifacts:
                            artifacts[f] = obj
                    except Exception:
                        artifacts[f] = None
        return model_obj, artifacts, model_path, folder, download_time
    except Exception as e:
        print("Error loading model from Hopsworks:", e)
        traceback.print_exc()
        return None, {}, None, None, 0.0


def run_with_timeout(fn, timeout=10, *args, **kwargs):
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise TimeoutError(f"call timed out after {timeout}s")
        except Exception:
            raise


@st.cache_data(ttl=600)
def get_latest_data_from_fg(timeout_seconds: int = 30):
    if hopsworks is None:
        now = pd.Timestamp.now(tz="UTC")
        idx = pd.date_range(end=now, periods=168, freq="H", tz="UTC")
        return pd.DataFrame({
            "time": idx,
            "pm2_5": np.random.uniform(10, 80, len(idx)),
            "pm10": np.random.uniform(20, 100, len(idx)),
            "standard_aqi": np.random.uniform(30, 150, len(idx)),
            "AQI": np.random.randint(1, 6, len(idx)),
        })
    def _read_fg():
        project = get_hopsworks_project()
        if project is None:
            raise RuntimeError("Hopsworks project unavailable")
        fs = project.get_feature_store()
        fg_name = os.getenv("HOPSWORKS_FEATURE_GROUP", "aqi_current")
        fg_version = int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "1"))
        fg = fs.get_feature_group(name=fg_name, version=fg_version)
        try:
            df = fg.select_all().read().sort_values("time", ascending=False).head(168)
        except Exception:
            df = fg.read().sort_values("time", ascending=False).head(168)
        return df.reset_index(drop=True)
    try:
        df = run_with_timeout(_read_fg, timeout_seconds)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        return df
    except TimeoutError as e:
      st.warning(f".")

    except Exception as e:
        st.warning(f"Failed reading Hopsworks feature group: {e}")
        traceback.print_exc()
    now = pd.Timestamp.now(tz="UTC")
    idx = pd.date_range(end=now, periods=168, freq="H", tz="UTC")
    return pd.DataFrame({
        "time": idx,
        "pm2_5": np.random.uniform(10, 80, len(idx)),
        "pm10": np.random.uniform(20, 100, len(idx)),
        "standard_aqi": np.random.uniform(30, 150, len(idx)),
        "AQI": np.random.randint(1, 6, len(idx)),
    })


# ---------- Forecast loader (IN-PROCESS predictor) ----------
@st.cache_data(ttl=300)
def load_forecast():
    try:
        if predictor is not None and hasattr(predictor, "get_predictions_for_dashboard"):
            try:
                ft, preds_us, preds_1to5 = predictor.get_predictions_for_dashboard(use_24h_model=False, prefer_hopsworks=True)
                df = pd.DataFrame({"time": pd.to_datetime(ft), "us_aqi": preds_us, "aqi_1to5": preds_1to5})
                df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
                try:
                    df.to_csv("predictions.csv", index=False)
                except Exception:
                    pass
                return df.sort_values("time").reset_index(drop=True)
            except Exception as e:
                print("[load_forecast] predictor.get_predictions_for_dashboard failed:", e)
                traceback.print_exc()
    except Exception:
        pass
    try:
        if os.path.exists("predictions.csv"):
            df = pd.read_csv("predictions.csv", parse_dates=["time"])
            if "US_AQI" in df.columns and "us_aqi" not in df.columns:
                df = df.rename(columns={"US_AQI": "us_aqi"})
            if "AQI_1to5" in df.columns and "aqi_1to5" not in df.columns:
                df = df.rename(columns={"AQI_1to5": "aqi_1to5"})
            try:
                df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            except Exception:
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
                df["time"] = df["time"].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
            if "us_aqi" not in df.columns:
                df["us_aqi"] = np.nan
            if "aqi_1to5" not in df.columns:
                df["aqi_1to5"] = df["us_aqi"].apply(lambda x: int(np.clip(round(x/50), 1, 5)) if pd.notna(x) else np.nan)
            return df.sort_values("time").reset_index(drop=True)
    except Exception as e:
        print("[load_forecast] Failed reading predictions.csv:", e)
    ft = pd.date_range(start=pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1), periods=72, freq="H", tz="UTC")
    preds_1to5 = np.clip(np.random.normal(3, 0.8, 72), 1, 5).astype(int)
    preds_us = preds_1to5 * 50
    df = pd.DataFrame({"time": ft, "us_aqi": preds_us, "aqi_1to5": preds_1to5})
    return df


def get_current_openweather_aqi():
    try:
        if fetch_openweather_current is None:
            return None
        lat = float(os.getenv("LATITUDE", "24.8607"))
        lon = float(os.getenv("LONGITUDE", "67.0011"))
        df = fetch_openweather_current(lat, lon, os.getenv("OPENWEATHER_API_KEY"))
        if df is None or df.empty:
            return None
        row = df.iloc[0].copy()
        try:
            row["time"] = pd.to_datetime(row.get("time", pd.Timestamp.now()), utc=True)
        except Exception:
            pass
        std = None
        cat = None
        try:
            if compute_aqi_row is not None:
                std = compute_aqi_row(row)
                if aqi_category is not None:
                    cat = aqi_category(std)
        except Exception:
            std = None
            cat = None
        return {"time": row.get("time"), "standard_aqi": std, "category": cat, "raw": row.to_dict()}
    except Exception:
        return None


def aqi_1to5_label_color(code):
    try:
        c = int(code)
    except Exception:
        return "N/A", "#777777"
    mapping = {
        1: ("Good", "#2ECC71"),
        2: ("Moderate", "#F1C40F"),
        3: ("Unhealthy for Sensitive", "#E67E22"),
        4: ("Unhealthy", "#E74C3C"),
        5: ("Very Unhealthy / Hazardous", "#8E44AD"),
    }
    return mapping.get(c, ("N/A", "#777777"))


def us_aqi_category_and_color(aqi_value):
    if pd.isna(aqi_value):
        return "N/A", "#777777"
    try:
        aqi = float(aqi_value)
    except Exception:
        return "N/A", "#777777"
    if aqi <= 50:
        return "Good", "#2ECC71"
    if aqi <= 100:
        return "Moderate", "#F1C40F"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#E67E22"
    if aqi <= 200:
        return "Unhealthy", "#E74C3C"
    if aqi <= 300:
        return "Very Unhealthy", "#8E44AD"
    return "Hazardous", "#7B241C"


# ---- SHAP summarizer helper ----
def summarize_shap(shap_exp, X, top_k=10):
    vals = shap_exp.values
    if isinstance(vals, list):
        vals_arr = np.array(vals[0])
    else:
        vals_arr = np.array(vals)
    if vals_arr.ndim > 2:
        vals_arr = vals_arr[:, :, 0]
    if vals_arr.ndim == 1:
        vals_arr = vals_arr.reshape(-1, 1)
    n_samples, n_features = vals_arr.shape
    feat_names = list(X.columns)

    mean_abs = np.mean(np.abs(vals_arr), axis=0)
    mean_signed = np.mean(vals_arr, axis=0)
    median = np.median(vals_arr, axis=0)
    std = np.std(vals_arr, axis=0)
    pos_frac = np.mean(vals_arr > 0, axis=0)
    neg_frac = np.mean(vals_arr < 0, axis=0)

    df = pd.DataFrame({
        "feature": feat_names,
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_signed,
        "median_shap": median,
        "std_shap": std,
        "pos_fraction": pos_frac,
        "neg_fraction": neg_frac,
    })
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    top_pos = df[df["mean_shap"] > 0].sort_values("mean_shap", ascending=False).head(top_k)
    top_neg = df[df["mean_shap"] < 0].sort_values("mean_shap").head(top_k)

    def fmt_list(df_slice, n=5):
        return ", ".join([f"{row.feature} ({row.mean_shap:.2f})" for _, row in df_slice.head(n).iterrows()])

    summary_lines = []
    if not top_pos.empty:
        summary_lines.append(
            f"Top features that increase predicted AQI (avg positive SHAP): {fmt_list(top_pos, n=min(5, len(top_pos)))}."
        )
    else:
        summary_lines.append("No features with a clear positive average contribution found.")

    if not top_neg.empty:
        summary_lines.append(
            f"Top features that decrease predicted AQI (avg negative SHAP): {fmt_list(top_neg, n=min(5, len(top_neg)))}."
        )
    else:
        summary_lines.append("No features with a clear negative average contribution found.")

    summary_lines.append(
        "The table below lists top features by average absolute SHAP (importance magnitude). "
        "mean_abs_shap gives the importance magnitude; mean_shap shows the average direction (+ increases prediction, - decreases)."
    )

    summary_text = " ".join(summary_lines)
    return df, summary_text


def main():
    st.markdown(
        """
        <style>
        .title {font-size:44px; font-weight:700; margin-bottom:6px;}
        .subtitle {color: #cfcfcf; margin-top:-6px; margin-bottom:18px;}
        .card {border-radius:12px;padding:18px;color:#fff;}
        .card-title {font-weight:700;font-size:18px;margin-bottom:6px;}
        .kpi-value {font-size:36px;font-weight:700;margin-top:6px;}
        .kpi-label {color:rgba(255,255,255,0.85);font-size:13px;}
        .small-note {color:#bdbdbd;font-size:13px;margin-top:8px;}
        .frozen-card { position: -webkit-sticky; position: sticky; top:20px; box-shadow: 0 8px 20px rgba(0,0,0,0.45); border: 1px solid rgba(255,255,255,0.04); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="title">Karachi Air Quality Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Historical data, live current AQI (OpenWeather) and forecast (next 3 days).</div>', unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        horizon = st.selectbox("Forecast horizon (hours shown)", options=[24, 48, 72], index=2)
        smoothing = st.slider("Smoothing window (hours)", 1, 24, 3)
        show_pm = st.multiselect("Show pollutant charts", options=["pm2_5", "pm10"], default=["pm2_5", "pm10"])
        shap_possible = True
        try:
            import shap  # type: ignore
        except Exception:
            shap_possible = False
        show_shap = st.checkbox("Show SHAP explanations (if available)", value=False, help="SHAP can be heavy — only enable when model & features are available.")
        st.markdown("---")
        st.markdown("Visual options")
        use_plotly = st.checkbox("Use interactive Plotly charts (hover tooltips, zoom)", value=(PLOTLY_AVAILABLE and True), disabled=not PLOTLY_AVAILABLE)
        show_markers = st.checkbox("Show markers on forecast (high/low)", value=True)
        st.markdown("---")
        st.markdown("Refresh predictions")
        refresh_days = st.selectbox("Select days to fetch", options=[1, 3], index=1, format_func=lambda x: f"{x} day(s)")
        if st.button("Refresh predictions now"):
            load_forecast.clear()
            get_latest_data_from_fg.clear()
            st.info("Refreshing predictions... this may take 10-90s.")
            ran_ok = False
            msg = ""
            if predictor is not None and hasattr(predictor, "get_predictions_for_dashboard"):
                try:
                    ft, us, one5 = predictor.get_predictions_for_dashboard(use_24h_model=False, prefer_hopsworks=True)
                    dfp = pd.DataFrame({"time": ft, "us_aqi": us, "aqi_1to5": one5})
                    try:
                        dfp["time"] = pd.to_datetime(dfp["time"], utc=True)
                    except Exception:
                        dfp["time"] = pd.to_datetime(dfp["time"], errors="coerce")
                        dfp["time"] = dfp["time"].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
                    try:
                        dfp.to_csv("predictions.csv", index=False)
                    except Exception:
                        pass
                    ran_ok = True
                    msg = "Predictions regenerated via predictor module (Hopsworks preferred)."
                except Exception as e:
                    msg = f"predictor.get_predictions_for_dashboard failed: {e}"
                    traceback.print_exc()
            if not ran_ok:
                msg = msg or "No predictor module available and no local predictions.csv present; using fallback synthetic forecast."
            st.success(msg)
            load_forecast.clear()
            get_latest_data_from_fg.clear()
            safe_rerun()

    # Load data
    forecast_df = load_forecast()
    hist = get_latest_data_from_fg()
    current_obs = get_current_openweather_aqi()

    try:
        forecast_df["time"] = pd.to_datetime(forecast_df["time"], utc=True, errors="coerce")
    except Exception:
        forecast_df["time"] = pd.to_datetime(forecast_df["time"], errors="coerce")
        forecast_df["time"] = forecast_df["time"].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")

    now_utc = pd.Timestamp.now(tz="UTC")
    start_cut = now_utc - pd.Timedelta(minutes=30)
    end_utc = now_utc + pd.Timedelta(days=3)
    forecast_df = forecast_df[(forecast_df["time"] >= start_cut) & (forecast_df["time"] <= end_utc)].sort_values("time").reset_index(drop=True)
    fdf = forecast_df.copy().head(horizon)

    avg_aqi_us = fdf["us_aqi"].mean() if not fdf.empty else np.nan
    max_aqi_us = fdf["us_aqi"].max() if not fdf.empty else np.nan
    avg_aqi_1to5 = fdf["aqi_1to5"].mean() if not fdf.empty else np.nan
    hazardous_hours = int((fdf["us_aqi"] > 200).sum()) if not fdf.empty else 0

    # Prepare current observation values
    cur_aqi = None
    cur_time_display = "N/A"
    cur_cat_label = "N/A"
    cur_cat_color = "#777777"
    if current_obs is not None:
        cur_aqi = current_obs.get("standard_aqi")
        try:
            cur_time_display = pd.to_datetime(current_obs.get("time")).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            cur_time_display = str(current_obs.get("time"))
        if cur_aqi is not None and not pd.isna(cur_aqi):
            cur_cat_label, cur_cat_color = us_aqi_category_and_color(cur_aqi)
        else:
            cur_cat_label = current_obs.get("category") or "N/A"

    forecast_card_color = "#0d6efd"
    current_card_color = cur_cat_color if cur_cat_color else "#6c757d"

    # Two cards row — Forecast (left) and Current (right). Current will use the "frozen" styling.
    cols = st.columns([2, 1])
    with cols[0]:
        st.markdown(
            f"""
            <div class="card" style="background:linear-gradient(135deg, {forecast_card_color}, #0056d6);">
                <div class="card-title">Forecast metrics <span style="font-weight:400;font-size:12px;color:rgba(255,255,255,0.9)">(from model predictions)</span></div>
                <div style="display:flex;gap:18px;align-items:center">
                    <div>
                        <div class="kpi-label">Avg US AQI (horizon)</div>
                        <div class="kpi-value">{avg_aqi_us:.0f}</div>
                    </div>
                    <div>
                        <div class="kpi-label">Max US AQI (horizon)</div>
                        <div class="kpi-value">{max_aqi_us:.0f}</div>
                    </div>
                    <div>
                        <div class="kpi-label">Avg AQI (1-5)</div>
                        <div class="kpi-value">{avg_aqi_1to5:.1f}</div>
                    </div>
                    <div>
                        <div class="kpi-label">Hazardous hours (AQI&gt;200)</div>
                        <div class="kpi-value">{hazardous_hours}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        cur_aqi_display = f"{cur_aqi:.0f}" if cur_aqi is not None and not pd.isna(cur_aqi) else "N/A"
        st.markdown(
            f"""
            <div class="card frozen-card" style="background:{current_card_color};">
                <div class="card-title">Current observation (OpenWeather)</div>
                <div style="margin-top:6px">
                    <div class="kpi-label">Current AQI (US)</div>
                    <div class="kpi-value">{cur_aqi_display}</div>
                    <div class="kpi-label" style="margin-top:10px">Time (UTC):</div>
                    <div style="font-weight:600;color:rgba(255,255,255,0.95)">{cur_time_display}</div>
                    <div style="margin-top:8px" class="kpi-label">Category:</div>
                    <div style="font-weight:700;color:rgba(255,255,255,0.95)">{cur_cat_label}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Forecast table & download
    st.markdown("### Forecast table — next 3 days (72 hours, model predictions)")
    if not forecast_df.empty:
        display_df = forecast_df.copy()
        display_df["time"] = pd.to_datetime(display_df["time"], utc=True)
        if "us_aqi" not in display_df.columns:
            display_df["us_aqi"] = np.nan
        if "aqi_1to5" not in display_df.columns:
            display_df["aqi_1to5"] = display_df["us_aqi"].apply(lambda x: int(np.clip(round(x/50), 1, 5)) if pd.notna(x) else np.nan)
        def aqi1to5_label(v):
            try:
                lbl, _ = aqi_1to5_label_color(int(v))
                return lbl
            except Exception:
                return "N/A"
        display_df["AQI_category"] = display_df["aqi_1to5"].apply(aqi1to5_label)
        table_df = display_df[["time", "us_aqi", "aqi_1to5", "AQI_category"]].reset_index(drop=True)
        table_df = table_df.rename(columns={"time": "Time (UTC)", "us_aqi": "US_AQI", "aqi_1to5": "AQI_1to5", "AQI_category": "AQI_category_label"})
        st.dataframe(table_df.style.format({"Time (UTC)": lambda t: pd.to_datetime(t).strftime("%Y-%m-%d %H:%M"), "US_AQI": "{:.0f}"}), height=420, width=1100)
        try:
            download_df = table_df.copy()
            download_df["Time (UTC)"] = pd.to_datetime(download_df["Time (UTC)"], utc=True).dt.strftime("%Y-%m-%d %H:%M")
            csv_bytes = download_df.to_csv(index=False).encode("utf-8")
            st.download_button(label="Download full next-3-days CSV", data=csv_bytes, file_name="predictions_next_3days.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Download button failed to prepare CSV: {e}")
    else:
        st.info("No forecast data available for the next 3 days.")

    # Charts
    st.markdown("## Forecast charts & pollutant forecast")

    # -- Forecast chart: use Plotly (interactive hover) when available --
    try:
        use_plotly_local = (PLOTLY_AVAILABLE and use_plotly)
        if use_plotly_local:
            fig = go.Figure()
            if not fdf.empty:
                fig.add_trace(
                    go.Scatter(
                        x=fdf["time"],
                        y=fdf["us_aqi"],
                        name="US AQI",
                        mode="lines+markers" if show_markers else "lines",
                        line=dict(color="royalblue", width=3),
                        fill="tozeroy",
                        fillcolor="rgba(65, 105, 225, 0.08)",
                        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>US AQI: %{y:.0f}<extra></extra>",
                        yaxis="y1",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=fdf["time"],
                        y=fdf["aqi_1to5"],
                        name="AQI (1-5)",
                        mode="lines+markers" if show_markers else "lines",
                        line=dict(color="orange", width=2, dash="dash"),
                        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>AQI (1-5): %{y:.1f}<extra></extra>",
                        yaxis="y2",
                    )
                )
                try:
                    idx_max = int(fdf["us_aqi"].idxmax())
                    max_time = fdf.loc[idx_max, "time"]
                    max_val = fdf.loc[idx_max, "us_aqi"]
                    fig.add_trace(
                        go.Scatter(
                            x=[max_time],
                            y=[max_val],
                            mode="markers+text" if show_markers else "markers",
                            marker=dict(color="red", size=10),
                            name="Max (US AQI)",
                            text=[f"Max: {max_val:.0f}"],
                            textposition="top center",
                            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Max US AQI: %{y:.0f}<extra></extra>",
                            yaxis="y1",
                        )
                    )
                except Exception:
                    pass

            fig.update_layout(
                xaxis=dict(title="Time (UTC)"),
                yaxis=dict(title="US AQI", side="left", rangemode="tozero"),
                yaxis2=dict(title="AQI (1-5)", overlaying="y", side="right", rangemode="tozero"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=40, b=40),
                template="plotly_white",
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots(figsize=(12, 4))
            if not fdf.empty:
                ax1.plot(fdf["time"], fdf["us_aqi"], color="tab:blue", lw=2, label="US AQI")
                ax1.fill_between(fdf["time"], fdf["us_aqi"], color="tab:blue", alpha=0.08)
            ax1.set_ylabel("US AQI", color="tab:blue")
            if st.checkbox("Show 1-5 series on same chart", value=True):
                ax2 = ax1.twinx()
                if not fdf.empty:
                    ax2.plot(fdf["time"], fdf["aqi_1to5"], color="tab:orange", lw=2, label="AQI (1-5)")
                ax2.set_ylabel("AQI (1-5)", color="tab:orange")
            ax1.set_xlabel("Time (UTC)")
            ax1.grid(alpha=0.2)
            try:
                ax1.set_xticklabels([t.strftime("%m-%d %H:%M") for t in fdf["time"].iloc[::max(1, len(fdf) // 8)]], rotation=30)
            except Exception:
                pass
            ax1.legend(loc="upper left")
            st.pyplot(fig)
    except Exception:
        st.info("Unable to render forecast chart.")

    # -- Pollutant forecast only: use Plotly interactive series (with smoothing) --
    if show_pm:
        st.markdown("### Pollutant forecast — next 3 days (forecast only)")
        for pollutant in show_pm:
            try:
                # Forecast series (next horizon hours) from fdf if present
                has_fc = pollutant in fdf.columns
                if has_fc:
                    fc_series = fdf[pollutant].astype(float).reset_index(drop=True)
                    fc_times = pd.to_datetime(fdf["time"]).reset_index(drop=True)
                else:
                    # If forecast pollutant missing, synthesize using recent history pattern when available
                    if hist is not None and pollutant in hist.columns:
                        horizon_len = len(fdf) if not fdf.empty else int(horizon)
                        pattern_len = min(72, len(hist), horizon_len)
                        if pattern_len <= 0:
                            st.info(f"No data available to produce a forecast for '{pollutant}'.")
                            continue
                        recent_pattern = hist[pollutant].astype(float).iloc[::-1].values[:pattern_len][::-1]
                        repeats = int(np.ceil(horizon_len / pattern_len))
                        extended = np.tile(recent_pattern, repeats)[:horizon_len].astype(float)
                        hist_std = float(np.nanstd(recent_pattern)) if pattern_len > 1 else max(1.0, float(np.nanmedian(recent_pattern)) * 0.05)
                        noise_scale = 0.25
                        rng = np.random.RandomState(42)
                        noise = rng.normal(loc=0.0, scale=hist_std * noise_scale, size=horizon_len)
                        synth_vals = np.maximum(0.0, extended + noise)
                        if not fdf.empty:
                            fc_times = pd.to_datetime(fdf["time"]).reset_index(drop=True)
                        else:
                            start = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1)
                            fc_times = pd.date_range(start=start, periods=horizon_len, freq="H", tz="UTC")
                            fc_times = pd.Series(fc_times)
                        fc_series = pd.Series(synth_vals)
                        has_fc = True
                    else:
                        st.info(f"No forecast or history available to plot '{pollutant}'.")
                        continue

                # compute forecast smoothing
                fc_smoothed = fc_series.rolling(smoothing, min_periods=1, center=True).mean() if smoothing > 1 else fc_series

                # Plot only forecast traces (no history)
                use_plotly_plots = (PLOTLY_AVAILABLE and use_plotly)
                if use_plotly_plots:
                    pfig = go.Figure()
                    pfig.add_trace(
                        go.Scatter(
                            x=fc_times,
                            y=fc_series,
                            name=f"{pollutant} (forecast)",
                            mode="lines+markers",
                            line=dict(color="rgba(255,165,0,0.9)", width=2, dash="dot"),
                            marker=dict(symbol="circle-open", size=6),
                            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Forecast " + pollutant + ": %{y:.1f}<extra></extra>",
                        )
                    )
                    try:
                        pfig.add_trace(
                            go.Scatter(
                                x=fc_times,
                                y=fc_smoothed,
                                name=f"{smoothing}h MA (forecast)",
                                mode="lines",
                                line=dict(color="rgba(255,165,0,0.6)", width=2, dash="dash"),
                                hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Forecast smoothed: %{y:.1f}<extra></extra>",
                            )
                        )
                    except Exception:
                        pass

                    pfig.update_layout(
                        template="plotly_white",
                        height=260,
                        margin=dict(l=30, r=30, t=20, b=30),
                        xaxis_title="Time (UTC)",
                        yaxis_title=pollutant,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(pfig, use_container_width=True)
                else:
                    import matplotlib.pyplot as plt
                    pfig, pax = plt.subplots(figsize=(12, 2.5))
                    pax.plot(fc_times, fc_series, linestyle=":", color="orange", marker="o", markersize=3, label=f"{pollutant} (forecast)")
                    try:
                        pax.plot(fc_times, fc_smoothed, linestyle="--", color="orange", label=f"{smoothing}h MA (forecast)")
                    except Exception:
                        pass
                    pax.set_ylabel(pollutant)
                    pax.set_xlabel("Time (UTC)")
                    pax.grid(alpha=0.2)
                    pax.legend()
                    st.pyplot(pfig)

            except Exception as e:
                st.warning(f"Failed rendering pollutant forecast chart for {pollutant}: {e}")

    # SHAP explanations (only attempt when user opted in and all prerequisites present)
    if show_shap:
        st.markdown("## SHAP explanations (model interpretability)")
        try:
            try:
                import shap  # type: ignore
                shap_installed = True
            except Exception:
                shap_installed = False
            if not shap_installed:
                st.info("SHAP library is not installed. Install shap and restart the app to enable SHAP explanations.")
            else:
                mdl, artifacts, mdl_path, folder, dt = load_model_and_artifacts_hopsworks()
                if mdl is None:
                    mdl, mdl_path = find_local_model()
                    artifacts = {}
                    source = "local" if mdl is not None else None
                else:
                    source = "hopsworks"
                if mdl is None:
                    st.info("No model artifact found locally or in Hopsworks. SHAP unavailable.")
                else:
                    st.success(f"Using model from {source}: {mdl_path}")
                    expected = getattr(mdl, "n_features_in_", None)
                    st.write("Model expects n_features_in_:", expected)

                    feature_cols = None
                    medians = None
                    for k, v in artifacts.items():
                        kl = k.lower()
                        if ("feature_cols" in kl or "feature_list" in kl or "feature_names" in kl):
                            if isinstance(v, (list, tuple, pd.Index)):
                                feature_cols = list(v)
                                break
                            if isinstance(v, pd.DataFrame):
                                feature_cols = list(v.iloc[:, 0].values)
                                break
                    for k, v in artifacts.items():
                        kl = k.lower()
                        if "median" in kl or "medians" in kl or "feature_medians" in kl:
                            medians = v
                            break

                    try:
                        X_sample = hist.copy().head(100)
                    except Exception:
                        X_sample = None
                    if X_sample is None or X_sample.empty:
                        st.info("Not enough recent history rows to compute SHAP sample.")
                    else:
                        if feature_cols:
                            X = pd.DataFrame(index=X_sample.index, columns=feature_cols, dtype=float)
                            for c in feature_cols:
                                if c in X_sample.columns:
                                    X[c] = X_sample[c].values
                                else:
                                    X[c] = np.nan
                        else:
                            X = X_sample.select_dtypes(include=[np.number]).copy()
                        X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
                        if medians is not None and isinstance(medians, (dict, pd.Series)):
                            for c in X.columns:
                                if X[c].isna().any():
                                    fill = medians.get(c) if isinstance(medians, dict) else (medians[c] if c in medians else None)
                                    if fill is not None and not pd.isna(fill):
                                        X[c].fillna(float(fill), inplace=True)
                        if expected is not None:
                            if X.shape[1] < expected:
                                if feature_cols and len(feature_cols) >= expected:
                                    pass
                                else:
                                    for i in range(expected - X.shape[1]):
                                        X[f"pad_extra_{i}"] = 0.0
                            elif X.shape[1] > expected:
                                if feature_cols:
                                    keep = [c for c in feature_cols if c in X.columns][:expected]
                                    X = X[keep]
                                else:
                                    X = X.iloc[:, :expected]
                        try:
                            X = X.fillna(X.median().to_dict())
                        except Exception:
                            X = X.fillna(0.0)
                        if X.isna().any().any():
                            X = X.fillna(0.0)
                        X = X.astype(float)
                        st.write("Prepared SHAP input shape:", X.shape)

                        try:
                            import shap as _shap  # type: ignore
                            name = type(mdl).__name__.lower()
                            estimator = mdl
                            try:
                                if hasattr(mdl, "named_steps") and "model" in mdl.named_steps:
                                    estimator = mdl.named_steps["model"]
                            except Exception:
                                estimator = mdl
                            if any(k in name for k in ("randomforest", "xgb", "lightgbm", "lgbm", "catboost", "forest")) or "sklearn.ensemble" in str(type(estimator)):
                                explainer = _shap.TreeExplainer(estimator)
                            else:
                                explainer = _shap.Explainer(mdl.predict, X)
                            shap_exp = explainer(X)

                            # SHAP summary table and text
                            try:
                                shap_df, shap_summary_text = summarize_shap(shap_exp, X, top_k=15)
                                st.markdown("### SHAP summary")
                                st.markdown(shap_summary_text)

                                top_n = min(15, shap_df.shape[0])
                                display_table = shap_df.head(top_n).copy()
                                display_table["mean_abs_shap"] = display_table["mean_abs_shap"].round(4)
                                display_table["mean_shap"] = display_table["mean_shap"].round(4)
                                display_table["pos_fraction"] = (display_table["pos_fraction"] * 100).round(1).astype(str) + "%"
                                display_table["neg_fraction"] = (display_table["neg_fraction"] * 100).round(1).astype(str) + "%"

                                st.table(display_table[["feature", "mean_abs_shap", "mean_shap", "pos_fraction", "neg_fraction"]].rename(columns={
                                    "feature": "Feature",
                                    "mean_abs_shap": "Mean |SHAP|",
                                    "mean_shap": "Mean SHAP",
                                    "pos_fraction": "% positive",
                                    "neg_fraction": "% negative",
                                }))

                                try:
                                    if PLOTLY_AVAILABLE:
                                        fig_imp = px.bar(shap_df.head(top_n).iloc[::-1], x="mean_abs_shap", y="feature", orientation="h",
                                                         labels={"mean_abs_shap": "Mean |SHAP| (importance)", "feature": ""})
                                        st.plotly_chart(fig_imp, use_container_width=True)
                                    else:
                                        import matplotlib.pyplot as plt
                                        fig_imp, ax = plt.subplots(figsize=(6, max(2, top_n * 0.35)))
                                        subset = shap_df.head(top_n).iloc[::-1]
                                        ax.barh(subset["feature"], subset["mean_abs_shap"], color="cornflowerblue")
                                        ax.set_xlabel("Mean |SHAP| (importance)")
                                        ax.set_title("Top SHAP feature importances")
                                        plt.tight_layout()
                                        st.pyplot(fig_imp)
                                except Exception as e_plot_imp:
                                    st.warning(f"Could not render SHAP importance chart: {e_plot_imp}")

                            except Exception as e_sum:
                                st.warning(f"SHAP summarization failed: {e_sum}")
                                traceback.print_exc()

                            # Try beeswarm (optional)
                            try:
                                _shap.plots.beeswarm(shap_exp, show=False)
                                import matplotlib.pyplot as plt
                                fig = plt.gcf()
                                st.pyplot(fig)
                                plt.clf()
                            except Exception as e_plot:
                                st.warning(f"SHAP beeswarm plot failed: {e_plot}. (Summary shown instead.)")
                        except Exception as e_shap:
                            st.error(f"SHAP computation failed: {e_shap}")
                            traceback.print_exc()
        except Exception as e:
            st.error(f"Explanation block failed: {e}")
            traceback.print_exc()

    st.markdown("---")
    st.caption("Notes: Current AQI above is fetched live from OpenWeather (not predicted). Forecast metrics shown above are generated by the model prediction (US AQI). Pollutant forecast lines appear only when forecast data contains pm2_5/pm10; otherwise a synthetic forecast is shown for layout/debugging. SHAP can be heavy and only runs when you opt-in and prerequisites (model & shap) are available.")


if __name__ == "__main__":
    main()
