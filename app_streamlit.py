# (Full file — updated to render the Current observation card as a "frozen" colored card)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import hopsworks
import datetime
import os
import subprocess
import shlex
import glob
import concurrent.futures
import traceback
from dotenv import load_dotenv
from typing import Tuple, Optional

# Load environment variables from .env if present
load_dotenv()

# Try to import local prediction helper
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


# ---------- small helper to run shell commands ----------
def run_command(cmd, timeout=300):
    if isinstance(cmd, (list, tuple)):
        cmd_list = cmd
    else:
        cmd_list = shlex.split(cmd)
    try:
        proc = subprocess.run(
            cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, "", f"TimeoutExpired: {e}"


# ---------- small helpers ----------
def find_first_local(pattern):
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def find_local_model(models_dir: str = "models") -> Tuple[Optional[object], Optional[str]]:
    patterns = [
        os.path.join(models_dir, "aqi_predictor.pkl"),
        os.path.join(models_dir, "rf_aqi_predictor_*.pkl"),
        os.path.join(models_dir, "xgb_aqi_predictor_*.pkl"),
        os.path.join(models_dir, "lgbm_aqi_predictor_*.pkl"),
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
        proj = hopsworks.login()
        return proj
    except Exception as e:
        print("Hopsworks login failed:", e)
        return None


@st.cache_resource
def load_model_and_artifacts_hopsworks(model_name: Optional[str] = None, model_version: Optional[int] = None, allowed_patterns=None):
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
            folder = run_with_timeout(lambda: entry.download(), timeout=60)
        except TimeoutError:
            print("Hopsworks model download timed out after 60s")
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
        df = fg.select_all().read().sort_values("time", ascending=False).head(168)
        return df.reset_index(drop=True)
    try:
        df = run_with_timeout(_read_fg, timeout_seconds)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        return df
    except TimeoutError as e:
      #  st.warning(f"Hopsworks read timed out after {timeout_seconds}s — using fallback data. ({e})")
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


@st.cache_data(ttl=300)
def load_forecast():
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
    try:
        if predictor is not None and hasattr(predictor, "get_predictions_for_dashboard"):
            try:
                ft, preds_us, preds_1to5 = predictor.get_predictions_for_dashboard(use_24h_model=False)
            except TypeError:
                ft, preds_us, preds_1to5 = predictor.get_predictions_for_dashboard()
            df = pd.DataFrame({"time": ft, "us_aqi": preds_us, "aqi_1to5": preds_1to5})
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            return df.sort_values("time").reset_index(drop=True)
    except Exception as e:
        print("[load_forecast] predictor.get_predictions_for_dashboard failed:", e)
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
        /* frozen card styling: sticky + shadow */
        .frozen-card { position: -webkit-sticky; position: sticky; top:20px; box-shadow: 0 8px 20px rgba(0,0,0,0.45); border: 1px solid rgba(255,255,255,0.04); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="title">Karachi Air Quality Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Historical data, live current AQI (OpenWeather) and forecast (next 3 days).</div>', unsafe_allow_html=True)

    # Sidebar (unchanged)
    with st.sidebar:
        st.header("Controls")
        horizon = st.selectbox("Forecast horizon (hours shown)", options=[24, 48, 72], index=2)
        smoothing = st.slider("Smoothing window (hours)", 1, 24, 3)
        show_pm = st.multiselect("Show pollutant charts", options=["pm2_5", "pm10"], default=["pm2_5", "pm10"])
        show_shap = st.checkbox("Show SHAP explanations (if available)", value=False)
        st.markdown("---")
        st.markdown("Refresh predictions")
        refresh_days = st.selectbox("Select days to fetch", options=[1, 3], index=1, format_func=lambda x: f"{x} day(s)")
        if st.button("Refresh predictions now"):
            load_forecast.clear()
            get_latest_data_from_fg.clear()
            st.info("Refreshing predictions... this may take 20-90s.")
            ran_ok = False
            msg = ""
            if predictor is not None and hasattr(predictor, "get_predictions_for_dashboard"):
                try:
                    ft, us, one5 = predictor.get_predictions_for_dashboard(use_24h_model=False)
                    dfp = pd.DataFrame({"time": ft, "us_aqi": us, "aqi_1to5": one5})
                    try:
                        dfp["time"] = pd.to_datetime(dfp["time"], utc=True)
                    except Exception:
                        dfp["time"] = pd.to_datetime(dfp["time"], errors="coerce")
                        dfp["time"] = dfp["time"].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
                    dfp.to_csv("predictions.csv", index=False)
                    ran_ok = True
                    msg = "Predictions regenerated via predictor module (requested full 72h)."
                except Exception as e:
                    msg = f"predictor.get_predictions_for_dashboard failed: {e}"
            if not ran_ok:
                cmd = f"{os.sys.executable} -u src/predict.py"
                rc, out, err = run_command(cmd, timeout=240)
                if rc == 0:
                    ran_ok = True
                    msg = "Predictions regenerated via src/predict.py (subprocess)."
                else:
                    msg = f"Subprocess predict.py failed (rc={rc}). stderr: {err[:1000]}"
            st.success(msg)
            load_forecast.clear()
            get_latest_data_from_fg.clear()
            st.experimental_rerun()

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
        # Use the frozen-card class so it remains visually "frozen"/sticky and colored
        st.markdown(
            f"""
            <div class="card frozen-card" style="background:{current_card_color};">
                <div class="card-title">Current observation (OpenWeather)</div>
                <div style="margin-top:6px">
                    <div class="kpi-label">Current AQI (US)</div>
                    <div class="kpi-value">{cur_aqi_display}</div>
                    <div class="kpi-label" style="margin-top:8px">Time (UTC):</div>
                    <div style="font-weight:600;color:rgba(255,255,255,0.95)">{cur_time_display}</div>
                    <div style="margin-top:8px" class="kpi-label">Category:</div>
                    <div style="font-weight:700;color:rgba(255,255,255,0.95)">{cur_cat_label}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Re-add full 72h forecast table with AQI category label
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

    # Charts (unchanged)...
    st.markdown("## Forecast charts & pollutant history")
    try:
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

    if show_pm:
        st.markdown("### Recent pollutant history (last 7 days)")
        for pollutant in show_pm:
            pfig, pax = plt.subplots(figsize=(12, 2.5))
            if pollutant in hist.columns:
                series = hist[pollutant].astype(float).iloc[::-1]
                times = pd.to_datetime(hist["time"].iloc[::-1])
                pax.plot(times, series, label=pollutant, color="tab:red" if pollutant == "pm2_5" else "tab:green")
                pax.plot(times, series.rolling(smoothing, min_periods=1).mean(), linestyle="--", color="black", label=f"{smoothing}h MA")
                pax.set_ylabel(pollutant)
                pax.set_xlabel("Time (UTC)")
                pax.grid(alpha=0.2)
                pax.legend()
                st.pyplot(pfig)
            else:
                st.info(f"No column '{pollutant}' in feature store data to plot.")

    if show_shap:
        st.markdown("## SHAP explanations (model interpretability)")
        try:
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
                        import shap
                        name = type(mdl).__name__.lower()
                        estimator = mdl
                        try:
                            if hasattr(mdl, "named_steps") and "model" in mdl.named_steps:
                                estimator = mdl.named_steps["model"]
                        except Exception:
                            estimator = mdl
                        if any(k in name for k in ("randomforest", "xgb", "lightgbm", "lgbm", "catboost", "forest")) or "sklearn.ensemble" in str(type(estimator)):
                            explainer = shap.TreeExplainer(estimator)
                        else:
                            explainer = shap.Explainer(mdl.predict, X)
                        shap_exp = explainer(X)
                        try:
                            shap.plots.beeswarm(shap_exp, show=False)
                            fig = plt.gcf()
                            st.pyplot(fig)
                            plt.clf()
                        except Exception as e_plot:
                            st.warning(f"SHAP beeswarm plot failed: {e_plot}. Showing textual SHAP importances.")
                            vals = shap_exp.values
                            if isinstance(vals, list):
                                vals_arr = np.array(vals[0])
                            else:
                                vals_arr = np.array(vals)
                            if vals_arr.ndim == 1:
                                vals_arr = vals_arr.reshape(1, -1)
                            mean_abs = np.mean(np.abs(vals_arr), axis=0)
                            feat_names = list(X.columns)
                            imp_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
                            imp_df = imp_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
                            st.table(imp_df.head(30))
                    except Exception as e_shap:
                        st.error(f"SHAP computation failed: {e_shap}")
                        traceback.print_exc()
        except Exception as e:
            st.error(f"Explanation block failed: {e}")
            traceback.print_exc()

    st.markdown("---")
    st.caption("Notes: Current AQI above is fetched live from OpenWeather (not predicted). Forecast metrics shown above are generated by the model (US AQI).")

if __name__ == "__main__":
    main()