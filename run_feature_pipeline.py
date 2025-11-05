"""
Hourly feature pipeline:

- Fetch current observation from OpenWeather (fetch_openweather_current).
- Compute standard AQI and category (compute_aqi_row / aqi_category).
- Gather recent history from Hopsworks feature group (aqi_hourly) if available,
  combine with current row and run transform.engineer_features to produce lag/roll features.
- Ingest engineered latest row into Hopsworks (ingest_to_hopsworks).
- Optionally trigger training (python -m src/train) with environment flags:
    - RETRAIN=true (will call train.py)
    - REGISTRY_MIN_R2 (threshold to upload)
    - SAVE_LOCAL_RF (true/false)
"""
import os
import sys
from datetime import datetime
import traceback

from dotenv import load_dotenv
load_dotenv()  # loads .env at repo root

import pandas as pd
import numpy as np

from fetcher import fetch_openweather_current
from compute_aqi import compute_aqi_row, aqi_category
from transform import engineer_features
from hopsworks_ingest import ingest_to_hopsworks

# Optional: import hopsworks here only if available (we handle exceptions)
try:
    import hopsworks
except Exception:
    hopsworks = None

# Config (can override with environment variables)
LAT = float(os.getenv("LATITUDE", "24.8607"))
LON = float(os.getenv("LONGITUDE", "67.0011"))
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_FEATURE_GROUP_NAME = os.getenv("HOPSWORKS_FEATURE_GROUP", "aqi_hourly")
HOPSWORKS_FEATURE_GROUP_VERSION = int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "1"))
RETRAIN_AFTER_INGEST = os.getenv("RETRAIN_AFTER_INGEST", "false").lower() in ("1","true","yes")
BACKFILL_HOURS = int(os.getenv("BACKFILL_HOURS", "0"))  # if >0, fetch historical range and ingest

# --- Added helper: ensure that the engineered record matches the aqi_current FG schema ---
# Feature-group columns (from your FG schema). Keep in sync with your aqi_current FG.
FG_COLUMNS = [
    "time","pm2_5","pm10","co","no2","o3","so2","nh3",
    "temperature","humidity","wind_speed","pressure",
    "standard_aqi","hour","day","month","year","weekday","is_weekend",
    "standard_aqi_lag_1","standard_aqi_lag_3","standard_aqi_lag_6","standard_aqi_lag_12",
    "standard_aqi_lag_24","standard_aqi_lag_48","standard_aqi_lag_72",
    "standard_aqi_roll_3","standard_aqi_roll_6","standard_aqi_roll_12","standard_aqi_roll_24",
    "standard_aqi_roll_48","standard_aqi_roll_72","pm_ratio","oxidant_sum",
    "standard_aqi_change_1h","standard_aqi_change_24h","standard_aqi_pct_change_1h","standard_aqi_pct_change_24h",
    "standard_aqi_next_24h","time_ms","hour_sin","hour_cos","month_sin","month_cos",
    "standard_aqi_roll_std_3","standard_aqi_roll_std_6","standard_aqi_roll_std_12","standard_aqi_roll_std_24",
    "standard_aqi_roll_std_48","standard_aqi_roll_std_72","standard_aqi_change_3h","standard_aqi_pct_change_3h",
    "standard_aqi_change_6h","standard_aqi_pct_change_6h","standard_aqi_change_12h","standard_aqi_pct_change_12h",
    "pm2_5_lag_1","pm2_5_lag_3","pm2_5_lag_6","pm2_5_lag_24","pm2_5_roll_24",
    "pm10_lag_1","pm10_lag_3","pm10_lag_6","pm10_lag_24","pm10_roll_24",
    "no2_lag_1","no2_lag_3","no2_lag_6","no2_lag_24","no2_roll_24",
    "o3_lag_1","o3_lag_3","o3_lag_6","o3_lag_24","o3_roll_24","temp_humidity"
]

def clean_record_for_fg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df matches the aqi_current FG: add missing columns, compute time_ms/month/year,
    cast integer columns to int64, compute temp_humidity, and drop unknown columns.
    """
    df = df.copy()

    # ensure time dtype
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    else:
        df["time"] = pd.Timestamp.utcnow()

    # compute time_ms, month, year
    # pandas Timestamp -> int64 may include timezone-aware values; coerce safely
    try:
        df["time_ms"] = (df["time"].astype("int64") // 10 ** 6).astype("int64")
    except Exception:
        # fallback for timezone-aware or other types
        df["time_ms"] = df["time"].apply(lambda t: int(pd.Timestamp(t).value // 10 ** 6)).astype("int64")

    df["month"] = df["time"].dt.month
    df["year"] = df["time"].dt.year

    # ensure weather columns exist (nullable) so insert schema passes
    for c in ["temperature", "humidity", "wind_speed", "pressure"]:
        if c not in df.columns:
            df[c] = np.nan

    # compute month_sin/month_cos if needed
    if "month_sin" not in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * (df["month"].fillna(1) - 1) / 12)
    if "month_cos" not in df.columns:
        df["month_cos"] = np.cos(2 * np.pi * (df["month"].fillna(1) - 1) / 12)

    # compute temp_humidity (nullable)
    if {"temperature", "humidity"}.issubset(df.columns):
        df["temp_humidity"] = (pd.to_numeric(df["temperature"], errors="coerce") *
                               pd.to_numeric(df["humidity"], errors="coerce")).astype("float64")
    else:
        df["temp_humidity"] = np.nan

    # ensure standard_aqi_next_24h exists (nullable)
    if "standard_aqi_next_24h" not in df.columns:
        df["standard_aqi_next_24h"] = np.nan

    # Ensure integer-like columns are int64 (bigint in FG)
    for c in ["hour","day","month","year","weekday","is_weekend","time_ms"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
        else:
            # create it as int64 zero to satisfy schema
            df[c] = pd.Series([0] * len(df), index=df.index, dtype="int64")

    # Ensure numeric typed columns are numeric
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            pass

    # Drop columns not in FG_COLUMNS (prevents "does not exist in feature group" errors)
    keep = [c for c in FG_COLUMNS if c in df.columns]
    df = df[keep]

    return df

# ---------------------------------------------------------------------------

def fetch_current():
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY missing in environment (.env).")
    df = fetch_openweather_current(LAT, LON, OPENWEATHER_API_KEY)
    if df is None or df.empty:
        raise RuntimeError("No data returned from fetch_openweather_current()")
    return df

def get_recent_history_from_hopsworks(hours=48):
    """
    Attempt to read the latest `hours` rows from the Hopsworks feature group.
    Returns DataFrame or None on failure.
    This version is defensive: checks fg existence and tries alternative read methods.
    """
    if hopsworks is None:
        return None
    try:
        project = hopsworks.login()
        fs = project.get_feature_store()
    except Exception:
        # Can't log in / feature store not available
        traceback.print_exc()
        return None

    # Try to get the feature group; if not present, return None
    try:
        fg = fs.get_feature_group(name=HOPSWORKS_FEATURE_GROUP_NAME, version=HOPSWORKS_FEATURE_GROUP_VERSION)
    except Exception:
        fg = None

    if fg is None:
        print(f"Feature group '{HOPSWORKS_FEATURE_GROUP_NAME}' v{HOPSWORKS_FEATURE_GROUP_VERSION} not found. Proceeding without history.")
        return None

    # Try common read APIs (some SDK versions differ)
    try:
        # Preferred: select_all().read()
        query = fg.select_all()
        df = query.read()
    except Exception:
        try:
            # Alternative: fg.read()
            df = fg.read()
        except Exception:
            try:
                # Some SDKs expose to_pandas or read_table-like methods
                df = fg.to_pandas()
            except Exception:
                traceback.print_exc()
                return None

    if df is None or df.empty:
        return None

    # Ensure time column parsed & sorted
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

    return df.tail(hours).reset_index(drop=True)

def engineer_single_record_with_history(current_df, history_df):
    """
    Combine history_df and current_df, engineer features and return the last engineered row.
    If history_df is None, run engineer_features(current_df) which will create time-only features.
    Use return_last_row=True so we get a single ready row for ingest.

    This version normalizes the time column to timezone-aware UTC (utc=True) to avoid
    pd.to_datetime conversion errors inside transform.engineer_features.
    """
    # Defensive copies
    current_df = current_df.copy()
    if history_df is not None:
        history_df = history_df.copy()

    # Ensure time exists on current_df and convert to timezone-aware UTC
    if "time" in current_df.columns:
        current_df["time"] = pd.to_datetime(current_df["time"], errors="coerce", utc=True)
    else:
        # create a UTC timezone-aware timestamp if missing
        current_df["time"] = pd.to_datetime(pd.Timestamp.utcnow()).tz_localize("UTC")

    # If history is present, ensure its time column is parsed and converted to UTC
    if history_df is not None and "time" in history_df.columns:
        history_df["time"] = pd.to_datetime(history_df["time"], errors="coerce", utc=True)

    # Combine and engineer features
    if history_df is not None and not history_df.empty:
        combined = pd.concat([history_df, current_df], ignore_index=True, sort=False)
        # Normalize the combined time column to UTC (this makes all entries timezone-aware)
        combined["time"] = pd.to_datetime(combined["time"], errors="coerce", utc=True)
        combined = combined.sort_values("time").reset_index(drop=True)
        engineered = engineer_features(combined, target="standard_aqi_next_24h", drop_future_targets=True, return_last_row=True)
        return engineered
    else:
        # No history available; ensure current_df['time'] is timezone-aware and engineer features
        current_df["time"] = pd.to_datetime(current_df["time"], errors="coerce", utc=True)
        engineered = engineer_features(current_df, target="standard_aqi_next_24h", drop_future_targets=True, return_last_row=True)
        return engineered

def run(retrain_model=False):
    print(f"[{datetime.utcnow().isoformat()}] Starting feature pipeline run")
    try:
        current = fetch_current()
        print(f"[{datetime.utcnow().isoformat()}] Fetched current observation: {current['time'].iloc[0]}")

        # compute standard AQI & category
        current['standard_aqi'] = current.apply(compute_aqi_row, axis=1)
        current['aqi_category'] = current['standard_aqi'].apply(aqi_category)
        print(f"[{datetime.utcnow().isoformat()}] Computed standard_aqi: {current['standard_aqi'].iloc[0]}")

        # fetch history from Hopsworks if possible
        history = get_recent_history_from_hopsworks(hours=48)
        if history is not None:
            print(f"[{datetime.utcnow().isoformat()}] Retrieved {len(history)} history rows from Hopsworks")
        else:
            print(f"[{datetime.utcnow().isoformat()}] No history available from Hopsworks; using only current row")

        # engineer — get a single engineered row ready for ingestion
        new_record = engineer_single_record_with_history(current, history)
        print(f"[{datetime.utcnow().isoformat()}] Engineered features for new record. Columns: {list(new_record.columns)}")

        # --- Clean the engineered record so it matches the aqi_current FG before ingesting ---
        try:
            new_record = clean_record_for_fg(new_record)
        except Exception as e:
            print("Failed to clean record for FG:", e)
            traceback.print_exc()
            return None

        # ingest to Hopsworks via provided helper (ingest_to_hopsworks now accepts DataFrame)
        try:
            ok = ingest_to_hopsworks(new_record)
            if ok:
                print(f"[{datetime.utcnow().isoformat()}] Ingested engineered record into Hopsworks feature store")
            else:
                print(f"[{datetime.utcnow().isoformat()}] Ingest failed (see logs in hopsworks_ingest).")
        except Exception:
            print(f"[{datetime.utcnow().isoformat()}] ingest_to_hopsworks() raised an exception — printing traceback and continuing")
            traceback.print_exc()

        # Optionally trigger model retrain (subprocess so training runs as standalone process)
        if retrain_model or RETRAIN_AFTER_INGEST:
            print(f"[{datetime.utcnow().isoformat()}] RETRAIN requested — launching training run (subprocess)")
            import subprocess, shlex
            cmd = f"{sys.executable} -u src/train.py"
            try:
                subprocess.run(shlex.split(cmd), check=True)
                print(f"[{datetime.utcnow().isoformat()}] Retrain subprocess finished")
            except subprocess.CalledProcessError as e:
                print(f"[{datetime.utcnow().isoformat()}] Retrain subprocess failed: {e}")
    except Exception as e:
        print(f"[{datetime.utcnow().isoformat()}] Feature pipeline failed: {e}")
        traceback.print_exc()
        return None

    print(f"[{datetime.utcnow().isoformat()}] Feature pipeline run completed successfully")
    return new_record

if __name__ == "__main__":
    # If run directly, allow optional CLI flag to retrain
    retrain_flag = os.environ.get("RETRAIN", "false").lower() in ("1","true","yes")
    run(retrain_model=retrain_flag)