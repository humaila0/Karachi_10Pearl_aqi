"""
Prepare an engineered row to match an existing Hopsworks Feature Group schema and ingest it.

Usage:
  - python src/prepare.py

What it does:
  1) Logs into Hopsworks and reads the target feature group schema (columns + dtypes).
  2) Loads one engineered row (by calling transform.engineer_features on latest local row) OR
     you can load a CSV (adjust the code below).
  3) Adds any missing columns required by the FG (as NaN), coerces dtypes to int64/float,
     computes time_ms if missing, and ensures primary-key column exists.
  4) Drops any extra columns not present in the FG (so we don't attempt to insert unknown features).
  5) Writes a cleaned CSV debug_ingest_row_cleaned.csv for inspection.
  6) Attempts fg.insert(df) (upsert). If insert fails it prints the FG schema and dataframe
     dtypes and preserves the cleaned CSV for manual upload/debug.
Notes:
  - This script does not change the feature group schema. If your engineered row truly
    requires a different schema, either (A) modify transform to write the expected columns,
    or (B) create a new FG version in Hopsworks with the new schema.
"""
import os
import sys
import traceback
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np

try:
    import hopsworks
except Exception:
    hopsworks = None

from transform import engineer_features  # assumes you have transform.engineer_features available

FG_NAME = os.getenv("HOPSWORKS_FEATURE_GROUP", "aqi_current")
FG_VERSION = int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "1"))
CLEANED_CSV = os.getenv("CLEANED_DEBUG_CSV", "debug_ingest_row_cleaned.csv")

def get_fg_columns_and_types(fg):
    """
    Try to read a small sample from the FG to infer column names and dtypes.
    Returns: dict {col_name: dtype_str}
    """
    try:
        # try preferred read API
        df_sample = fg.select_all().read().head(1)
    except Exception:
        try:
            df_sample = fg.read().head(1)
        except Exception:
            # if no rows or API differs, try to inspect fg.features if available
            try:
                features = getattr(fg, "features", None)
                if features:
                    return {f.name: getattr(f, "type", "unknown") for f in features}
            except Exception:
                pass
            return {}

    # if sample available, map pandas dtypes to simple strings
    if df_sample is None or df_sample.empty:
        return {}
    col_types = {}
    for c, t in df_sample.dtypes.items():
        if pd.api.types.is_integer_dtype(t):
            col_types[c] = "bigint"
        elif pd.api.types.is_float_dtype(t):
            col_types[c] = "double"
        elif pd.api.types.is_bool_dtype(t):
            col_types[c] = "boolean"
        else:
            col_types[c] = "string"
    return col_types

def prepare_dataframe_for_fg(df: pd.DataFrame, fg_col_types: dict) -> pd.DataFrame:
    """
    Make df compatible with the FG schema by adding missing columns and coercing dtypes.
    This is conservative: adds missing columns as NaN and casts numeric ints -> int64 where needed.
    """
    df = df.copy()

    # Ensure time column
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    else:
        df["time"] = pd.Timestamp.utcnow()

    # Compute time_ms if missing
    if "time_ms" not in df.columns or df["time_ms"].isnull().all():
        df["time_ms"] = (df["time"].astype("int64") // 10 ** 6).astype("int64")

    # Add any missing FG columns as NaN with appropriate dtype hints
    for col, fg_type in fg_col_types.items():
        if col not in df.columns:
            if fg_type in ("bigint",):
                df[col] = 0
            else:
                df[col] = np.nan

    # Coerce dtype for integer-like columns to int64 (FG expects bigint)
    for col, fg_type in fg_col_types.items():
        if fg_type == "bigint":
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
            except Exception:
                df[col] = df[col].fillna(0).astype("int64")

        # floats -> ensure numeric
        if fg_type == "double":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # Make sure columns order is consistent (not strictly required but helpful)
    ordered_cols = list(fg_col_types.keys())
    # append any additional columns at the end
    for c in df.columns:
        if c not in ordered_cols:
            ordered_cols.append(c)
    df = df[ordered_cols]
    return df

def main():
    if hopsworks is None:
        print("hopsworks SDK not installed. Exiting.")
        sys.exit(1)

    try:
        project = hopsworks.login()
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
    except Exception as e:
        print("Failed to access Hopsworks or feature group. Error:")
        traceback.print_exc()
        sys.exit(1)

    print(f"Connected to project. Target feature group: {FG_NAME} v{FG_VERSION}")

    # Get FG schema (best-effort)
    fg_col_types = get_fg_columns_and_types(fg)
    if not fg_col_types:
        print("Could not infer FG schema from sample rows or FG features. Continuing with conservative defaults.")
    else:
        print("Feature group sample schema (inferred):")
        for k, v in fg_col_types.items():
            print(f"  {k}: {v}")

    # --- Load or build the engineered row to ingest ---
    local_input_csv = os.getenv("PREPROCESSED_CSV", "features_with_standard_aqi.csv")
    if os.path.exists(local_input_csv):
        print("Loading local CSV and engineering features...")
        raw = pd.read_csv(local_input_csv, parse_dates=["time"])
        # Use last few rows to compute lags; engineer_features will handle single-row too
        eng = engineer_features(raw.tail(48), target="standard_aqi_next_24h", drop_future_targets=True, return_last_row=True)
        df_to_ingest = eng.copy()
    else:
        print(f"Local CSV {local_input_csv} not found. Building a minimal synthetic row for test.")
        now = pd.Timestamp.utcnow()
        df_to_ingest = pd.DataFrame([{
            "time": now,
            "pm2_5": 1.0,
            "pm10": 1.0,
            "standard_aqi": 1.0
        }])
        df_to_ingest = engineer_features(df_to_ingest, target="standard_aqi_next_24h", drop_future_targets=True, return_last_row=True)

    print("Engineered row columns/dtypes before compatibility pass:")
    print(df_to_ingest.dtypes)
    print(df_to_ingest.head(1).T)

    # Prepare df to match FG schema
    if fg_col_types:
        df_ready = prepare_dataframe_for_fg(df_to_ingest, fg_col_types)
    else:
        # conservative fallback: ensure key columns and types
        df_ready = df_to_ingest.copy()
        if "time_ms" not in df_ready.columns:
            df_ready["time_ms"] = (df_ready["time"].astype("int64") // 10 ** 6).astype("int64")
        # ensure ints cast
        for c in ["hour", "day", "weekday", "is_weekend"]:
            if c in df_ready.columns:
                df_ready[c] = pd.to_numeric(df_ready[c], errors="coerce").fillna(0).astype("int64")

    print("Prepared row columns/dtypes to ingest:")
    print(df_ready.dtypes)
    print(df_ready.head(1).T)

    # -------- NEW: Drop any columns not present in FG schema (prevents "unknown column" errors) -------
    if fg_col_types:
        fg_columns = list(fg_col_types.keys())
        # keep only columns that exist in FG (preserve order)
        cols_to_keep = [c for c in fg_columns if c in df_ready.columns]
        # also ensure target/time_ms are present even if not in fg_columns mapping
        if "standard_aqi_next_24h" in df_ready.columns and "standard_aqi_next_24h" not in cols_to_keep:
            cols_to_keep.append("standard_aqi_next_24h")
        if "time_ms" in df_ready.columns and "time_ms" not in cols_to_keep:
            cols_to_keep.append("time_ms")
        df_ready = df_ready[cols_to_keep]

        # Re-coerce dtypes for kept columns according to fg_col_types
        for col, fg_type in fg_col_types.items():
            if col in df_ready.columns:
                if fg_type == "bigint":
                    df_ready[col] = pd.to_numeric(df_ready[col], errors="coerce").fillna(0).astype("int64")
                elif fg_type == "double":
                    df_ready[col] = pd.to_numeric(df_ready[col], errors="coerce").astype("float64")

    # Write cleaned CSV so you can inspect or manually upload if needed
    try:
        df_ready.to_csv(CLEANED_CSV, index=False)
        print(f"Wrote cleaned ingest CSV: {CLEANED_CSV} (columns: {len(df_ready.columns)})")
    except Exception:
        print("Failed to write cleaned CSV - continuing to attempt FG insert")

    print("Final DF columns being inserted:")
    print(df_ready.columns.tolist())
    print(df_ready.dtypes)

    # Try insert into FG (upsert)
    try:
        print("Attempting to insert into feature group (upsert)...")
        fg.insert(df_ready, write_options={"wait_for_job": True})
        print("Insert succeeded.")
    except Exception:
        print("Feature group insert failed. Traceback and DF/FG info printed below:")
        traceback.print_exc()
        print("FG inferred schema keys:", list(fg_col_types.keys()))
        print("DF columns:", df_ready.columns.tolist())
        print("DF dtypes:")
        print(df_ready.dtypes)
        # cleaned CSV already written above (CLEANED_CSV)
        print(f"Wrote prepared row to {CLEANED_CSV} for manual inspection/upload.")
        sys.exit(1)

    print("Ingest completed successfully.")
    return

if __name__ == "__main__":
    main()