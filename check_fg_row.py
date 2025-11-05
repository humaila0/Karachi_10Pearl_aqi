"""
Check Hopsworks Feature Group for the prepared row.

Usage:
  python src/check_fg_row.py

Reads debug_ingest_row_cleaned.csv in the repo root to get time_ms of the test row,
polls the feature group (aqi_current v1) until the row appears (or times out),
then prints the matching row and exits.
"""
import time
import sys
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

try:
    import hopsworks
except Exception as e:
    print("hopsworks SDK not available:", e)
    sys.exit(1)

CLEANED_CSV = "debug_ingest_row_cleaned.csv"
FG_NAME = "aqi_current"
FG_VERSION = 1
POLL_SECONDS = 2
MAX_TRIES = 30  # ~60 seconds total

def main():
    try:
        local = pd.read_csv(CLEANED_CSV, parse_dates=["time"])
    except Exception as e:
        print(f"Failed to read {CLEANED_CSV}:", e)
        sys.exit(1)

    if "time_ms" not in local.columns:
        print("time_ms column not found in cleaned CSV; ensure prepare.py wrote it.")
        sys.exit(1)

    target_time_ms = int(local["time_ms"].iloc[0])
    print("Looking for time_ms =", target_time_ms)

    p = hopsworks.login()
    fs = p.get_feature_store()
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    for i in range(MAX_TRIES):
        try:
            df = fg.select_all().read()
        except Exception as e:
            print(f"[attempt {i+1}] Read failed:", e)
            df = None

        if df is not None and "time_ms" in df.columns:
            if target_time_ms in df["time_ms"].values:
                print("Row found in feature group. Row:")
                row = df[df["time_ms"] == target_time_ms]
                print(row.T)
                return 0
            else:
                print(f"[attempt {i+1}] Row not found yet; retrying in {POLL_SECONDS}s...")
        else:
            print(f"[attempt {i+1}] FG read returned no rows or no time_ms; retrying in {POLL_SECONDS}s...")

        time.sleep(POLL_SECONDS)

    print("Row not found after polling. Open the job execution logs in Hopsworks UI and check for errors.")
    return 2

if __name__ == "__main__":
    sys.exit(main())