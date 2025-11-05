"""
Ingest helper for Hopsworks.

ingest_to_hopsworks(record_or_path, feature_group_name='aqi_current', version=1)

- Accepts either a pandas DataFrame (single-row or multi-row) or a local CSV path.
- If the target Feature Group exists and supports `insert`, it will insert the DataFrame directly.
- Otherwise it uploads the CSV to the project's dataset area as a fallback.
- Uses forward-slash remote path for Hopsworks API (fixes Windows backslash issue).
"""
import os
import tempfile
import traceback
from dotenv import load_dotenv
load_dotenv()

try:
    import hopsworks
except Exception:
    hopsworks = None

import pandas as pd

DEFAULT_FEATURE_GROUP = os.getenv("HOPSWORKS_FEATURE_GROUP", "aqi_current")
DEFAULT_FG_VERSION = int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "1"))

def ingest_to_hopsworks(record_or_path, feature_group_name=DEFAULT_FEATURE_GROUP, version=DEFAULT_FG_VERSION):
    """
    record_or_path: pandas.DataFrame or path to CSV
    feature_group_name: feature group to insert into
    version: feature group version
    Returns True on success, False on failure.
    """
    if hopsworks is None:
        print("hopsworks SDK not installed - cannot ingest to Hopsworks.")
        return False

    try:
        project = hopsworks.login()
        fs = project.get_feature_store()
    except Exception:
        print("Failed to login to Hopsworks project for ingestion.")
        traceback.print_exc()
        return False

    # Try to get the feature group; it's OK if not present (we fallback)
    try:
        fg = fs.get_feature_group(name=feature_group_name, version=version)
    except Exception:
        fg = None

    # Normalize input: DataFrame -> df, path -> read df
    df = None
    csv_path = None
    if isinstance(record_or_path, pd.DataFrame):
        df = record_or_path.copy()
        # ensure time column dtype is datetime if present
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
    else:
        csv_path = str(record_or_path)
        if not os.path.exists(csv_path):
            print("CSV path does not exist:", csv_path)
            return False
        try:
            df = pd.read_csv(csv_path, parse_dates=["time"])
        except Exception:
            print("Failed to read CSV:", csv_path)
            traceback.print_exc()
            return False

    # If fg exists and supports insert, use it
    if fg is not None:
        try:
            if hasattr(fg, "insert"):
                print(f"Inserting dataframe into feature group: {feature_group_name} v{version}")
                fg.insert(df, write_options={"wait_for_job": True})
                print("Insert succeeded.")
                return True
            else:
                # Some SDKs use write() or upsert methods
                try:
                    print(f"Attempting fg.write for feature group: {feature_group_name}")
                    fg.write(df)
                    print("fg.write succeeded.")
                    return True
                except Exception:
                    pass
        except Exception:
            print("Feature group insert/write failed — will attempt fallback upload. Traceback:")
            traceback.print_exc()

    # Fallback: upload CSV to project dataset area
    try:
        dataset_api = project.get_dataset_api()
        # write df to temp csv if necessary
        if csv_path is None:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            csv_path = tf.name
            tf.close()
            df.to_csv(csv_path, index=False)
            created_temp = True
        else:
            created_temp = False

        target_dir = f"/Projects/{project.name}/feature_ingest"
        # Use forward slashes when building remote Hopsworks URI (avoid os.path.join)
        remote_path = f"{target_dir}/{os.path.basename(csv_path)}"
        print(f"Uploading CSV fallback to project dataset: {remote_path}")
        dataset_api.upload(csv_path, remote_path)
        print("Fallback upload complete.")

        # cleanup temp file if created
        if created_temp:
            try:
                os.remove(csv_path)
            except Exception:
                pass
        return True
    except Exception:
        print("Fallback dataset upload failed — traceback:")
        traceback.print_exc()
        return False