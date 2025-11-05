"""
Preprocess backfilled CSV into features_preprocessed.csv.

- Reads /content/features_with_standard_aqi.csv (or BACKFILL_CSV env var)
- Computes standard_aqi_next_1h, standard_aqi_next_24h, standard_aqi_next_72h if missing
- Drops the specific future-target columns requested by user:
    standard_aqi_next_1h, standard_aqi_next_16h, standard_aqi_next_72h
  (drop is silent if a column does not exist)
- Adds lags/rolls and simple engineered features
- Fills NaNs: forward -> backward -> median
- Writes features_preprocessed.csv
"""
import os
from datetime import datetime
import pandas as pd
import numpy as np

INPUT = os.getenv("BACKFILL_CSV", "features_with_standard_aqi.csv")
OUTPUT = os.getenv("PREPROCESSED_CSV", "features_preprocessed.csv")

# Columns to drop from preprocessed output (no leakage from these into the FS)
DROP_TARGETS = ["standard_aqi_next_1h", "standard_aqi_next_16h", "standard_aqi_next_72h"]

def add_time_and_basic_features(df):
    df = df.sort_values("time").reset_index(drop=True)
    # time-based features
    if 'time' in df.columns:
        df['hour'] = df['time'].dt.hour
        df['day'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['year'] = df['time'].dt.year
        df['weekday'] = df['time'].dt.weekday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    return df

def add_lags_rolls(df):
    # AQI lags and rolls
    lags = [1, 3, 6, 12, 24, 48, 72]
    rolls = [3, 6, 12, 24, 48, 72]
    for lag in lags:
        col = f'standard_aqi_lag_{lag}'
        if col not in df.columns:
            df[col] = df['standard_aqi'].shift(lag)
    for w in rolls:
        df[f'standard_aqi_roll_{w}'] = df['standard_aqi'].rolling(window=w, min_periods=1).mean()
        df[f'standard_aqi_roll_std_{w}'] = df['standard_aqi'].rolling(window=w, min_periods=1).std().fillna(0)
    # change / pct change
    diffs = [1,3,6,12,24]
    for d in diffs:
        change_col = f'standard_aqi_change_{d}h'
        pct_col = f'standard_aqi_pct_change_{d}h'
        df[change_col] = df['standard_aqi'] - df['standard_aqi'].shift(d)
        denom = df['standard_aqi'].shift(d).replace(0, np.nan)
        df[pct_col] = 0.0
        df.loc[denom.notna(), pct_col] = (df.loc[denom.notna(), change_col] / denom.loc[denom.notna()]) * 100
    # pollutant lags & 24h rolls (if pollutants exist)
    for pollutant in ['pm2_5','pm10','no2','o3']:
        if pollutant in df.columns:
            for lag in [1,3,6,24]:
                df[f'{pollutant}_lag_{lag}'] = df[pollutant].shift(lag)
            df[f'{pollutant}_roll_24'] = df[pollutant].rolling(window=24, min_periods=1).mean()
    # interaction features
    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['temp_humidity'] = df['temperature'] * df['humidity']
    if 'pm2_5' in df.columns and 'pm10' in df.columns:
        df['pm_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
    if 'o3' in df.columns and 'no2' in df.columns:
        df['oxidant_sum'] = df['o3'] + df['no2']
    return df

def preprocess(input_path=INPUT, output_path=OUTPUT):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Run backfill first.")
    print(f"\n=== PREPROCESSING AQI DATA ===\nReading: {input_path}")
    df = pd.read_csv(input_path, parse_dates=['time'])
    # Basic cleaning + time-based features
    df = add_time_and_basic_features(df)

    # Ensure standard_aqi exists
    if 'standard_aqi' not in df.columns:
        raise ValueError("standard_aqi column missing in input CSV")

    # Compute targets (if missing)
    if 'standard_aqi_next_1h' not in df.columns:
        df['standard_aqi_next_1h'] = df['standard_aqi'].shift(-1)
    if 'standard_aqi_next_24h' not in df.columns:
        df['standard_aqi_next_24h'] = df['standard_aqi'].shift(-24)
    if 'standard_aqi_next_72h' not in df.columns:
        df['standard_aqi_next_72h'] = df['standard_aqi'].shift(-72)
    # Note: we compute 1h/24h/72h here for completeness, but will drop specific ones per request below.

    # Add lags, rolls, diffs, interactions
    df = add_lags_rolls(df)

    # --- Drop ONLY the requested future-target columns to prevent leakage from them in the preprocessed file ---
    # User requested dropping: standard_aqi_next_1h, standard_aqi_next_16h, standard_aqi_next_72h
    # We'll drop standard_aqi_next_1h and standard_aqi_next_72h if present.
    # standard_aqi_next_16h may not exist; drop harmlessly if present.
    to_drop = [c for c in DROP_TARGETS if c in df.columns]
    if to_drop:
        print(f"Dropping these future target columns from preprocessed file (to avoid leakage): {to_drop}")
        df = df.drop(columns=to_drop, errors='ignore')

    # For training we use the 24h target (standard_aqi_next_24h), so drop rows where it's missing
    if 'standard_aqi_next_24h' not in df.columns:
        raise ValueError("standard_aqi_next_24h target missing after preprocessing (cannot proceed)")
    before = len(df)
    df = df.dropna(subset=['standard_aqi_next_24h']).reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before - after} rows missing standard_aqi_next_24h (training target)")

    # Fill numeric NaNs: forward -> backward -> median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(method='ffill').fillna(method='bfill')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill object columns with 'Unknown'
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    df[obj_cols] = df[obj_cols].fillna('Unknown')

    # Ensure time_ms exists
    if 'time_ms' not in df.columns:
        df['time_ms'] = (pd.to_datetime(df['time']).astype('int64') // 10 ** 6)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved preprocessed features to {output_path} ({len(df)} rows)")

    return df

if __name__ == "__main__":
    preprocess()