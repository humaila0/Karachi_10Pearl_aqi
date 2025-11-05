"""
Feature engineering utilities for AQI pipeline.

Usage:
 - engineer_features(df, target=None, drop_future_targets=True, return_last_row=False)

Notes:
 - This module only does feature engineering. It does NOT train models or upload to registry.
 - It will drop leakage columns matching 'standard_aqi_next_*' when drop_future_targets=True,
   except for the 'target' column name you provide.
"""
from typing import Optional
import numpy as np
import pandas as pd

# Feature engineering config (adjustable)
LAGS = [1, 3, 6, 12, 24, 48, 72]
ROLLS = [3, 6, 12, 24, 48, 72]
DIFFS = [1, 3, 6, 12, 24]
POLLUTANTS = ['pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2', 'nh3']


def _safe_pct_change(numerator, denominator):
    """Return percent change safely (avoid division by zero)."""
    denom = denominator.replace(0, np.nan)
    out = np.zeros_like(numerator, dtype=float)
    mask = denom.notna()
    out[mask] = (numerator[mask] / denom[mask]) * 100.0
    return out


def engineer_features(df: pd.DataFrame,
                      target: Optional[str] = None,
                      drop_future_targets: bool = True,
                      return_last_row: bool = False) -> pd.DataFrame:
    """
    Engineer features required by the model.

    Args:
      df: DataFrame with at least columns ['time', 'standard_aqi'] and pollutant columns if available.
          'time' must be parseable as datetime dtype.
      target: optional target column name to KEEP if present (e.g. 'standard_aqi_next_24h').
              When None, no future-target is preserved.
      drop_future_targets: if True, drop any column matching 'standard_aqi_next_*' except `target`.
      return_last_row: if True, return only the last engineered row (useful for ingesting a single record).

    Returns:
      DataFrame with engineered numeric features added.
    """
    df = df.copy()

    # ensure time is datetime and series is sorted
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # if standard_aqi missing, user should compute it before calling this function
    if 'standard_aqi' not in df.columns:
        raise ValueError("Input dataframe must contain 'standard_aqi' column before feature engineering.")

    # Drop future-target leakage columns if requested (keep explicit target if provided)
    if drop_future_targets:
        future_cols = [c for c in df.columns if c.startswith("standard_aqi_next_")]
        if target:
            future_cols = [c for c in future_cols if c != target]
        if future_cols:
            df = df.drop(columns=future_cols, errors='ignore')

    # Create lags for target (standard_aqi)
    for lag in LAGS:
        df[f"standard_aqi_lag_{lag}"] = df['standard_aqi'].shift(lag)

    # Rolling stats for target (min_periods=1 to allow early rows)
    for r in ROLLS:
        df[f"standard_aqi_roll_{r}"] = df['standard_aqi'].rolling(window=r, min_periods=1).mean()
        df[f"standard_aqi_roll_std_{r}"] = df['standard_aqi'].rolling(window=r, min_periods=1).std().fillna(0.0)

    # Differences and percent-changes for target
    for d in DIFFS:
        change_col = f"standard_aqi_change_{d}h"
        pct_col = f"standard_aqi_pct_change_{d}h"
        df[change_col] = df['standard_aqi'] - df['standard_aqi'].shift(d)
        # safe percent change
        shifted = df['standard_aqi'].shift(d)
        df[pct_col] = 0.0
        mask = shifted.replace(0, np.nan).notna()
        df.loc[mask, pct_col] = (df.loc[mask, change_col] / shifted.loc[mask]) * 100.0

    # Pollutant lags and 24h rolls if pollutant columns exist
    for pollutant in POLLUTANTS:
        if pollutant in df.columns:
            for lag in [1, 3, 6, 24]:
                df[f"{pollutant}_lag_{lag}"] = df[pollutant].shift(lag)
            df[f"{pollutant}_roll_24"] = df[pollutant].rolling(window=24, min_periods=1).mean()

    # Interaction features
    if {'temperature', 'humidity'}.issubset(df.columns):
        df['temp_humidity'] = df['temperature'] * df['humidity']
    if {'pm2_5', 'pm10'}.issubset(df.columns):
        # add small eps to avoid division by zero
        df['pm_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
    if {'no2', 'o3'}.issubset(df.columns):
        df['oxidant_sum'] = df['no2'] + df['o3']

    # Cyclic time features
    if 'time' in df.columns:
        df['hour'] = df['time'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day'] = df['time'].dt.day
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['weekday'] = df['time'].dt.weekday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        # rush hour indicator
        df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))).astype(int)
        df['morning'] = ((df['hour'] >= 6) & (df['hour'] <= 10)).astype(int)
        df['afternoon'] = ((df['hour'] >= 11) & (df['hour'] <= 16)).astype(int)
        df['evening'] = ((df['hour'] >= 17) & (df['hour'] <= 21)).astype(int)
        df['night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

    # After feature creation: replace inf/-inf and fill NaNs moderately (ffill->bfill->median)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    # Avoid dropping rows here — leave handling to training code; but fill so single-row inference works
    df[num_cols] = df[num_cols].ffill().bfill()
    medians = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)

    # For small inputs (single row) return the single engineered row if requested
    if return_last_row:
        return df.iloc[[-1]].reset_index(drop=True)

    return df