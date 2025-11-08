"""
Train RandomForest for standard_aqi_next_24h,
save artifacts locally and upload the RandomForest to Hopsworks Model Registry.

Behavior:
 - Loads environment (.env) via python-dotenv (so HOPSWORKS_* and other flags can be set there).
 - Builds lags/rolls/diffs BEFORE dropping target rows (avoids leakage).
 - Trains RandomForest only by default (optional XGBoost / LightGBM if TRAIN_EXTRA_MODELS=true).
 - Saves artifacts to models/: RF model, scaler, feature column order, feature medians.
 - Attempts to register the RF model in Hopsworks model registry if test R² >= REGISTRY_MIN_R2.
 - Prints detailed errors on registry failures (doesn't crash the training run).
"""
import os
import joblib
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Load .env early
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# optional models
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor  # type: ignore
except Exception:
    LGBMRegressor = None

# optional hopsworks
try:
    import hopsworks  # type: ignore
except Exception:
    hopsworks = None

# CONFIG (can be overridden via .env)
INPUT_CSV = os.getenv("INPUT_CSV", "features_with_standard_aqi.csv")
TARGET = os.getenv("TARGET", "standard_aqi_next_24h")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
REGISTRY_MIN_R2 = float(os.getenv("REGISTRY_MIN_R2", "0.70"))  # only upload if test R² >= threshold
HOPSWORKS_MODEL_NAME = os.getenv("HOPSWORKS_MODEL_NAME", f"aqi_rf_{TARGET}")

# Control whether to train extra models (XGBoost / LightGBM). Default: False (RF only).
TRAIN_EXTRA_MODELS = os.getenv("TRAIN_EXTRA_MODELS", "false").lower() in ("1", "true", "yes")

os.makedirs(MODEL_DIR, exist_ok=True)

RF_MODEL_FILE = os.path.join(MODEL_DIR, f"rf_aqi_predictor_{TARGET}.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, f"scaler_rf_{TARGET}.pkl")
FEATCOLS_FILE = os.path.join(MODEL_DIR, f"rf_feature_cols_{TARGET}.pkl")
FEATURE_MEDIANS_FILE = os.path.join(MODEL_DIR, f"feature_medians_{TARGET}.pkl")

# Feature engineering config
LAGS = [1, 3, 6, 12, 24, 48, 72]
ROLLS = [3, 6, 12, 24, 48, 72]
DIFFS = [1, 3, 6, 12, 24]

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
RF_PARAMS = {
    "n_estimators": int(os.getenv("RF_N_ESTIMATORS", "1000")),
    "max_depth": int(os.getenv("RF_MAX_DEPTH", "15")),
    "min_samples_split": int(os.getenv("RF_MIN_SAMPLES_SPLIT", "5")),
    "min_samples_leaf": int(os.getenv("RF_MIN_SAMPLES_LEAF", "3")),
    "max_features": os.getenv("RF_MAX_FEATURES", "sqrt"),
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


def calc_metrics(y_true, y_pred):
    """
    Compute common regression metrics in a way compatible with multiple scikit-learn versions.
    Returns dict with r2, mae, rmse.
    """
    r2_val = float(r2_score(y_true, y_pred))
    mae_val = float(mean_absolute_error(y_true, y_pred))

    # Compute RMSE without using the 'squared' kwarg to remain compatible across sklearn versions
    try:
        mse_val = float(mean_squared_error(y_true, y_pred))
        rmse_val = float(np.sqrt(mse_val))
    except Exception:
        # Defensive fallback: manual computation
        try:
            arr = np.asarray(y_true) - np.asarray(y_pred)
            mse_val = float(np.mean(arr ** 2))
            rmse_val = float(np.sqrt(mse_val))
        except Exception:
            rmse_val = float("nan")

    return {"r2": r2_val, "mae": mae_val, "rmse": rmse_val}


# ---------- data / feature engineering ----------
def load_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run preprocess/backfill first.")
    df = pd.read_csv(path, parse_dates=["time"])
    return df


def add_features(df):
    # ensure sorted by time
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if "standard_aqi" not in df.columns:
        raise ValueError("standard_aqi column missing in input")

    # lags
    for lag in LAGS:
        df[f"aqi_lag_{lag}"] = df["standard_aqi"].shift(lag)

    # rolling stats
    for r in ROLLS:
        df[f"aqi_roll_mean_{r}"] = df["standard_aqi"].rolling(window=r, min_periods=1).mean()
        df[f"aqi_roll_std_{r}"] = df["standard_aqi"].rolling(window=r, min_periods=1).std().fillna(0)

    # diffs & pct changes
    for d in DIFFS:
        df[f"aqi_change_{d}h"] = df["standard_aqi"] - df["standard_aqi"].shift(d)
        denom = df["standard_aqi"].shift(d).replace(0, np.nan)
        pct_col = f"aqi_pct_change_{d}h"
        df[pct_col] = 0.0
        df.loc[denom.notna(), pct_col] = (df.loc[denom.notna(), f"aqi_change_{d}h"] / denom.loc[denom.notna()]) * 100

    # interactions and cyclic time features
    if {"temperature", "humidity"}.issubset(df.columns):
        df["temp_humidity"] = df["temperature"] * df["humidity"]
    if {"pm2_5", "pm10"}.issubset(df.columns):
        df["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)
    if {"no2", "o3"}.issubset(df.columns):
        df["oxidant_sum"] = df["no2"] + df["o3"]

    if "time" in df.columns:
        df["hour"] = df["time"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day"] = df["time"].dt.day
        df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
        df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)

    return df


def prepare_targets_and_clean(df):
    # compute target if missing
    if TARGET not in df.columns and "standard_aqi" in df.columns:
        try:
            n = int("".join(filter(str.isdigit, TARGET)))
        except Exception:
            n = 24
        df[TARGET] = df["standard_aqi"].shift(-n)

    # drop known leakage cols
    leak_cols = ["standard_aqi_next_6h", "standard_aqi_next_12h", "standard_aqi_next_72h", "standard_aqi_next_1h"]
    to_drop = [c for c in leak_cols if c in df.columns]
    if to_drop:
        print("Dropping leakage columns from df:", to_drop)
        df = df.drop(columns=to_drop, errors="ignore")

    # drop rows missing target
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    # clean numeric NaNs / inf
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].ffill().bfill()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df


def build_X_y(df):
    drop_cols = [TARGET, "time", "time_ms"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()
    y = df[TARGET].copy()
    return X, y


# ---------- Hopsworks registry uploader ----------
def register_model_hopsworks(model_path: str, metrics: dict):
    """
    Upload model_path to Hopsworks Model Registry as a timestamped model name.
    Returns True on success, False on failure. Prints detailed tracebacks on error.
    """
    if hopsworks is None:
        print("Hopsworks SDK not installed; skipping registry upload.")
        return False

    try:
        # login (relies on load_dotenv and env vars like HOPSWORKS_API_KEY/HOPSWORKS_HOST)
        project = hopsworks.login()
    except Exception:
        print("Hopsworks login failed — traceback:")
        traceback.print_exc()
        return False

    try:
        mr = project.get_model_registry()
    except Exception:
        print("Failed to get model registry client — traceback:")
        traceback.print_exc()
        return False

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_name = f"{HOPSWORKS_MODEL_NAME}_{ts}"

    try:
        print(f"Creating model entry '{model_name}' in Hopsworks...")
        model_entity = mr.python.create_model(name=model_name, metrics=metrics or {}, description=f"Auto-upload {HOPSWORKS_MODEL_NAME}")
    except Exception:
        print("create_model failed — traceback:")
        traceback.print_exc()
        return False

    try:
        print("Uploading model file to registry:", model_path)
        model_entity.save(model_path)
        print("Uploaded model file.")
    except Exception:
        print("model_entity.save() failed — traceback:")
        traceback.print_exc()
        return False

    # attach artifacts if they exist (best-effort)
    for extra in [SCALER_FILE, FEATCOLS_FILE, FEATURE_MEDIANS_FILE]:
        if os.path.exists(extra):
            try:
                try:
                    model_entity.add_file(extra)
                    print("Attached artifact via add_file():", extra)
                except Exception:
                    model_entity.save(extra)
                    print("Attached artifact via save():", extra)
            except Exception:
                print(f"Failed to attach artifact {extra} — traceback:")
                traceback.print_exc()

    return True


# ---------- main ----------
def main():
    print("Starting training:", datetime.utcnow().isoformat())

    df = load_df(INPUT_CSV)
    print("Loaded rows:", len(df))

    # 1) features
    df = add_features(df)

    # 2) target & clean
    df = prepare_targets_and_clean(df)
    print("Rows after prepare:", len(df))

    # 3) X, y
    X, y = build_X_y(df)
    print("Feature matrix shape:", X.shape)

    # 4) chronological split
    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split].copy(), X.iloc[split:].copy()
    y_train, y_test = y.iloc[:split].copy(), y.iloc[split:].copy()
    print("Train rows:", len(X_train), "Test rows:", len(X_test))

    # 5) scale
    feat_cols = X_train.columns.tolist()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train[feat_cols])
    X_test_s = scaler.transform(X_test[feat_cols])

    # 6) train RF
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train_s, y_train)
    y_tr_rf = rf.predict(X_train_s)
    y_te_rf = rf.predict(X_test_s)
    rf_metrics_train = calc_metrics(y_train, y_tr_rf)
    rf_metrics_test = calc_metrics(y_test, y_te_rf)

    print("\nRandomForest → Train R²: {:.4f}, Test R²: {:.4f}, MAE(test): {:.3f}, RMSE(test): {:.3f}".format(
        rf_metrics_train["r2"], rf_metrics_test["r2"], rf_metrics_test["mae"], rf_metrics_test["rmse"]
    ))

    # 7) optional XGBoost / LightGBM (only run if TRAIN_EXTRA_MODELS=true)
    xgb_res = None
    if TRAIN_EXTRA_MODELS and XGBRegressor is not None:
        try:
            xgb = XGBRegressor(objective="reg:squarederror", n_estimators=500, max_depth=6,
                               learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                               random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
            xgb.fit(X_train_s, y_train)
            xgb_res = {
                "train": calc_metrics(y_train, xgb.predict(X_train_s)),
                "test": calc_metrics(y_test, xgb.predict(X_test_s))
            }
            print("XGBoost      → Train R²: {:.4f}, Test R²: {:.4f}".format(xgb_res["train"]["r2"], xgb_res["test"]["r2"]))
            joblib.dump(xgb, os.path.join(MODEL_DIR, f"xgb_aqi_predictor_{TARGET}.pkl"))
        except Exception:
            print("XGBoost training failed — traceback:")
            traceback.print_exc()

    lgbm_res = None
    if TRAIN_EXTRA_MODELS and LGBMRegressor is not None:
        try:
            lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE)
            lgbm.fit(X_train_s, y_train)
            lgbm_res = {
                "train": calc_metrics(y_train, lgbm.predict(X_train_s)),
                "test": calc_metrics(y_test, lgbm.predict(X_test_s))
            }
            print("LightGBM     → Train R²: {:.4f}, Test R²: {:.4f}".format(lgbm_res["train"]["r2"], lgbm_res["test"]["r2"]))
            joblib.dump(lgbm, os.path.join(MODEL_DIR, f"lgbm_aqi_predictor_{TARGET}.pkl"))
        except Exception:
            print("LightGBM training failed — traceback:")
            traceback.print_exc()

    # 8) Save artifacts (joblib)
    try:
        joblib.dump(rf, RF_MODEL_FILE)
        print("Saved RF model to:", RF_MODEL_FILE)
    except Exception:
        print("Failed to save RF model locally — traceback:")
        traceback.print_exc()

    try:
        joblib.dump(scaler, SCALER_FILE)
        print("Saved scaler to:", SCALER_FILE)
    except Exception:
        print("Failed to save scaler locally — traceback:")
        traceback.print_exc()

    try:
        joblib.dump(feat_cols, FEATCOLS_FILE)
        print("Saved feature columns list to:", FEATCOLS_FILE)
    except Exception:
        print("Failed to save feature columns locally — traceback:")
        traceback.print_exc()

    # save medians for single-row inference
    try:
        feature_medians = X_train[feat_cols].median()
        joblib.dump(feature_medians, FEATURE_MEDIANS_FILE)
        print("Saved feature medians to:", FEATURE_MEDIANS_FILE)
    except Exception:
        print("Failed to save feature medians — traceback:")
        traceback.print_exc()

    # 9) Registry upload guard + attempt
    rf_test_r2 = rf_metrics_test.get("r2")
    print(f"\nRF test R² = {rf_test_r2:.4f}; upload threshold = {REGISTRY_MIN_R2:.2f}")
    if rf_test_r2 is not None and rf_test_r2 >= REGISTRY_MIN_R2:
        print("Test R² meets threshold — attempting to upload model to Hopsworks Model Registry...")
        uploaded = register_model_hopsworks(RF_MODEL_FILE, rf_metrics_test)
        print("Registry upload attempted:", uploaded)
    else:
        print("Skipping registry upload because RF test R² is below threshold or missing.")

    print("\nSaved artifacts to:", MODEL_DIR)
    return {
        "rf_train_metrics": rf_metrics_train,
        "rf_test_metrics": rf_metrics_test,
        "xgb": xgb_res,
        "lgbm": lgbm_res
    }


if __name__ == "__main__":
    out = main()
    print("\nDone at", datetime.utcnow().isoformat())
