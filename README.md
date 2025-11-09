# Karachi Air Quality Forecast (Streamlit)

A Streamlit dashboard and MLOps pipeline for **hourly AQI forecasting in Karachi**.
Built with **OpenWeather**, **Hopsworks Feature Store**, and **GitHub Actions** for automated ingestion, daily retraining, and live predictions.
The dashboard displays **current AQI** and a **72-hour forecast** via an in-process predictor.

---

## 1. Overview

* Fetches hourly OpenWeather data (12-month history)
* Engineers ~77 features from 10 raw weather/pollutant variables
* Trains models daily and stores artifacts in Hopsworks
* Predicts AQI up to 72 h ahead (recursive routine)
* Streamlit app visualizes current and forecasted AQI

---

## 2. Data & Features

**Source:** OpenWeather API (hourly data)
**Raw (10):** temperature, humidity, wind speed, pressure, PM2.5, PM10, NO₂, O₃, CO, visibility
**Engineered (~77):**

* Time-based (hour, weekday, hour_sin, etc.)
* Lags (1 h–72 h)
* Rolling stats (mean/std/max)
* Change rates & interactions

**Target:** `standard_aqi_next_24h`
AQI computed via **EPA linear interpolation** between pollutant concentration breakpoints.

---

## 3. Missing Value Handling

* **Numeric:** forward-fill → backward-fill → median fill
* **Categorical:** filled with `"Unknown"`
* **Future leakage prevention:** exclude future targets (e.g., `next_1h`, `next_12h`) from inputs
* Ensures continuity in hourly series and robust model training

---

## 4. Model Training

* 80/20 time-ordered split
* Models: Random Forest, XGBoost, LightGBM, SARIMAX
* **Best:** Random Forest (balanced accuracy/generalization)

| Model   | Train R² | Test R² | MAE  | RMSE |
| ------- | -------- | ------- | ---- | ---- |
| RF      | 0.93     | 0.70    | 17.6 | 21.8 |
| XGB     | 0.99     | 0.57    | 21.3 | 26.1 |
| LGBM    | 0.99     | 0.59    | 21.0 | 25.3 |
| SARIMAX | 0.96     | −3.1    | 57.7 | 80.1 |

Artifacts saved in **Hopsworks Model Registry**.

---

## 5. Forecasting Pipeline

`predict.py` recursively forecasts the next 72 h:

1. Load model & recent features
2. Predict hour t + 1, append result do recursive prediction for 72h.
3. Repeat recursively → save `predictions.csv`
   Best accuracy within 24 h; gradual drift beyond.

---

## 6. Automation (CI/CD)

**GitHub Actions:**

* `hourly_ingest_pip.yml` → hourly ingestion
* `train_daily.yml` → daily retraining (02:00 UTC)

Steps: setup Python 3.12 · install deps · load secrets · run ingestion/training · upload logs/artifacts.

---

## 7. Streamlit Dashboard

**File:** `streamlit_app.py`
Displays:

* Current AQI (live OpenWeather)
* 72-hour forecast & pollutant charts
* Optional SHAP explanations

Falls back to synthetic data if APIs/Hopsworks unavailable.
Run:

```bash
streamlit run streamlit_app.py
```

→ [http://localhost:000](http://localhost:0009)

---

## 8. Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM · Statsmodels
Hopsworks · Streamlit · Plotly · SHAP · GitHub Actions

---

## 9. Developer Notes

* Extend predictor to return pollutant forecasts (pm2_5, pm10).
* Precompute SHAP summaries offline for faster load.
* Increase Hopsworks timeout or retry on errors.
* Disable synthetic fallback in production.

---

**Maintainer:** [@humaila0](https://github.com/humaila0)
*End-to-end AQI forecasting system with automated retraining, feature store integration, and dashboard.*
