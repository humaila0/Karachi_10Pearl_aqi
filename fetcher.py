"""
OpenWeather fetch helpers (timestamps normalized to timezone-aware UTC)

This file replaces datetime.fromtimestamp(...) with timezone-aware conversions so all
returned 'time' values are datetime objects with tzinfo=UTC.

Usage: import and call fetch_openweather_current / fetch_openweather_historical / fetch_openweather_forecast
"""
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import time

# API endpoints
OPENWEATHER_CUR = "https://api.openweathermap.org/data/2.5/air_pollution"
OPENWEATHER_HIST = "https://api.openweathermap.org/data/2.5/air_pollution/history"
OPENWEATHER_FORECAST = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"


def _ts_to_utc(ts):
    """Convert Unix epoch seconds (int) to timezone-aware UTC datetime."""
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc)
    except Exception:
        # fallback
        return datetime.utcfromtimestamp(int(ts)).replace(tzinfo=timezone.utc)


def fetch_openweather_historical(lat, lon, start_date, end_date, api_key):
    """
    Fetch historical air quality data from OpenWeather API

    Returns a DataFrame with a timezone-aware UTC 'time' column.
    """
    print(f"Fetching historical data from {start_date} to {end_date}...")

    # Convert to Unix timestamps (required by OpenWeather API)
    start_timestamp = int(time.mktime(start_date.timetuple()))
    end_timestamp = int(time.mktime(end_date.timetuple()))

    # Construct the URL
    url = f"{OPENWEATHER_HIST}?lat={lat}&lon={lon}&start={start_timestamp}&end={end_timestamp}&appid={api_key}"

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if 'list' in data and len(data['list']) > 0:
        record_count = len(data['list'])
        print(f"✅ Retrieved {record_count} historical records")

        records = []
        for item in data['list']:
            timestamp = _ts_to_utc(item['dt'])
            components = item.get('components', {})
            original_aqi = item.get('main', {}).get('aqi')

            record = {
                'time': timestamp,
                'pm2_5': components.get('pm2_5'),
                'pm10': components.get('pm10'),
                'co': components.get('co'),
                'no': components.get('no', 0),
                'no2': components.get('no2'),
                'o3': components.get('o3'),
                'so2': components.get('so2'),
                'nh3': components.get('nh3', 0),
                'original_aqi': original_aqi  # OpenWeather's 1-5 scale
            }
            records.append(record)

        df = pd.DataFrame(records)
        return df
    else:
        print("No data received from OpenWeather API")
        return pd.DataFrame()


def fetch_openweather_current(lat, lon, api_key):
    """
    Fetch current air quality data from OpenWeather API

    Returns a DataFrame with a single timezone-aware UTC 'time' row.
    """
    print("Fetching current air quality data...")

    url = f"{OPENWEATHER_CUR}?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if 'list' in data and len(data['list']) > 0:
        item = data['list'][0]
        timestamp = _ts_to_utc(item['dt'])
        components = item.get('components', {})
        original_aqi = item.get('main', {}).get('aqi')

        record = {
            'time': timestamp,
            'pm2_5': components.get('pm2_5'),
            'pm10': components.get('pm10'),
            'co': components.get('co'),
            'no': components.get('no', 0),
            'no2': components.get('no2'),
            'o3': components.get('o3'),
            'so2': components.get('so2'),
            'nh3': components.get('nh3', 0),
            'original_aqi': original_aqi  # OpenWeather's 1-5 scale
        }

        return pd.DataFrame([record])
    else:
        print("No data received from OpenWeather API")
        return pd.DataFrame()


def fetch_openweather_forecast(lat, lon, api_key):
    """
    Fetch air quality forecast data from OpenWeather API

    Returns timezone-aware UTC 'time' values.
    """
    print("Fetching forecast air quality data...")

    url = f"{OPENWEATHER_FORECAST}?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if 'list' in data and len(data['list']) > 0:
        records = []
        for item in data['list']:
            timestamp = _ts_to_utc(item['dt'])
            components = item.get('components', {})
            original_aqi = item.get('main', {}).get('aqi')

            record = {
                'time': timestamp,
                'pm2_5': components.get('pm2_5'),
                'pm10': components.get('pm10'),
                'co': components.get('co'),
                'no': components.get('no', 0),
                'no2': components.get('no2'),
                'o3': components.get('o3'),
                'so2': components.get('so2'),
                'nh3': components.get('nh3', 0),
                'original_aqi': original_aqi
            }
            records.append(record)

        df = pd.DataFrame(records)
        print(f"Retrieved {len(df)} forecast records")
        return df
    else:
        print("No forecast data received from OpenWeather API")
        return pd.DataFrame()


# Legacy function for backward compatibility
def fetch_open_meteo(*args, **kwargs):
    print("WARNING: fetch_open_meteo is deprecated. Use fetch_openweather_historical instead.")
    raise NotImplementedError("This function is no longer supported.")