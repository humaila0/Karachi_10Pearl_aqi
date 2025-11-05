import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import requests
import time
from dotenv import load_dotenv
from compute_aqi import compute_aqi_row  # Import standard AQI computation logic

load_dotenv()
# OpenWeather API key
API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# Karachi coordinates
LAT, LON = 24.8607, 67.0011


def fetch_historical_data(days=365, user="humaila0"):
    """Fetch historical air quality and weather data, compute features, and save to CSV."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== FETCHING {days} DAYS OF HISTORICAL DATA ===")
    print(f"Started at: {current_time} UTC")
    print(f"User: {user}")
    print(f"Location: Karachi (Lat: {LAT}, Lon: {LON})")

    if not API_KEY:
        raise ValueError("OpenWeather API key not set. Set the OPENWEATHER_API_KEY environment variable.")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    all_records = []
    current_end = end_date

    # Break the request into 90-day chunks to avoid API limits
    while current_end > start_date:
        chunk_start = max(start_date, current_end - timedelta(days=90))
        print(f"\nFetching data from {chunk_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")

        # Convert to Unix timestamps
        chunk_start_ts = int(time.mktime(chunk_start.timetuple()))
        chunk_end_ts = int(time.mktime(current_end.timetuple()))

        # OpenWeather historical API endpoint
        url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={chunk_start_ts}&end={chunk_end_ts}&appid={API_KEY}"

        try:
            print("Requesting data from OpenWeather API...")
            response = requests.get(url)
            data = response.json()

            if 'list' not in data or len(data['list']) == 0:
                print(f"⚠️ No data received for this chunk or API error: {data.get('message', 'Unknown error')}")
            else:
                # Process records
                for item in data['list']:
                    timestamp = datetime.fromtimestamp(item['dt'])
                    components = item['components']
                    weather = item.get('weather', [{}])[0]  # Placeholder in case weather is missing

                    record = {
                        'time': timestamp,
                        'pm2_5': components.get('pm2_5'),
                        'pm10': components.get('pm10'),
                        'co': components.get('co'),
                        'no2': components.get('no2'),
                        'o3': components.get('o3'),
                        'so2': components.get('so2'),
                        'nh3': components.get('nh3', 0),
                        'temperature': item.get('temp', np.nan),  # Add weather data if available
                        'humidity': item.get('humidity', np.nan),
                        'wind_speed': item.get('wind_speed', np.nan),
                        'pressure': item.get('pressure', np.nan)
                    }

                    # Calculate standard AQI based on pollutants
                    record['standard_aqi'] = compute_aqi_row(record)
                    all_records.append(record)

            print(f"Retrieved {len(all_records)} records so far.")
            current_end = chunk_start - timedelta(seconds=1)

            # Add delay to avoid hitting API rate limits
            time.sleep(2)

        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            break

    # Create dataframe from all records
    if not all_records:
        print("No data was retrieved.")
        return None

    df = pd.DataFrame(all_records)
    print(f"\nCombined {len(df)} records from all chunks")

    # Sort data chronologically
    df = df.sort_values('time')

    # Add time-based features
    print("Adding time-based features...")
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['weekday'] = df['time'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    # Add lag and rolling features for standard AQI
    print("Creating lag, rolling, and interaction features...")
    for lag in [1, 3, 6, 12, 24, 48, 72]:
        df[f'standard_aqi_lag_{lag}'] = df['standard_aqi'].shift(lag)

    for window in [3, 6, 12, 24, 48, 72]:
        df[f'standard_aqi_roll_{window}'] = df['standard_aqi'].rolling(window, min_periods=1).mean()

    # Interaction features
    df['pm_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
    df['oxidant_sum'] = df['o3'] + df['no2']

    # Create AQI change rate features
    print("Calculating AQI change rate features...")
    df['standard_aqi_change_1h'] = df['standard_aqi'] - df['standard_aqi_lag_1']
    df['standard_aqi_change_24h'] = df['standard_aqi'] - df['standard_aqi_lag_24']
    df['standard_aqi_pct_change_1h'] = (df['standard_aqi_change_1h'] / df['standard_aqi_lag_1'].replace(0, np.nan)) * 100
    df['standard_aqi_pct_change_24h'] = (df['standard_aqi_change_24h'] / df['standard_aqi_lag_24'].replace(0, np.nan)) * 100

    # Replace infinities and NaN in change rates
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Create target variables for forecasting
    print("Creating target variables...")
    for h in [1, 6, 12, 24, 72]:
        df[f'standard_aqi_next_{h}h'] = df['standard_aqi'].shift(-h)

    # Create time_ms column for Hopsworks primary key
    df['time_ms'] = pd.to_datetime(df['time']).astype('int64') // 10 ** 6

    # Save to CSV
    csv_filename = 'features_with_standard_aqi.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Saved {len(df)} records to {csv_filename}")

    # Print summary stats
    print("\n=== DATA SUMMARY ===")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"Average PM2.5: {df['pm2_5'].mean():.2f} μg/m³")
    print(f"Average PM10: {df['pm10'].mean():.2f} μg/m³")
    print(f"Average Standard AQI: {df['standard_aqi'].mean():.2f}")

    return df


if __name__ == "__main__":
    # Get days from command line argument if provided
    try:
        if len(sys.argv) > 1:
            days = int(sys.argv[1])
        else:
            days = 365  # Default to full year
    except ValueError:
        print("Invalid number of days. Using default (365).")
        days = 365

    # Get username if provided in environment
    username = os.environ.get("USER", os.environ.get("USERNAME", "humaila0"))

    # Fetch the data
    fetch_historical_data(days=days, user=username)