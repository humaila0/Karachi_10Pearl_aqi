import pandas as pd
import numpy as np

# EPA Breakpoints for PM2.5 (μg/m³)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),  # Good
    (12.1, 35.4, 51, 100),  # Moderate
    (35.5, 55.4, 101, 150),  # Unhealthy for Sensitive Groups
    (55.5, 150.4, 151, 200),  # Unhealthy
    (150.5, 250.4, 201, 300),  # Very Unhealthy
    (250.5, 500.4, 301, 500)  # Hazardous
]

# EPA Breakpoints for PM10 (μg/m³)
PM10_BREAKPOINTS = [
    (0, 54, 0, 50),  # Good
    (55, 154, 51, 100),  # Moderate
    (155, 254, 101, 150),  # Unhealthy for Sensitive Groups
    (255, 354, 151, 200),  # Unhealthy
    (355, 424, 201, 300),  # Very Unhealthy
    (425, 604, 301, 500)  # Hazardous
]


def sub_aqi(C, breakpoints):
    """Calculate AQI for a single pollutant concentration"""
    if C is None:
        return None

    for C_lo, C_hi, I_lo, I_hi in breakpoints:
        if C_lo <= C <= C_hi:
            return (I_hi - I_lo) / (C_hi - C_lo) * (C - C_lo) + I_lo

    # If concentration is above the highest breakpoint
    if C > breakpoints[-1][1]:
        return breakpoints[-1][3]

    return None


def compute_aqi_row(row):
    """Compute AQI from a row of data (works with both OpenMeteo and OpenWeather)"""
    # Handle both pm2_5 and pm10 columns (case insensitive)
    pm25 = row.get('pm2_5') if 'pm2_5' in row else row.get('PM2_5')
    pm10 = row.get('pm10') if 'pm10' in row else row.get('PM10')

    # Calculate individual AQIs
    a_pm25 = sub_aqi(pm25, PM25_BREAKPOINTS) if pm25 is not None else None
    a_pm10 = sub_aqi(pm10, PM10_BREAKPOINTS) if pm10 is not None else None

    # Get the maximum AQI value
    candidates = [v for v in [a_pm25, a_pm10] if v is not None]
    if not candidates:
        return None

    return max(candidates)


# Changed function name from get_aqi_category to aqi_category for compatibility
def aqi_category(aqi):
    """Get the AQI category string based on the value"""
    if aqi is None:
        return 'Unknown'
    elif aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'


def convert_openweather_aqi_to_category(ow_aqi):
    """Convert OpenWeather's 1-5 scale AQI to a category string"""
    categories = {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor"
    }
    return categories.get(ow_aqi, "Unknown")


def convert_openweather_aqi_to_standard(ow_aqi, pm25, pm10):
    """
    Convert OpenWeather's 1-5 scale AQI to standard AQI by calculating from pollutant values.
    Falls back to estimation if pollutant values are missing.
    """
    # If we have pollutant values, use the precise calculation
    if pm25 is not None and pm10 is not None:
        a_pm25 = sub_aqi(pm25, PM25_BREAKPOINTS)
        a_pm10 = sub_aqi(pm10, PM10_BREAKPOINTS)
        if a_pm25 is not None and a_pm10 is not None:
            return max(a_pm25, a_pm10)

    # Fallback: rough estimate based on OpenWeather scale
    # This is a very rough mapping and should only be used if no pollutant data is available
    ow_to_standard = {
        1: 25,  # Good -> mid-Good
        2: 75,  # Fair -> mid-Moderate
        3: 125,  # Moderate -> mid-USG
        4: 175,  # Poor -> mid-Unhealthy
        5: 300  # Very Poor -> Very Unhealthy
    }
    return ow_to_standard.get(ow_aqi, 150)