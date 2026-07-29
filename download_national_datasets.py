import openmeteo_requests
import requests_cache
import pandas as pd
import os
from datetime import date, timedelta
from retry_requests import retry

def fetch_dataset_for_location(lat, lon, filename, out_dir):
    print(f"Downloading dataset for {filename} at ({lat}, {lon})...")
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=2.0)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    forecast_url = "https://api.open-meteo.com/v1/forecast"

    definitive_end = (date.today() - timedelta(days=5)).strftime('%Y-%m-%d')
    provisional_end = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    provisional_start = (date.today() - timedelta(days=4)).strftime('%Y-%m-%d')

    # Fetch Archive (Historical)
    params_archive = {
        'latitude': lat,
        'longitude': lon,
        'start_date': '1940-01-01',
        'end_date': definitive_end,
        'daily': ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum'],
        'timezone': 'auto'
    }
    responses = openmeteo.weather_api(archive_url, params=params_archive)
    daily_arch = responses[0].Daily()

    offset_arch = responses[0].UtcOffsetSeconds()
    df_arch = pd.DataFrame({
        "date": pd.date_range(
            start=pd.to_datetime(daily_arch.Time() + offset_arch, unit="s", utc=True),
            end=pd.to_datetime(daily_arch.TimeEnd() + offset_arch, unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily_arch.Interval()),
            inclusive="left"
        ),
        "temperature_2m_max": daily_arch.Variables(0).ValuesAsNumpy(),
        "temperature_2m_min": daily_arch.Variables(1).ValuesAsNumpy(),
        "precipitation_sum": daily_arch.Variables(2).ValuesAsNumpy(),
        "is_historical": "T"
    })

    # Fetch Forecast (Provisional for recent days)
    params_forecast = {
        'latitude': lat,
        'longitude': lon,
        'start_date': provisional_start,
        'end_date': provisional_end,
        'daily': ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum'],
        'timezone': 'auto'
    }
    responses_fc = openmeteo.weather_api(forecast_url, params=params_forecast)
    daily_fc = responses_fc[0].Daily()

    offset_fc = responses_fc[0].UtcOffsetSeconds()
    df_fc = pd.DataFrame({
        "date": pd.date_range(
            start=pd.to_datetime(daily_fc.Time() + offset_fc, unit="s", utc=True),
            end=pd.to_datetime(daily_fc.TimeEnd() + offset_fc, unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily_fc.Interval()),
            inclusive="left"
        ),
        "temperature_2m_max": daily_fc.Variables(0).ValuesAsNumpy(),
        "temperature_2m_min": daily_fc.Variables(1).ValuesAsNumpy(),
        "precipitation_sum": daily_fc.Variables(2).ValuesAsNumpy(),
        "is_historical": "F"
    })

    # Filter forecast dates > archive max date
    max_arch_date = df_arch['date'].max()
    df_fc = df_fc[df_fc['date'] > max_arch_date]

    df_full = pd.concat([df_arch, df_fc], ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    df_full.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(df_full)} rows (dates: {df_full['date'].min().strftime('%Y-%m-%d')} to {df_full['date'].max().strftime('%Y-%m-%d')}).")

def main():
    # Italia centroid (Rieti/Central Italy)
    fetch_dataset_for_location(42.5000, 12.5000, 'Italia.csv', 'data_italy')
    # Europa centroid (Central Europe)
    fetch_dataset_for_location(50.0000, 10.0000, 'Europa.csv', 'data')

if __name__ == '__main__':
    main()
