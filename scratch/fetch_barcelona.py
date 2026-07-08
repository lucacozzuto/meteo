import openmeteo_requests
import requests_cache
import pandas as pd
import time
from datetime import date, timedelta
from retry_requests import retry

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=2.0)
openmeteo = openmeteo_requests.Client(session=retry_session)

url_archive = "https://archive-api.open-meteo.com/v1/archive"
url_forecast = "https://api.open-meteo.com/v1/forecast"

definitive_end_date_target = (date.today() - timedelta(days=5)).strftime('%Y-%m-%d')
provisional_end_date_target = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')

coords = (41.3851, 2.1734)

def fetch_data(url, params):
    while True:
        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            daily = response.Daily()
            if daily.Variables(0).ValuesAsNumpy().size == 0:
                return None
            daily_data = {"date": pd.date_range(
                start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
                end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = daily.Interval()),
                inclusive = "left"
            )}
            daily_data["temperature_2m_max"] = daily.Variables(0).ValuesAsNumpy()
            daily_data["temperature_2m_min"] = daily.Variables(1).ValuesAsNumpy()
            daily_data["precipitation_sum"] = daily.Variables(2).ValuesAsNumpy()
            return pd.DataFrame(data=daily_data)
        except Exception as e:
            print(f"Error encountered: {e}. Sleeping 60s...")
            time.sleep(60)

print(f"Fetching definitive data for Barcelona...")
params_archive = {
    "latitude": coords[0],
    "longitude": coords[1],
    "start_date": "1940-01-01",
    "end_date": definitive_end_date_target,
    "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
    "timezone": "auto"
}
df_archive = fetch_data(url_archive, params_archive)
df_archive['date'] = pd.to_datetime(df_archive['date']).dt.tz_localize(None)
df_archive['is_historical'] = 'T'

print(f"Fetching provisional data for Barcelona...")
start_date_forecast = (pd.to_datetime(definitive_end_date_target) + timedelta(days=1)).strftime('%Y-%m-%d')
params_forecast = {
    "latitude": coords[0],
    "longitude": coords[1],
    "start_date": start_date_forecast,
    "end_date": provisional_end_date_target,
    "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
    "timezone": "auto"
}
df_forecast = fetch_data(url_forecast, params_forecast)
df_forecast['date'] = pd.to_datetime(df_forecast['date']).dt.tz_localize(None)
df_forecast['is_historical'] = 'F'

df = pd.concat([df_archive, df_forecast], ignore_index=True)
df['date'] = df['date'].dt.strftime('%Y-%m-%d 00:00:00+00:00')
df.to_csv('data/Barcelona.csv', index=False)
print("Barcelona created successfully.")
