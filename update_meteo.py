import openmeteo_requests
import requests_cache
import pandas as pd
import os
import time
from datetime import date, timedelta
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=2.0)
openmeteo = openmeteo_requests.Client(session=retry_session)

url_archive = "https://archive-api.open-meteo.com/v1/archive"
url_forecast = "https://api.open-meteo.com/v1/forecast"

# Definitive data comes from archive, which lags by 5 days
definitive_end_date_target = (date.today() - timedelta(days=5)).strftime('%Y-%m-%d')
# Provisional data comes from forecast, up to yesterday
provisional_end_date_target = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')

data_dirs = ['data', 'data_italy']
latitudes = {
    # Europe
    'London': (51.5074, -0.1278), 'Paris': (48.8566, 2.3522), 'Berlin': (52.5200, 13.4050),
    'Madrid': (40.4168, -3.7038), 'Barcelona': (41.3851, 2.1734), 'Rome': (41.9028, 12.4964), 'Amsterdam': (52.3676, 4.9041),
    'Brussels': (50.8503, 4.3517), 'Vienna': (48.2082, 16.3738), 'Prague': (50.0755, 14.4378),
    'Warsaw': (52.2297, 21.0122), 'Budapest': (47.4979, 19.0402), 'Stockholm': (59.3293, 18.0686),
    'Oslo': (59.9127, 10.7461), 'Copenhagen': (55.6761, 12.5683), 'Helsinki': (60.1695, 24.9354),
    'Dublin': (53.3498, -6.2603), 'Athens': (37.9838, 23.7275), 'Lisbon': (38.7223, -9.1393),
    'Reykjavik': (64.1466, -21.9426), 'Moscow': (55.7558, 37.6173), 'Kyiv': (50.4501, 30.5234),
    'Bucharest': (44.4268, 26.1025), 'Sofia': (42.6977, 23.3219), 'Belgrade': (44.8125, 20.4612),
    'Zagreb': (45.8150, 15.9819), 'Sarajevo': (43.8563, 18.4131), 'Skopje': (42.0000, 21.4333),
    'Tirana': (41.3275, 19.8189), 'Podgorica': (42.4411, 19.2636), 'Pristina': (42.6629, 21.1655),
    'Bratislava': (48.1486, 17.1077), 'Ljubljana': (46.0569, 14.5058), 'Tallinn': (59.4370, 24.7536),
    'Riga': (56.9496, 24.1052), 'Vilnius': (54.6872, 25.2797), 'Chisinau': (47.0105, 28.8638),
    'Minsk': (53.9006, 27.5590), 'Bern': (46.9480, 7.4474),
    # Italy
    'Aosta': (45.7373, 7.3201), 'Torino': (45.0703, 7.6869), 'Genova': (44.4056, 8.9463),
    'Milano': (45.4642, 9.1900), 'Trento': (46.0697, 11.1211), 'Venezia': (45.4408, 12.3155),
    'Trieste': (45.6495, 13.7768), 'Bologna': (44.4949, 11.3426), 'Firenze': (43.7696, 11.2558),
    'Perugia': (43.1107, 12.3908), 'Ancona': (43.6158, 13.5189), 'LAquila': (42.3498, 13.3995),
    'Campobasso': (41.5603, 14.6627), 'Napoli': (40.8518, 14.2681), 'Bari': (41.1171, 16.8719),
    'Potenza': (40.6404, 15.8056), 'Catanzaro': (38.9098, 16.5877), 'Palermo': (38.1157, 13.3615),
    'Cagliari': (39.2238, 9.1116), 'Roma': (41.9028, 12.4964)
}

def fetch_data(url, params):
    while True:
        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            daily = response.Daily()
            
            if daily.Variables(0).ValuesAsNumpy().size == 0:
                return None
                
            offset = response.UtcOffsetSeconds()
            daily_data = {"date": pd.date_range(
                start = pd.to_datetime(daily.Time() + offset, unit = "s", utc = True),
                end = pd.to_datetime(daily.TimeEnd() + offset, unit = "s", utc = True),
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

for out_dir in data_dirs:
    if not os.path.exists(out_dir):
        continue
    for file in os.listdir(out_dir):
        if not file.endswith('.csv'):
            continue
            
        file_path = os.path.join(out_dir, file)
        city = file.replace('.csv', '')
        
        if city not in latitudes:
            print(f"WARNING: Coords not found for {city}, skipping.")
            continue
            
        coords = latitudes[city]
        
        try:
            df = pd.read_csv(file_path)
            # Strip timezone
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            
            # Step 1: Remove any provisional rows from previous runs
            df_cleaned = df[df['is_historical'] == 'T'].copy()
            
            # Step 2: Fetch missing definitive data
            last_date = df_cleaned['date'].max()
            start_date_archive = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            new_definitive_df = None
            if start_date_archive <= definitive_end_date_target:
                print(f"Updating {city} DEFINITIVE from {start_date_archive} to {definitive_end_date_target}...")
                params_archive = {
                    "latitude": coords[0],
                    "longitude": coords[1],
                    "start_date": start_date_archive,
                    "end_date": definitive_end_date_target,
                    "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
                    "timezone": "auto"
                }
                new_definitive_df = fetch_data(url_archive, params_archive)
                if new_definitive_df is not None:
                    # Strip timezone from new dates to match
                    new_definitive_df['date'] = pd.to_datetime(new_definitive_df['date']).dt.tz_localize(None)
                    new_definitive_df['is_historical'] = 'T'
                    df_cleaned = pd.concat([df_cleaned, new_definitive_df], ignore_index=True)
                time.sleep(1) # respect API limits
                
            # Step 3: Fetch provisional data up to yesterday
            start_date_forecast = (pd.to_datetime(definitive_end_date_target) + timedelta(days=1)).strftime('%Y-%m-%d')
            
            new_provisional_df = None
            if start_date_forecast <= provisional_end_date_target:
                print(f"Updating {city} PROVISIONAL from {start_date_forecast} to {provisional_end_date_target}...")
                params_forecast = {
                    "latitude": coords[0],
                    "longitude": coords[1],
                    "start_date": start_date_forecast,
                    "end_date": provisional_end_date_target,
                    "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
                    "timezone": "auto"
                }
                new_provisional_df = fetch_data(url_forecast, params_forecast)
                if new_provisional_df is not None:
                    new_provisional_df['date'] = pd.to_datetime(new_provisional_df['date']).dt.tz_localize(None)
                    new_provisional_df['is_historical'] = 'F'
                    df_cleaned = pd.concat([df_cleaned, new_provisional_df], ignore_index=True)
                time.sleep(1) # respect API limits
            
            # Format dates to string with timezone offset (so scripts don't break)
            df_cleaned['date'] = df_cleaned['date'].dt.strftime('%Y-%m-%d 00:00:00+00:00')
            
            df_cleaned = df_cleaned.drop_duplicates(subset=['date'], keep='last').sort_values('date')
            
            # Write everything back to CSV
            df_cleaned.to_csv(file_path, index=False)
            print(f"{city} fully updated.")
            
        except Exception as e:
            print(f"Error processing {city}: {e}")

print("Update complete!")
