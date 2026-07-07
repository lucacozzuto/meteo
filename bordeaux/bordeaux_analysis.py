import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

lat, lon = 44.8378, -0.5792

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

start_date_archive = "1940-01-01"
print(f"Fetching definitive data from {start_date_archive} to {definitive_end_date_target}...")
params_archive = {
    "latitude": lat,
    "longitude": lon,
    "start_date": start_date_archive,
    "end_date": definitive_end_date_target,
    "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
    "timezone": "auto"
}
df_def = fetch_data(url_archive, params_archive)
df_def['date'] = pd.to_datetime(df_def['date']).dt.tz_localize(None)

start_date_forecast = (pd.to_datetime(definitive_end_date_target) + timedelta(days=1)).strftime('%Y-%m-%d')
print(f"Fetching provisional data from {start_date_forecast} to {provisional_end_date_target}...")
params_forecast = {
    "latitude": lat,
    "longitude": lon,
    "start_date": start_date_forecast,
    "end_date": provisional_end_date_target,
    "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
    "timezone": "auto"
}
df_prov = fetch_data(url_forecast, params_forecast)
df_prov['date'] = pd.to_datetime(df_prov['date']).dt.tz_localize(None)

df = pd.concat([df_def, df_prov], ignore_index=True)
df = df.drop_duplicates(subset=['date'], keep='last').sort_values('date')
df['city'] = 'Bordeaux'
df['year'] = df['date'].dt.year
df = df[df['year'] >= 1940]

output_dir = '/Users/lcozzuto/.gemini/antigravity/brain/e9a954cd-73d4-4826-91e2-4e6263f6d002'

def plot_records_max(df, output_path):
    yearly_data = df.groupby(['city', 'year'])['temperature_2m_max'].max().reset_index()
    yearly_data = yearly_data.sort_values(by=['city', 'year'])
    yearly_data['prev_max'] = yearly_data.groupby('city')['temperature_2m_max'].transform(lambda x: x.cummax().shift(1))
    yearly_data['is_record'] = (yearly_data['temperature_2m_max'] > yearly_data['prev_max']).astype(int)
    yearly_data.loc[yearly_data['year'] < 1955, 'is_record'] = -1

    heatmap_data = yearly_data.pivot(index='city', columns='year', values='is_record').astype(int)
    heatmap_temps = yearly_data.pivot(index='city', columns='year', values='temperature_2m_max')
    annot_data = np.where(heatmap_data == 1, heatmap_temps.round(1).astype(str), "")
    
    fig, ax = plt.subplots(figsize=(24, 2))
    cmap = sns.color_palette(["lightgray", "white", "#ef4444"])
    sns.heatmap(heatmap_data, cmap=cmap, ax=ax, annot=annot_data, fmt="", annot_kws={"size": 10, "weight": "bold", "color": "black"}, cbar=False, linewidths=0.5, linecolor='gray', xticklabels=True)
    ax.set_title('Nuovi Record di Temperatura Massima Assoluta (Bordeaux)', fontsize=14)
    ax.set_xlabel('Anno')
    ax.set_ylabel('')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

def plot_records_min(df, output_path):
    yearly_data = df.groupby(['city', 'year'])['temperature_2m_min'].max().reset_index()
    yearly_data = yearly_data.sort_values(by=['city', 'year'])
    yearly_data['prev_min'] = yearly_data.groupby('city')['temperature_2m_min'].transform(lambda x: x.cummax().shift(1))
    yearly_data['is_record'] = (yearly_data['temperature_2m_min'] > yearly_data['prev_min']).astype(int)
    yearly_data.loc[yearly_data['year'] < 1955, 'is_record'] = -1

    heatmap_data = yearly_data.pivot(index='city', columns='year', values='is_record').astype(int)
    heatmap_temps = yearly_data.pivot(index='city', columns='year', values='temperature_2m_min')
    annot_data = np.where(heatmap_data == 1, heatmap_temps.round(1).astype(str), "")
    
    fig, ax = plt.subplots(figsize=(24, 2))
    cmap = sns.color_palette(["lightgray", "white", "#3b82f6"])
    sns.heatmap(heatmap_data, cmap=cmap, ax=ax, annot=annot_data, fmt="", annot_kws={"size": 10, "weight": "bold", "color": "black"}, cbar=False, linewidths=0.5, linecolor='gray', xticklabels=True)
    ax.set_title('Nuovi Record di Temperatura Minima Più Alta (Bordeaux)', fontsize=14)
    ax.set_xlabel('Anno')
    ax.set_ylabel('')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

def plot_bar(df, metric_col, threshold, title, color, output_path):
    df_filtered = df.copy()
    df_filtered['is_event'] = (df_filtered[metric_col] >= threshold).astype(int)
    yearly_data = df_filtered.groupby('year')['is_event'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(yearly_data['year'], yearly_data['is_event'], color=color)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Anno')
    ax.set_ylabel('Numero di giorni')
    plt.xticks(np.arange(1940, yearly_data['year'].max() + 1, 5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

plot_records_max(df, f'{output_dir}/bordeaux_record_max.png')
plot_records_min(df, f'{output_dir}/bordeaux_record_min.png')
plot_bar(df, 'temperature_2m_max', 30, 'Giorni Torridi (> 30°C) - Bordeaux', 'orange', f'{output_dir}/bordeaux_hot_days.png')
plot_bar(df, 'temperature_2m_min', 20, 'Notti Tropicali (>= 20°C) - Bordeaux', 'purple', f'{output_dir}/bordeaux_tropical_nights.png')
plot_bar(df, 'precipitation_sum', 50, 'Precipitazioni Estreme (>= 50mm) - Bordeaux', 'blue', f'{output_dir}/bordeaux_extreme_rain.png')

print("All Bordeaux plots generated.")
