import requests
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load GWIS / OWID weekly and annual burnt area data
url_weekly = "https://ourworldindata.org/grapher/weekly-area-burnt-by-wildfires.csv"
url_gwis = "https://ourworldindata.org/grapher/annual-area-burnt-by-wildfires-gwis.csv"

print("Downloading OWID data...")
r_w = requests.get(url_weekly, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
df_w = pd.read_csv(io.StringIO(r_w.text))
df_w['Day'] = pd.to_datetime(df_w['Day'])
df_w['year'] = df_w['Day'].dt.year

# Calculate summer (JJA) burnt area (2012-2026)
summer_df = df_w[df_w['Day'].dt.month.isin([6, 7, 8])].copy()
summer_yearly = summer_df.groupby(['Entity', 'year'])['Area burnt by wildfires'].sum().reset_index()

# Also download GWIS annual (2002-2024)
r_g = requests.get(url_gwis, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
df_g = pd.read_csv(io.StringIO(r_g.text))

print("Weekly sample entities:", summer_yearly['Entity'].unique()[:30])

# European countries sorted by Latitude (North to South)
countries_lat = {
    'Finland': 61.9,
    'Norway': 60.4,
    'Sweden': 60.1,
    'Estonia': 58.6,
    'Latvia': 56.9,
    'Lithuania': 55.2,
    'Denmark': 56.3,
    'Ireland': 53.4,
    'United Kingdom': 55.3,
    'Netherlands': 52.1,
    'Poland': 51.9,
    'Germany': 51.2,
    'Belgium': 50.5,
    'Czechia': 49.8,
    'Slovakia': 48.7,
    'Ukraine': 48.3,
    'Austria': 47.5,
    'Hungary': 47.2,
    'Switzerland': 46.8,
    'Slovenia': 46.1,
    'Croatia': 45.1,
    'Romania': 45.9,
    'Bosnia and Herzegovina': 43.9,
    'Serbia': 44.0,
    'Bulgaria': 42.7,
    'Montenegro': 42.7,
    'North Macedonia': 41.6,
    'Albania': 41.1,
    'Italy': 41.9,
    'Spain': 40.4,
    'Portugal': 39.4,
    'Greece': 39.0,
    'Cyprus': 35.1
}

# Filter for European countries in our list
df_eu = summer_yearly[summer_yearly['Entity'].isin(countries_lat.keys())].copy()
piv = df_eu.pivot(index='Entity', columns='year', values='Area burnt by wildfires').fillna(0)

# Sort by Latitude
piv['lat'] = piv.index.map(countries_lat)
piv = piv.sort_values('lat', ascending=False)
piv.index = [f"{c} ({lat:.1f}°N)" for c, lat in zip(piv.index, piv['lat'])]
piv = piv.drop(columns=['lat'])

print("\nPivot table 2012-2026 shape:", piv.shape)
print(piv.head(10))
