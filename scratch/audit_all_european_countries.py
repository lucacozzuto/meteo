import requests
import io
import pandas as pd

# Load GWIS and OWID datasets
url_gwis = "https://ourworldindata.org/grapher/annual-area-burnt-by-wildfires-gwis.csv"
url_weekly = "https://ourworldindata.org/grapher/weekly-area-burnt-by-wildfires.csv"

r_g = requests.get(url_gwis, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
df_g = pd.read_csv(io.StringIO(r_g.text))

r_w = requests.get(url_weekly, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
df_w = pd.read_csv(io.StringIO(r_w.text))

all_entities = set(df_g['Entity'].unique()).union(set(df_w['Entity'].unique()))

# Comprehensive list of all European sovereign states & territories with coordinates
ALL_EUROPEAN_SOVEREIGN = {
    'Iceland': 64.9,
    'Finland': 61.9,
    'Norway': 60.4,
    'Sweden': 60.1,
    'Estonia': 58.6,
    'Latvia': 56.9,
    'Lithuania': 55.2,
    'Denmark': 56.3,
    'Ireland': 53.4,
    'United Kingdom': 55.3,
    'Belarus': 53.7,
    'Netherlands': 52.1,
    'Poland': 51.9,
    'Germany': 51.2,
    'Belgium': 50.5,
    'Czechia': 49.8,
    'Luxembourg': 49.8,
    'Slovakia': 48.7,
    'Ukraine': 48.3,
    'Austria': 47.5,
    'Hungary': 47.2,
    'Switzerland': 46.8,
    'France': 46.2,
    'Moldova': 47.4,
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
    'Turkey': 38.9,
    'Malta': 35.9,
    'Cyprus': 35.1
}

# Check currently included in script
import sys
sys.path.append('.')
from plot_wildfire_heatmap_europe import countries_lat

print("--- EUROPEAN COUNTRIES AUDIT ---")
print(f"Total in comprehensive list: {len(ALL_EUROPEAN_SOVEREIGN)}")
print(f"Total currently in script: {len(countries_lat)}")

missing = set(ALL_EUROPEAN_SOVEREIGN.keys()) - set(countries_lat.keys())
print("\nMissing from script:")
for m in sorted(missing):
    in_owid = m in all_entities
    print(f" - {m} (Latitude {ALL_EUROPEAN_SOVEREIGN[m]}°N) -> Present in OWID dataset: {in_owid}")
