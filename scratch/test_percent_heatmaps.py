import requests
import io
import pandas as pd
import numpy as np

# Load forest area dataset (in hectares)
url_forest = "https://ourworldindata.org/grapher/forest-area-km.csv"
r_f = requests.get(url_forest, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
df_forest = pd.read_csv(io.StringIO(r_f.text))

# Build country forest area mapping (latest 2025 or mean)
forest_map = {}
for country, grp in df_forest.groupby('Entity'):
    latest_val = grp.sort_values('Year')['Forest area'].iloc[-1]
    if latest_val > 0:
        forest_map[country] = latest_val

# Fallbacks/fixes for any country names
forest_map['Czechia'] = forest_map.get('Czechia', 2677000.0)
forest_map['North Macedonia'] = forest_map.get('North Macedonia', 1001000.0)
forest_map['Bosnia and Herzegovina'] = forest_map.get('Bosnia and Herzegovina', 2185000.0)

print("Sample Forest Area Map (ha):")
for k in ['Spain', 'Portugal', 'Italy', 'Greece', 'France', 'Germany', 'Sweden']:
    print(f" {k}: {forest_map.get(k, 0):,.0f} ha")
