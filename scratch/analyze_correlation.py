import pandas as pd
import numpy as np
import json

with open("/Users/lcozzuto/git/meteo/docs/wildfire_test_italy.json") as f:
    data = json.load(f)

years = []
burnt = []
hot_roma = []
ext_fwi_roma = []

for y, v in data['years'].items():
    b = v['burnt_italy'].get('summer_ha')
    h = v['metrics_roma'].get('hot_days_90th')
    f = v['metrics_roma'].get('extreme_fire_days')
    if b is not None and h is not None:
        years.append(int(y))
        burnt.append(b)
        hot_roma.append(h)
        ext_fwi_roma.append(f)

df = pd.DataFrame({'year': years, 'burnt': burnt, 'hot_roma': hot_roma, 'ext_fwi_roma': ext_fwi_roma})

print("Pearson correlation (1980-2026):")
print(df[['burnt', 'hot_roma', 'ext_fwi_roma']].corr())

print("\nPearson correlation (2012-2026 satellite era):")
df_recent = df[df['year'] >= 2012]
print(df_recent[['burnt', 'hot_roma', 'ext_fwi_roma']].corr())

# Let's also check Palermo / Cagliari / Reggio Calabria if available!
print("\nChecking if Palermo/Cagliari exist in data_italy:")
import glob
print(glob.glob("/Users/lcozzuto/git/meteo/data_italy/*"))
