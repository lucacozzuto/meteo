import pandas as pd
import numpy as np
import json

def get_fwi(city_path):
    df = pd.read_csv(city_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df[df['date'].dt.month.isin([6, 7, 8])].copy()
    
    e0_min = 0.6108 * np.exp(17.27 * df['temperature_2m_min'] / (df['temperature_2m_min'] + 237.3))
    e0_max = 0.6108 * np.exp(17.27 * df['temperature_2m_max'] / (df['temperature_2m_max'] + 237.3))
    df['rh_min_est'] = 100.0 * (e0_min / e0_max)
    df['angstrom_B'] = df['rh_min_est'] / 20.0 + (27.0 - df['temperature_2m_max']) / 10.0
    df['fire_danger_score'] = 4.0 - df['angstrom_B']
    
    res = {}
    for year, grp in df.groupby(df['date'].dt.year):
        res[int(year)] = {
            "ext_days": (grp['fire_danger_score'] > 2.5).sum(),
            "avg_danger": grp['fire_danger_score'].mean()
        }
    return res

roma = get_fwi("/Users/lcozzuto/git/meteo/data_italy/Roma.csv")
palermo = get_fwi("/Users/lcozzuto/git/meteo/data_italy/Palermo.csv")
cagliari = get_fwi("/Users/lcozzuto/git/meteo/data_italy/Cagliari.csv")
catanzaro = get_fwi("/Users/lcozzuto/git/meteo/data_italy/Catanzaro.csv")
bari = get_fwi("/Users/lcozzuto/git/meteo/data_italy/Bari.csv")

with open("/Users/lcozzuto/git/meteo/docs/wildfire_test_italy.json") as f:
    data = json.load(f)

years, burnt, fwi_roma, fwi_palermo, fwi_cagliari, fwi_south_avg = [], [], [], [], [], []

for y in range(1980, 2027):
    b = data['years'].get(str(y), {}).get('burnt_italy', {}).get('summer_ha')
    if b is not None:
        years.append(y)
        burnt.append(b)
        fwi_roma.append(roma.get(y, {}).get('ext_days', 0))
        fwi_palermo.append(palermo.get(y, {}).get('ext_days', 0))
        fwi_cagliari.append(cagliari.get(y, {}).get('ext_days', 0))
        s_avg = np.mean([
            palermo.get(y, {}).get('ext_days', 0),
            cagliari.get(y, {}).get('ext_days', 0),
            catanzaro.get(y, {}).get('ext_days', 0),
            bari.get(y, {}).get('ext_days', 0)
        ])
        fwi_south_avg.append(s_avg)

df = pd.DataFrame({
    'year': years,
    'burnt': burnt,
    'roma': fwi_roma,
    'palermo': fwi_palermo,
    'cagliari': fwi_cagliari,
    'south_avg': fwi_south_avg
})

print("Correlations 1980-2026:")
print(df[['burnt', 'roma', 'palermo', 'cagliari', 'south_avg']].corr()['burnt'])

print("\nCorrelations 2002-2026 (Satellite MODIS/VIIRS era):")
df_sat = df[df['year'] >= 2002]
print(df_sat[['burnt', 'roma', 'palermo', 'cagliari', 'south_avg']].corr()['burnt'])
