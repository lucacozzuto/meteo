import pandas as pd
import numpy as np

# Let's test two FWI approaches on Madrid data:
# 1) Angstrom Fire Weather Index (B-index): B = R/20 + (27 - T)/10
#    Where R is relative humidity (%) and T is temperature (C).
#    If B > 2.5: Low fire danger
#    2.0 < B <= 2.5: Moderate
#    B <= 2.0: High / Extreme fire danger (often we invert it so higher = worse danger, e.g. Danger Score = 4 - B or 5 - B)

# 2) FAO-56 RH estimation from Tmin and Tmax:
#    e0(T) = 0.6108 * exp(17.27 * T / (T + 237.3))
#    RH_min = 100 * e0(Tmin) / e0(Tmax)

df = pd.read_csv("/Users/lcozzuto/git/meteo/data/Madrid.csv")
df['date'] = pd.to_datetime(df['date'])

# Calculate saturation vapor pressure
e0_min = 0.6108 * np.exp(17.27 * df['temperature_2m_min'] / (df['temperature_2m_min'] + 237.3))
e0_max = 0.6108 * np.exp(17.27 * df['temperature_2m_max'] / (df['temperature_2m_max'] + 237.3))
df['rh_min_est'] = 100 * (e0_min / e0_max)

# Angstrom index at peak heat (using Tmax and rh_min_est)
df['angstrom_B'] = df['rh_min_est'] / 20.0 + (27.0 - df['temperature_2m_max']) / 10.0

# Inverted Angstrom Danger Index (so higher value = higher fire danger)
# When T=40C and RH=15%, B = 15/20 + (27-40)/10 = 0.75 - 1.3 = -0.55 (Extreme!)
# When T=20C and RH=70%, B = 70/20 + (27-20)/10 = 3.5 + 0.7 = 4.20 (No danger)
df['fire_danger_score'] = 4.0 - df['angstrom_B']  # Higher is worse!

print("Madrid sample summer days (2023):")
summer2023 = df[(df['date'].dt.year == 2023) & (df['date'].dt.month.isin([6,7,8]))]
print(summer2023[['date', 'temperature_2m_max', 'rh_min_est', 'angstrom_B', 'fire_danger_score']].tail(10))

print("\nTop 10 most extreme fire danger days in Madrid history (1940-2026):")
print(df.sort_values('fire_danger_score', ascending=False)[['date', 'temperature_2m_max', 'rh_min_est', 'angstrom_B', 'fire_danger_score']].head(10))
