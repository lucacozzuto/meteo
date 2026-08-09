import requests
import io
import pandas as pd
import numpy as np

# Let's inspect data coverage per country in our matrix
HIST_SOUTHERN = {
    'Spain': list(range(1961, 2002)),
    'Italy': list(range(1961, 2002)),
    'Greece': list(range(1965, 2002)),
    'Portugal': list(range(1980, 2002)),
    'France': list(range(1973, 2002))
}

url_gwis = "https://ourworldindata.org/grapher/annual-area-burnt-by-wildfires-gwis.csv"
url_weekly = "https://ourworldindata.org/grapher/weekly-area-burnt-by-wildfires.csv"

r_g = requests.get(url_gwis, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
df_g = pd.read_csv(io.StringIO(r_g.text))

r_w = requests.get(url_weekly, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
df_w = pd.read_csv(io.StringIO(r_w.text))
df_w['Day'] = pd.to_datetime(df_w['Day'])
df_w['year'] = df_w['Day'].dt.year

countries = [
    'Finland', 'Norway', 'Sweden', 'Estonia', 'Latvia', 'Lithuania', 'Denmark',
    'Ireland', 'United Kingdom', 'Belarus', 'Netherlands', 'Poland', 'Germany',
    'Belgium', 'Czechia', 'Slovakia', 'Ukraine', 'Austria', 'Hungary', 'Switzerland',
    'Slovenia', 'Croatia', 'Romania', 'Bosnia and Herzegovina', 'Serbia', 'Bulgaria',
    'Montenegro', 'North Macedonia', 'Albania', 'Italy', 'Spain', 'Portugal', 'Greece', 'Cyprus'
]

print(f"{'Country':<25} | {'Monitoring Era':<15} | {'Coverage 2002-2026':<20} | {'Historical (pre-2002)'}")
print("-" * 85)

for c in sorted(countries):
    g_years = set(df_g[df_g['Entity'] == c]['Year'].astype(int))
    w_years = set(df_w[df_w['Entity'] == c]['year'].astype(int))
    sat_years = g_years.union(w_years)
    
    hist_years = set(HIST_SOUTHERN.get(c, []))
    all_active = sorted(list(hist_years.union(sat_years)))
    
    if all_active:
        min_y = min(all_active)
        max_y = max(all_active)
        era_str = f"{min_y} - {max_y}"
        sat_cov = f"{len([y for y in sat_years if 2002 <= y <= 2026])}/25 years"
        hist_cov = f"{len(hist_years)} years ({min_y}-2001)" if hist_years else "Satelliti dal 2002"
    else:
        era_str = "No data"
        sat_cov = "0/25"
        hist_cov = "Nessuno"
        
    print(f"{c:<25} | {era_str:<15} | {sat_cov:<20} | {hist_cov}")
