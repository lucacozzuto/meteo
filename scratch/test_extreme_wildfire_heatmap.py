import requests
import io
import pandas as pd
import numpy as np

# EFFIS official historical stats (1980-2001) for key Southern European nations (hectares)
HIST_SOUTHERN = {
    'Italy': {
        1980: 144200, 1981: 265000, 1982: 130000, 1983: 212600, 1984: 75000,
        1985: 190600, 1986: 119000, 1987: 120600, 1988: 186400, 1989: 95000,
        1990: 195300, 1991: 99800,  1992: 105400, 1993: 203700, 1994: 116400,
        1995: 47800,  1996: 58200,  1997: 111400, 1998: 155500, 1999: 71100,
        2000: 114600, 2001: 76400
    },
    'Spain': {
        1980: 256000, 1981: 289000, 1982: 152000, 1983: 125000, 1984: 165000,
        1985: 484000, 1986: 262000, 1987: 145000, 1988: 138000, 1989: 426000,
        1990: 203000, 1991: 260000, 1992: 105000, 1993: 89000,  1994: 437000,
        1995: 143000, 1996: 60000,  1997: 98000,  1998: 133000, 1999: 82000,
        2000: 188000, 2001: 93000
    },
    'Portugal': {
        1980: 44000,  1981: 89000,  1982: 39000,  1983: 47000,  1984: 53000,
        1985: 146000, 1986: 98000,  1987: 76000,  1988: 22000,  1989: 126000,
        1990: 137000, 1991: 182000, 1992: 57000,  1993: 49000,  1994: 77000,
        1995: 169000, 1996: 88000,  1997: 30000,  1998: 158000, 1999: 70000,
        2000: 159000, 2001: 111000
    },
    'Greece': {
        1980: 33000,  1981: 81000,  1982: 24000,  1983: 19000,  1984: 31000,
        1985: 105000, 1986: 24000,  1987: 46000,  1988: 110000, 1989: 41000,
        1990: 39000,  1991: 23000,  1992: 71000,  1993: 54000,  1994: 58000,
        1995: 27000,  1996: 25000,  1997: 52000,  1998: 93000,  1999: 83000,
        2000: 145000, 2001: 28000
    },
    'France': {
        1980: 22000,  1981: 27000,  1982: 54000,  1983: 53000,  1984: 27000,
        1985: 49000,  1986: 52000,  1987: 20000,  1988: 13000,  1989: 75000,
        1990: 76000,  1991: 10000,  1992: 17000,  1993: 26000,  1994: 25000,
        1995: 18000,  1996: 11000,  1997: 22000,  1998: 18000,  1999: 16000,
        2000: 24000,  2001: 21000
    }
}

# Fetch GWIS / OWID datasets (2002-2026)
url_gwis = "https://ourworldindata.org/grapher/annual-area-burnt-by-wildfires-gwis.csv"
url_weekly = "https://ourworldindata.org/grapher/weekly-area-burnt-by-wildfires.csv"

r_g = requests.get(url_gwis, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
df_g = pd.read_csv(io.StringIO(r_g.text))

r_w = requests.get(url_weekly, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
df_w = pd.read_csv(io.StringIO(r_w.text))
df_w['Day'] = pd.to_datetime(df_w['Day'])
df_w['year'] = df_w['Day'].dt.year

# Combine into a single matrix (1980 - 2026)
countries_lat = {
    'Finland': 61.9, 'Norway': 60.4, 'Sweden': 60.1, 'Estonia': 58.6,
    'Latvia': 56.9, 'Lithuania': 55.2, 'Denmark': 56.3, 'Ireland': 53.4,
    'United Kingdom': 55.3, 'Belarus': 53.7, 'Netherlands': 52.1,
    'Poland': 51.9, 'Germany': 51.2, 'Belgium': 50.5, 'Czechia': 49.8,
    'Slovakia': 48.7, 'Ukraine': 48.3, 'Austria': 47.5, 'Hungary': 47.2,
    'Switzerland': 46.8, 'Slovenia': 46.1, 'Croatia': 45.1, 'Romania': 45.9,
    'Bosnia and Herzegovina': 43.9, 'Serbia': 44.0, 'Bulgaria': 42.7,
    'Montenegro': 42.7, 'North Macedonia': 41.6, 'Albania': 41.1,
    'Italy': 41.9, 'Spain': 40.4, 'Portugal': 39.4, 'Greece': 39.0, 'Cyprus': 35.1
}

years = list(range(1980, 2027))
matrix = pd.DataFrame(index=list(countries_lat.keys()), columns=years, dtype=float).fillna(0.0)

for country in countries_lat.keys():
    # 1980-2001 from historical dictionary if present
    if country in HIST_SOUTHERN:
        for y, val in HIST_SOUTHERN[country].items():
            matrix.loc[country, y] = float(val)
            
    # 2002-2024 from GWIS
    df_c_g = df_g[df_g['Entity'] == country]
    for _, row in df_c_g.iterrows():
        y = int(row['Year'])
        val = float(row['Yearly burned area across all land types'])
        if 2002 <= y <= 2024:
            # Prefer GWIS for 2002-2024
            matrix.loc[country, y] = val

    # 2025-2026 from weekly dataset total
    df_c_w = df_w[df_w['Entity'] == country]
    for y in [2025, 2026]:
        val = df_c_w[df_c_w['year'] == y]['Area burnt by wildfires'].sum()
        if val > 0:
            matrix.loc[country, y] = float(val)

# Calculate 90th percentile threshold per country (row-wise)
p90 = matrix.quantile(0.90, axis=1)

# Create binary matrix: 1 if > 90th percentile, else 0
is_extreme = pd.DataFrame(0, index=matrix.index, columns=matrix.columns)
for c in matrix.index:
    thresh = p90[c]
    if thresh > 0:
        is_extreme.loc[c] = (matrix.loc[c] >= thresh).astype(int)

print("90th Percentile Burnt Area Thresholds (ha):")
print(p90)

print("\nNumber of extreme fire countries per year (1980-2026):")
ext_counts = is_extreme.sum(axis=0)
print(ext_counts)
print("\nTop 10 years with most countries exceeding 90th percentile:")
print(ext_counts.sort_values(ascending=False).head(10))
