import requests
import pandas as pd
import io

url = "https://ourworldindata.org/grapher/weekly-area-burnt-by-wildfires.csv"
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
print("Status:", r.status_code)
df = pd.read_csv(io.StringIO(r.text))
print("Columns:", df.columns.tolist())
print("Head:\n", df.head())
print("Unique entities (sample 20):", df['Entity'].unique()[:20])

# Let's see what date column or year column it has
print("Data types:", df.dtypes)
if 'Day' in df.columns or 'Date' in df.columns or 'Year' in df.columns:
    print("Sample dates:\n", df.tail(10))

# Also check annual-area-burnt-by-wildfires-gwis.csv
url_gwis = "https://ourworldindata.org/grapher/annual-area-burnt-by-wildfires-gwis.csv"
r2 = requests.get(url_gwis, headers={"User-Agent": "Mozilla/5.0"})
df2 = pd.read_csv(io.StringIO(r2.text))
print("GWIS Annual columns:", df2.columns.tolist())
print("GWIS years:", df2['Year'].min(), df2['Year'].max())
print("Italy GWIS:\n", df2[df2['Entity'] == 'Italy'].tail(10))
