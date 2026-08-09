import requests
import pandas as pd
import io

url = "https://ourworldindata.org/grapher/annual-area-burnt-by-wildfires-gwis.csv"
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
print("Status:", r.status_code)
df = pd.read_csv(io.StringIO(r.text))
print("Columns:", df.columns.tolist())
italy = df[df['Entity'] == 'Italy']
print("Italy GWIS 2002-2024:\n", italy)
