import requests
import io
import pandas as pd

# Let's test OWID share of total land area burnt or forest area datasets
urls = [
    "https://ourworldindata.org/grapher/share-of-the-total-land-area-burnt-by-wildfires-each-year.csv",
    "https://ourworldindata.org/grapher/forest-area-km.csv",
    "https://ourworldindata.org/grapher/forest-area-as-a-share-of-land-area.csv"
]

for u in urls:
    r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    print(f"URL: {u.split('/')[-1]} -> Status: {r.status_code}")
    if r.status_code == 200:
        df = pd.read_csv(io.StringIO(r.text))
        print("  Columns:", df.columns.tolist())
        print("  Sample Italy:\n", df[df['Entity'] == 'Italy'].tail(3))
