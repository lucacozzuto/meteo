import requests
import pandas as pd
import io

# Let's search OWID for any wildfire csv by querying grapher or GitHub
urls = [
    "https://ourworldindata.org/grapher/annual-area-burnt-by-wildfires.csv",
    "https://ourworldindata.org/grapher/monthly-burned-area-by-wildfires.csv",
    "https://ourworldindata.org/grapher/burned-area-from-wildfires-monthly.csv",
    "https://ourworldindata.org/grapher/wildfire-burned-area-monthly.csv",
    "https://ourworldindata.org/grapher/monthly-wildfire-burned-area.csv",
    "https://ourworldindata.org/grapher/share-of-total-land-area-burnt-by-wildfires.csv"
]

for u in urls:
    r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code == 200 and "Entity" in r.text:
        print("SUCCESS OWID:", u)
        df = pd.read_csv(io.StringIO(r.text))
        print("Columns:", df.columns.tolist())
        print("Years:", df['Year'].min(), df['Year'].max())
        italy = df[df['Entity'] == 'Italy']
        print("Italy rows:", len(italy))
        if len(italy) > 0:
            print("Italy tail:\n", italy.tail(5))
