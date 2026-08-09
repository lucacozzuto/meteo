import requests
import io
import pandas as pd

url_forest = "https://ourworldindata.org/grapher/forest-area-km.csv"
r = requests.get(url_forest, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
df = pd.read_csv(io.StringIO(r.text))

# In OWID forest-area-km.csv, values are in hectares (e.g. 9.4M ha for Italy)
countries = ['Spain', 'Portugal', 'Italy', 'Greece', 'France', 'Germany', 'Sweden', 'Finland', 'Poland', 'United Kingdom']
latest_forest = df[(df['Entity'].isin(countries)) & (df['Year'] == 2025)]
print("Forest area in hectares (2025):")
print(latest_forest[['Entity', 'Forest area']].to_string(index=False))
