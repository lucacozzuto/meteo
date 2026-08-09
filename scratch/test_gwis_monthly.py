import requests
import re

url = "https://gwis.jrc.ec.europa.eu/apps/country.profile/downloads"
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

print("Links on page:")
links = re.findall(r'href=[\'"]?([^\'" >]+)', r.text)
for href in links:
    if any(ext in href for ext in ['.csv', '.zip', '.xls', '.xlsx', 'download', 'api', 'data', 'modis', 'mcd64a1']):
        print(" ->", href)

# Also let's check Our World in Data grapher search or API
owid_urls = [
    "https://ourworldindata.org/grapher/area-burnt-by-wildfires-by-week.csv",
    "https://ourworldindata.org/grapher/cumulative-area-burnt-by-wildfires.csv",
    "https://ourworldindata.org/grapher/weekly-wildfire-burn-area.csv",
    "https://ourworldindata.org/grapher/share-of-total-land-area-burnt-by-wildfires.csv",
    "https://ourworldindata.org/grapher/annual-share-of-total-land-area-burnt-by-wildfires.csv",
    "https://ourworldindata.org/grapher/monthly-area-burnt-by-wildfires.csv",
    "https://ourworldindata.org/grapher/area-burnt-by-wildfires-monthly.csv",
    "https://ourworldindata.org/grapher/wildfire-burned-area-by-week.csv"
]
for u in owid_urls:
    try:
        res = requests.head(u, timeout=5)
        if res.status_code == 200:
            print("OWID Found:", u)
    except:
        pass
