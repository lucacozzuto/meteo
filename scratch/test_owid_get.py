import requests

urls = [
    "https://ourworldindata.org/grapher/area-burnt-by-wildfires-by-week.csv",
    "https://ourworldindata.org/grapher/cumulative-area-burnt-by-wildfires.csv",
    "https://ourworldindata.org/grapher/weekly-wildfire-burn-area.csv",
    "https://ourworldindata.org/grapher/monthly-area-burnt-by-wildfires.csv",
    "https://ourworldindata.org/grapher/area-burnt-by-wildfires-monthly.csv",
    "https://ourworldindata.org/grapher/wildfire-burned-area-by-week.csv",
    "https://ourworldindata.org/grapher/area-burned-by-wildfire-weekly.csv",
    "https://ourworldindata.org/grapher/cumulative-area-burnt-by-wildfires-in-2023.csv",
    "https://ourworldindata.org/grapher/annual-area-burnt-by-wildfires-by-country.csv"
]

for u in urls:
    r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
    print(f"URL: {u.split('/')[-1]} -> Status: {r.status_code}")
    if r.status_code == 200 and "Entity" in r.text:
        print("   FIRST LINE:", r.text.splitlines()[0])
        print("   SECOND LINE:", r.text.splitlines()[1])
