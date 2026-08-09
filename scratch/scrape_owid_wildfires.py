import requests
import re

url = "https://ourworldindata.org/wildfires"
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
print("Status:", r.status_code)

graphers = set(re.findall(r'grapher/([a-zA-Z0-9_-]+)', r.text))
print(f"Found {len(graphers)} graphers on OWID wildfires page:")
for g in sorted(graphers):
    if "fire" in g or "burn" in g:
        print(" ->", g)
        # Test if .csv exists
        res = requests.head(f"https://ourworldindata.org/grapher/{g}.csv", headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code == 200:
            print(f"     [CSV 200 OK] https://ourworldindata.org/grapher/{g}.csv")
