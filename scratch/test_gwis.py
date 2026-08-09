import requests

# Test GWIS country profile downloads
urls = [
    "https://gwis.jrc.ec.europa.eu/apps/country.profile/downloads",
    "https://gwis.jrc.ec.europa.eu/apps/country.profile/api/countries",
    "https://ourworldindata.org/grapher/annual-area-burnt-by-wildfires.csv",
    "https://ourworldindata.org/grapher/cumulative-area-burnt-by-wildfires-weekly.csv"
]

for url in urls:
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        print(f"URL: {url} -> Status: {r.status_code}, Length: {len(r.content)}")
        if r.status_code == 200 and url.endswith(".csv"):
            print("First 5 lines:")
            print("\n".join(r.text.splitlines()[:5]))
    except Exception as e:
        print(f"URL: {url} -> Error: {e}")
