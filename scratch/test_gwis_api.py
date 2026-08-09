import requests

r = requests.get("https://gwis.jrc.ec.europa.eu/apps/country.profile/api/countries")
print("Countries API:", r.text[:500])

# Try fetching data for Italy (ITA or IT or 106)
for iso in ["ITA", "IT", "italy"]:
    for endpoint in [
        f"https://gwis.jrc.ec.europa.eu/apps/country.profile/api/countries/{iso}/stats",
        f"https://gwis.jrc.ec.europa.eu/apps/country.profile/api/countries/{iso}/burnedarea",
        f"https://gwis.jrc.ec.europa.eu/apps/country.profile/api/burnedarea/{iso}",
        f"https://gwis.jrc.ec.europa.eu/apps/country.profile/api/stats/{iso}",
        f"https://gwis.jrc.ec.europa.eu/apps/country.profile/api/monthly/{iso}",
        f"https://gwis.jrc.ec.europa.eu/apps/country.profile/api/modis/{iso}"
    ]:
        res = requests.get(endpoint)
        if res.status_code == 200:
            print(f"FOUND 200: {endpoint} -> {res.text[:200]}")
