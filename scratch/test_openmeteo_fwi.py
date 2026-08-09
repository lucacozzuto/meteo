import requests
import datetime

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 41.9028,
    "longitude": 12.4964,
    "start_date": "2023-07-01",
    "end_date": "2023-07-10",
    "daily": ["temperature_2m_max", "precipitation_sum", "wind_speed_10m_max", "relative_humidity_2m_min", "et0_fao_evapotranspiration"],
    "timezone": "auto"
}
resp = requests.get(url, params=params)
print("Status:", resp.status_code)
if resp.status_code == 200:
    print("Keys available:", resp.json().get("daily", {}).keys())
else:
    print("Error:", resp.text)

# Also test if there is a direct fire_weather_index or similar
for var_name in ["fire_weather_index", "vapour_pressure_deficit_max", "soil_moisture_0_to_7cm_mean"]:
    p = dict(params)
    p["daily"] = [var_name]
    r = requests.get(url, params=p)
    print(f"Test {var_name}: status={r.status_code}")
