import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime

with open('docs/monthly_records.json', 'r') as f:
    data = json.load(f)

for city_name, dataset in [('Roma', 'Italy'), ('Madrid', 'Europe')]:
    city = data[dataset].get(city_name)
    if not city or not city.get('daily_2026'):
        continue
    
    daily = city['daily_2026']
    dates_2026 = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in daily['dates']]
    temps_2026 = daily['temps']
    normals_2026 = daily['normals']
    dates_365 = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in daily['dates_365']]
    normals_365 = daily['normals_365']

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    # Plot normal 365
    ax.plot(dates_365, normals_365, color='#16a34a', linewidth=2.5, label='Temperatura normale (media 1991-2020)', zorder=3)
    
    # Plot actual 2026
    ax.plot(dates_2026, temps_2026, color='#1e293b', linewidth=1.2, label='Evoluzione 2026', zorder=4)
    
    # Fill positive anomaly
    ax.fill_between(dates_2026, temps_2026, normals_2026, where=[(t is not None and n is not None and t > n) for t, n in zip(temps_2026, normals_2026)],
                    interpolate=True, color='#ef4444', alpha=0.85, label='Anomalia positiva (sopra media)', zorder=2)
                    
    # Fill negative anomaly
    ax.fill_between(dates_2026, temps_2026, normals_2026, where=[(t is not None and n is not None and t < n) for t, n in zip(temps_2026, normals_2026)],
                    interpolate=True, color='#3b82f6', alpha=0.85, label='Anomalia negativa (sotto media)', zorder=2)

    ax.set_title(f"Evoluzione Giornaliera Temperatura Media (2026) - {city_name}", fontsize=16, fontweight='bold', pad=15, color='#1e293b')
    ax.set_ylabel("Temperatura Media (°C)", fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.set_xlim([datetime.datetime(2026, 1, 1), datetime.datetime(2026, 12, 31)])
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
    
    # Legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=10)
    
    plt.tight_layout()
    out_path = f"/Users/lcozzuto/.gemini/antigravity/brain/e9a954cd-73d4-4826-91e2-4e6263f6d002/aemet_2026_{city_name.lower()}.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated preview: {out_path}")
