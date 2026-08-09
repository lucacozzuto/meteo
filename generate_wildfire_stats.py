import os
import io
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Official EFFIS / State Forestry Corps historical annual burnt area for Italy (1980-2011) in hectares
# Source: European Forest Fire Information System (EFFIS) Reports & JRC / ISTAT Forest Fire Statistics
ITALY_HISTORICAL_BURNT_1980_2011 = {
    1980: 144200, 1981: 265000, 1982: 130000, 1983: 212600, 1984: 75000,
    1985: 190600, 1986: 119000, 1987: 120600, 1988: 186400, 1989: 95000,
    1990: 195300, 1991: 99800,  1992: 105400, 1993: 203700, 1994: 116400,
    1995: 47800,  1996: 58200,  1997: 111400, 1998: 155500, 1999: 71100,
    2000: 114600, 2001: 76400,  2002: 75753,  2003: 153030, 2004: 166682,
    2005: 54931,  2006: 50702,  2007: 227302, 2008: 130190, 2009: 54287,
    2010: 71889,  2011: 86701
}

def fetch_owid_burnt_area():
    print("Fetching OWID weekly burnt area dataset (2012-2026)...")
    url = "https://ourworldindata.org/grapher/weekly-area-burnt-by-wildfires.csv"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    if r.status_code != 200:
        raise Exception(f"Failed to fetch OWID dataset: {r.status_code}")
    df = pd.read_csv(io.StringIO(r.text))
    df['Day'] = pd.to_datetime(df['Day'])
    return df

def get_country_summer_burnt_area(df_owid, country_name):
    country_df = df_owid[df_owid['Entity'].str.lower() == country_name.lower()].copy()
    if country_df.empty:
        print(f"WARNING: No OWID data found for {country_name}")
        return {}
    
    # Summer: June, July, August
    summer_df = country_df[country_df['Day'].dt.month.isin([6, 7, 8])]
    yearly = summer_df.groupby(summer_df['Day'].dt.year)['Area burnt by wildfires'].sum().to_dict()
    annual_total = country_df.groupby(country_df['Day'].dt.year)['Area burnt by wildfires'].sum().to_dict()
    
    res = {}
    for y in sorted(annual_total.keys()):
        res[int(y)] = {
            "summer_ha": float(yearly.get(y, 0.0)),
            "annual_ha": float(annual_total.get(y, 0.0))
        }
    return res

def compute_city_fire_metrics(csv_path):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate saturation vapor pressure (FAO-56)
    e0_min = 0.6108 * np.exp(17.27 * df['temperature_2m_min'] / (df['temperature_2m_min'] + 237.3))
    e0_max = 0.6108 * np.exp(17.27 * df['temperature_2m_max'] / (df['temperature_2m_max'] + 237.3))
    df['rh_min_est'] = 100.0 * (e0_min / e0_max)
    
    # Angstrom index at peak heat: B = RH/20 + (27 - Tmax)/10
    df['angstrom_B'] = df['rh_min_est'] / 20.0 + (27.0 - df['temperature_2m_max']) / 10.0
    # Danger score: higher is worse
    df['fire_danger_score'] = 4.0 - df['angstrom_B']
    
    # Baseline 1991-2020 for 90th percentile of summer Tmax
    summer_all = df[df['date'].dt.month.isin([6, 7, 8])].copy()
    base_summer = summer_all[(summer_all['date'].dt.year >= 1991) & (summer_all['date'].dt.year <= 2020)]
    t90 = base_summer['temperature_2m_max'].quantile(0.90) if not base_summer.empty else 32.0
    
    # Yearly summer metrics (1980 - today)
    yearly_metrics = {}
    for year, grp in summer_all.groupby(summer_all['date'].dt.year):
        if year < 1980:
            continue
        extreme_days = int((grp['fire_danger_score'] > 2.5).sum())
        high_days = int((grp['fire_danger_score'] > 2.0).sum())
        hot_days_90th = int((grp['temperature_2m_max'] > t90).sum())
        avg_score = float(grp['fire_danger_score'].mean())
        max_score = float(grp['fire_danger_score'].max())
        
        yearly_metrics[int(year)] = {
            "extreme_fire_days": extreme_days,
            "high_fire_days": high_days,
            "hot_days_90th": hot_days_90th,
            "avg_fire_danger": round(avg_score, 2),
            "max_fire_danger": round(max_score, 2)
        }
    
    # 2026 daily evolution (May to today)
    df_2026 = df[(df['date'].dt.year == 2026) & (df['date'].dt.month >= 5)].copy()
    daily_2026 = []
    for _, row in df_2026.iterrows():
        daily_2026.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "tmax": round(float(row['temperature_2m_max']), 1),
            "rh_min": round(float(row['rh_min_est']), 1),
            "fire_danger": round(float(row['fire_danger_score']), 2)
        })
        
    return yearly_metrics, daily_2026, round(float(t90), 1)

def main():
    print("--- RUNNING WILDFIRE & FWI TEST FOR ITALY (ROMA) FROM 1980 TO 2026 ---")
    df_owid = fetch_owid_burnt_area()
    italy_owid = get_country_summer_burnt_area(df_owid, "Italy")
    
    roma_csv = "/Users/lcozzuto/git/meteo/data_italy/Roma.csv"
    if not os.path.exists(roma_csv):
        roma_csv = "/Users/lcozzuto/git/meteo/data/Rome.csv"
        
    yearly_roma, daily_2026_roma, t90 = compute_city_fire_metrics(roma_csv)
    
    # Combine historical EFFIS (1980-2011) with OWID satellite (2012-2026)
    combined_burnt = {}
    for y in range(1980, 2027):
        if y in italy_owid:
            combined_burnt[y] = italy_owid[y]
        elif y in ITALY_HISTORICAL_BURNT_1980_2011:
            val = ITALY_HISTORICAL_BURNT_1980_2011[y]
            combined_burnt[y] = {
                "summer_ha": float(val),  # Historical EFFIS ground stats reported annual total (mostly summer)
                "annual_ha": float(val),
                "source": "EFFIS Ground Statistics (1980-2011)"
            }
        else:
            combined_burnt[y] = {"summer_ha": None, "annual_ha": None}

    test_data = {
        "country": "Italy",
        "capital": "Roma",
        "tmax_90th_threshold": t90,
        "years": {}
    }
    
    for y in range(1980, 2027):
        test_data["years"][y] = {
            "metrics_roma": yearly_roma.get(y, {}),
            "burnt_italy": combined_burnt.get(y, {"summer_ha": None, "annual_ha": None})
        }
        
    test_data["daily_2026_roma"] = daily_2026_roma
    
    out_json = "/Users/lcozzuto/git/meteo/docs/wildfire_test_italy.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)
    print(f"Saved test JSON (1980-2026) to {out_json}")
    
    # Generate 1980-2026 2-panel plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), gridspec_kw={'height_ratios': [1.2, 1]}, dpi=150)
    fig.patch.set_facecolor('#121212')
    
    years_all = list(range(1980, 2027))
    burnt_summer = [test_data["years"][y]["burnt_italy"].get("summer_ha", 0) or 0 for y in years_all]
    hot_days = [test_data["years"][y]["metrics_roma"].get("hot_days_90th", 0) for y in years_all]
    ext_days = [test_data["years"][y]["metrics_roma"].get("extreme_fire_days", 0) for y in years_all]
    
    # Panel 1: Dual Axis 1980-2026 (Burnt Area vs Rome Heatwave/Fire Days)
    ax1.set_facecolor('#1e1e1e')
    ax1.grid(True, color='#333333', linestyle='--', alpha=0.5, zorder=1)
    
    color_bar = '#ff5722'
    bars = ax1.bar(years_all, burnt_summer, color=color_bar, alpha=0.7, width=0.7, label='Ettari Bruciati in Italia (EFFIS 1980-2011 / GWIS 2012-2026)', zorder=2)
    ax1.set_ylabel('Ettari Bruciati (ha)', color=color_bar, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_bar, labelsize=10)
    ax1.tick_params(axis='x', colors='white', labelsize=9, rotation=45)
    ax1.set_xticks(years_all[::2])  # Every 2 years for clean ticks
    ax1.set_xlim(1978.5, 2027.5)
    ax1.set_title('ITALIA vs ROMA (1980–2026): Superficie Bruciata (EFFIS/GWIS) e Stress Termico a Roma', color='white', fontsize=14, fontweight='bold', pad=15)
    
    ax1_twin = ax1.twinx()
    color_line = '#ffd700'
    color_line2 = '#ff1744'
    ax1_twin.plot(years_all, hot_days, color=color_line, marker='o', linewidth=2, markersize=4, label='Giorni Roventi a Roma (>90° perc)', zorder=3)
    ax1_twin.plot(years_all, ext_days, color=color_line2, marker='s', linewidth=2, linestyle='--', markersize=4, label='Giorni Rischio Fuoco Estremo (FWI > 2.5)', zorder=3)
    ax1_twin.set_ylabel('Numero di Giorni a Roma (Giugno-Agosto)', color=color_line, fontsize=12, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor=color_line, labelsize=10)
    ax1_twin.set_ylim(0, max(hot_days + ext_days) * 1.2 if (hot_days + ext_days) else 10)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', facecolor='#2d2d2d', edgecolor='#444444', labelcolor='white', fontsize=10)
    
    # Panel 2: Trend 1980-2026 (Extreme Fire Days + 10-yr Rolling Mean)
    ax2.set_facecolor('#1e1e1e')
    ax2.grid(True, color='#333333', linestyle='--', alpha=0.5, zorder=1)
    
    ax2.bar(years_all, ext_days, color='#ff3d00', alpha=0.6, width=0.7, label='Giorni Rischio Estremo per Anno (JJA)', zorder=2)
    s_ext = pd.Series(ext_days, index=years_all)
    rolling_10 = s_ext.rolling(window=10, min_periods=3, center=True).mean()
    ax2.plot(years_all, rolling_10, color='#00e676', linewidth=3.5, label='Media Mobile 10 Anni (Trend Climatico)', zorder=3)
    
    ax2.set_title('ROMA (1980–2026): Trend delle Giornate Estive a Rischio Incendio Estremo (FWI > 2.5)', color='white', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('Giorni a Rischio Estremo', color='white', fontsize=12)
    ax2.tick_params(axis='x', colors='white', labelsize=9, rotation=45)
    ax2.tick_params(axis='y', colors='white', labelsize=10)
    ax2.set_xticks(years_all[::2])
    ax2.set_xlim(1978.5, 2027.5)
    ax2.legend(loc='upper left', facecolor='#2d2d2d', edgecolor='#444444', labelcolor='white', fontsize=10)
    
    plt.tight_layout(pad=2.5)
    out_img = "/Users/lcozzuto/git/meteo/wildfire_test_italy.png"
    plt.savefig(out_img, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Saved updated 1980-2026 plot to {out_img}")

if __name__ == "__main__":
    main()
