import requests
import io
import os
import json
import pandas as pd
import numpy as np
from plot_wildfire_heatmaps import EFFIS_GROUND

def generate_wildfire_cumulative():
    url = "https://ourworldindata.org/grapher/weekly-area-burnt-by-wildfires.csv"
    print("Downloading weekly wildfire data from OWID...")
    res = requests.get(url)
    df = pd.read_csv(io.StringIO(res.text))

    df.rename(columns={'Area burnt by wildfires': 'area_ha'}, inplace=True)
    if 'Day' in df.columns:
        df['Day'] = pd.to_datetime(df['Day'])
    elif 'Week' in df.columns:
        df['Day'] = pd.to_datetime(df['Week'] + '-1', format='%G-W%V-%u')
    elif 'Date' in df.columns:
        df['Day'] = pd.to_datetime(df['Date'])
    elif 'date' in df.columns:
        df['Day'] = pd.to_datetime(df['date'])

    df['year'] = df['Day'].dt.year
    df['day_of_year'] = df['Day'].dt.dayofyear

    italian_months = {
        1: 'Gen', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mag', 6: 'Giu',
        7: 'Lug', 8: 'Ago', 9: 'Set', 10: 'Ott', 11: 'Nov', 12: 'Dic'
    }

    countries = {
        'Italy': 'Italia',
        'Spain': 'Spagna',
        'Greece': 'Grecia',
        'Portugal': 'Portogallo',
        'France': 'Francia'
    }

    output_data = {}
    df_med = df[df['Entity'].isin(countries.keys())].copy()

    # Create standard weekly template dates using a non-leap 2025 baseline
    template_df = df_med[df_med['year'] == 2025].groupby('Day').first().reset_index()
    template_df['day_of_year'] = template_df['Day'].dt.dayofyear
    template_df['month'] = template_df['Day'].dt.month
    template_df['day'] = template_df['Day'].dt.day
    template_weeks = template_df[['day_of_year', 'month', 'day']].sort_values('day_of_year').to_dict('records')

    # Process each country
    for en_name, it_name in countries.items():
        df_c = df_med[df_med['Entity'] == en_name].sort_values('Day')
        years_dict = {}

        # 1. Years 2012-2026 from OWID weekly data
        for y in range(2012, 2027):
            df_y = df_c[df_c['year'] == y].sort_values('Day')
            if df_y.empty:
                continue

            df_y = df_y.copy()
            df_y['cumsum_ha'] = df_y['area_ha'].cumsum()

            points = []
            for _, row in df_y.iterrows():
                dt = row['Day']
                day_label = f"{dt.day:02d} {italian_months[dt.month]}"
                points.append({
                    "date": dt.strftime('%Y-%m-%d'),
                    "day_of_year": int(row['day_of_year']),
                    "day_label": day_label,
                    "weekly_ha": float(round(row['area_ha'], 1)),
                    "cumulative_ha": float(round(row['cumsum_ha'], 1))
                })
            years_dict[str(y)] = points

        # Calculate average seasonal proportion profile (0 to 1) for 1980-2011 reconstruction
        props_list = []
        for y in range(2012, 2026):
            if str(y) in years_dict and len(years_dict[str(y)]) > 0:
                y_pts = years_dict[str(y)]
                tot_y = y_pts[-1]['cumulative_ha']
                if tot_y > 0:
                    prop = [p['cumulative_ha'] / tot_y for p in y_pts]
                    props_list.append(prop)

        min_len = min([len(p) for p in props_list]) if props_list else len(template_weeks)
        avg_profile = np.mean([p[:min_len] for p in props_list], axis=0) if props_list else np.linspace(0, 1, min_len)

        # 2. Years 1980-2011 from EFFIS_GROUND totals + seasonal profile
        effis_c = EFFIS_GROUND.get(en_name, {})
        for y in range(1980, 2012):
            annual_tot = effis_c.get(y, 0)
            points = []
            for idx, w_info in enumerate(template_weeks[:min_len]):
                day_label = f"{w_info['day']:02d} {italian_months[w_info['month']]}"
                cum_ha = float(round(annual_tot * avg_profile[idx], 1))
                points.append({
                    "date": f"{y}-{w_info['month']:02d}-{w_info['day']:02d}",
                    "day_of_year": int(w_info['day_of_year']),
                    "day_label": day_label,
                    "weekly_ha": 0.0,
                    "cumulative_ha": cum_ha
                })
            years_dict[str(y)] = points

        # Sort years chronologically
        years_dict = dict(sorted(years_dict.items(), key=lambda item: int(item[0])))
        output_data[it_name] = years_dict

    # Calculate TOTALE (Sum of all 5 countries per year)
    tot_years_dict = {}
    for y in range(1980, 2027):
        y_str = str(y)
        # Check if all 5 countries have data for year y
        sample_country = list(output_data.keys())[0]
        if y_str not in output_data[sample_country]:
            continue

        n_points = len(output_data[sample_country][y_str])
        tot_points = []
        for idx in range(n_points):
            ref_pt = output_data[sample_country][y_str][idx]
            sum_cum = sum([output_data[c][y_str][idx]['cumulative_ha'] for c in output_data.keys()])
            tot_points.append({
                "date": ref_pt["date"],
                "day_of_year": ref_pt["day_of_year"],
                "day_label": ref_pt["day_label"],
                "weekly_ha": 0.0,
                "cumulative_ha": float(round(sum_cum, 1))
            })
        tot_years_dict[y_str] = tot_points

    output_data['TOTALE'] = dict(sorted(tot_years_dict.items(), key=lambda item: int(item[0])))

    os.makedirs('docs', exist_ok=True)
    out_file = 'docs/wildfire_cumulative.json'
    with open(out_file, 'w') as f:
        json.dump(output_data, f)

    print(f"Successfully generated {out_file} for 1980-2026.")

if __name__ == '__main__':
    generate_wildfire_cumulative()
