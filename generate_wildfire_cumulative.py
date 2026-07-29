import requests
import io
import os
import json
import pandas as pd

def generate_wildfire_cumulative():
    url = "https://ourworldindata.org/grapher/weekly-area-burnt-by-wildfires.csv"
    print("Downloading weekly wildfire data from OWID...")
    res = requests.get(url)
    df = pd.read_csv(io.StringIO(res.text))

    # Rename column for convenience
    df.rename(columns={'Area burnt by wildfires': 'area_ha'}, inplace=True)
    df['Day'] = pd.to_datetime(df['Day'])
    df['year'] = df['Day'].dt.year
    df['day_of_year'] = df['Day'].dt.dayofyear

    # Italian month names
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

    # Filter for the 5 target countries
    df_med = df[df['Entity'].isin(countries.keys())].copy()

    # Calculate per country
    for en_name, it_name in countries.items():
        df_c = df_med[df_med['Entity'] == en_name].sort_values('Day')
        years_dict = {}

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

        output_data[it_name] = years_dict

    # Calculate TOTALE (Sum of the 5 countries for each week)
    df_tot = df_med.groupby(['year', 'Day'])['area_ha'].sum().reset_index().sort_values('Day')
    df_tot['day_of_year'] = df_tot['Day'].dt.dayofyear

    tot_years_dict = {}
    for y in range(2012, 2027):
        df_y = df_tot[df_tot['year'] == y].sort_values('Day')
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

        tot_years_dict[str(y)] = points

    output_data['TOTALE'] = tot_years_dict

    os.makedirs('docs', exist_ok=True)
    out_file = 'docs/wildfire_cumulative.json'
    with open(out_file, 'w') as f:
        json.dump(output_data, f)

    print(f"Successfully generated {out_file}.")

if __name__ == '__main__':
    generate_wildfire_cumulative()
