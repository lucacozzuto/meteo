import pandas as pd
import os
import json
import datetime

data_dirs = ["data", "data_italy"]
all_data = pd.DataFrame()

dfs = []
for data_dir in data_dirs:
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                city = file.replace('.csv', '')
                df = pd.read_csv(os.path.join(data_dir, file))
                df['city'] = city
                # Add location to distinguish between Europe and Italy if needed, but city name is enough
                dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
all_data['date'] = pd.to_datetime(all_data['date'])
all_data['year'] = all_data['date'].dt.year
all_data['month'] = all_data['date'].dt.month

# Filter from 1940 onwards
all_data = all_data[all_data['year'] >= 1940]

current_year = all_data['year'].max()

new_records = []

months_it = {1: 'Gennaio', 2: 'Febbraio', 3: 'Marzo', 4: 'Aprile', 5: 'Maggio', 6: 'Giugno',
             7: 'Luglio', 8: 'Agosto', 9: 'Settembre', 10: 'Ottobre', 11: 'Novembre', 12: 'Dicembre'}

for city in all_data['city'].unique():
    city_data = all_data[all_data['city'] == city]
    
    past_data = city_data[city_data['year'] < current_year]
    curr_data = city_data[city_data['year'] == current_year]
    
    if past_data.empty or curr_data.empty:
        continue
        
    # Yearly Record
    past_max_idx = past_data['temperature_2m_max'].idxmax()
    past_max_val = past_data.loc[past_max_idx, 'temperature_2m_max']
    past_max_date = past_data.loc[past_max_idx, 'date']
    
    curr_max_idx = curr_data['temperature_2m_max'].idxmax()
    curr_max_val = curr_data.loc[curr_max_idx, 'temperature_2m_max']
    curr_max_date = curr_data.loc[curr_max_idx, 'date']
    
    if curr_max_val > past_max_val:
        new_records.append({
            "city": city,
            "type": "annuale",
            "period": "Anno",
            "new_record": curr_max_val,
            "new_date": curr_max_date.strftime('%Y-%m-%d'),
            "old_record": past_max_val,
            "old_date": past_max_date.strftime('%Y-%m-%d')
        })
        
    # Monthly Records
    for month in range(1, 13):
        past_month_data = past_data[past_data['month'] == month]
        curr_month_data = curr_data[curr_data['month'] == month]
        
        if past_month_data.empty or curr_month_data.empty:
            continue
            
        p_idx = past_month_data['temperature_2m_max'].idxmax()
        p_val = past_month_data.loc[p_idx, 'temperature_2m_max']
        p_date = past_month_data.loc[p_idx, 'date']
        
        c_idx = curr_month_data['temperature_2m_max'].idxmax()
        c_val = curr_month_data.loc[c_idx, 'temperature_2m_max']
        c_date = curr_month_data.loc[c_idx, 'date']
        
        if c_val > p_val:
            new_records.append({
                "city": city,
                "type": "mensile",
                "period": months_it[month],
                "new_record": c_val,
                "new_date": c_date.strftime('%Y-%m-%d'),
                "old_record": p_val,
                "old_date": p_date.strftime('%Y-%m-%d')
            })

# Sort records by new_date descending
new_records.sort(key=lambda x: x['new_date'], reverse=True)

# Only keep records broken on the absolute last day of available data
max_date_str = all_data['date'].max().strftime('%Y-%m-%d')
new_records = [r for r in new_records if r['new_date'] == max_date_str]

os.makedirs('docs', exist_ok=True)
with open('docs/latest_records.json', 'w') as f:
    json.dump(new_records, f, indent=4)

print(f"Found {len(new_records)} new records on the last day ({max_date_str}). Saved to docs/latest_records.json")
