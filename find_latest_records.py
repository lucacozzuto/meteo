import pandas as pd
import os
import json

data_dirs = ["data", "data_italy"]
dfs = []

for data_dir in data_dirs:
    if os.path.exists(data_dir):
        region = "Italy" if data_dir == "data_italy" else "Europe"
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                city = file.replace('.csv', '')
                df = pd.read_csv(os.path.join(data_dir, file))
                df['city'] = city
                df['region'] = region
                dfs.append(df)

if not dfs:
    print("No data found.")
    exit()

all_data = pd.concat(dfs, ignore_index=True)
all_data['date'] = pd.to_datetime(all_data['date'])
all_data['year'] = all_data['date'].dt.year

# Filter from 1940 onwards
all_data = all_data[all_data['year'] >= 1940]

current_year = all_data['year'].max()

new_records = []

for city in all_data['city'].unique():
    city_data = all_data[all_data['city'] == city]
    region = city_data['region'].iloc[0]
    
    past_data = city_data[city_data['year'] < current_year]
    curr_data = city_data[city_data['year'] == current_year]
    
    if past_data.empty or curr_data.empty:
        continue
        
    # Yearly Record (MAX of max - Absolute Hottest Day)
    past_max_idx = past_data['temperature_2m_max'].idxmax()
    past_max_val = past_data.loc[past_max_idx, 'temperature_2m_max']
    past_max_date = past_data.loc[past_max_idx, 'date']
    
    curr_max_idx = curr_data['temperature_2m_max'].idxmax()
    curr_max_val = curr_data.loc[curr_max_idx, 'temperature_2m_max']
    curr_max_date = curr_data.loc[curr_max_idx, 'date']
    
    if curr_max_val > past_max_val:
        new_records.append({
            "city": city,
            "region": region,
            "metric": "massima",
            "new_record": curr_max_val,
            "new_date": curr_max_date.strftime('%Y-%m-%d'),
            "old_record": past_max_val,
            "old_date": past_max_date.strftime('%Y-%m-%d')
        })
        
    # Yearly Record (MAX of min - Warmest Night)
    past_min_idx = past_data['temperature_2m_min'].idxmax()
    past_min_val = past_data.loc[past_min_idx, 'temperature_2m_min']
    past_min_date = past_data.loc[past_min_idx, 'date']
    
    curr_min_idx = curr_data['temperature_2m_min'].idxmax()
    curr_min_val = curr_data.loc[curr_min_idx, 'temperature_2m_min']
    curr_min_date = curr_data.loc[curr_min_idx, 'date']
    
    if curr_min_val > past_min_val:
        new_records.append({
            "city": city,
            "region": region,
            "metric": "minima",
            "new_record": curr_min_val,
            "new_date": curr_min_date.strftime('%Y-%m-%d'),
            "old_record": past_min_val,
            "old_date": past_min_date.strftime('%Y-%m-%d')
        })

# Sort records by new_date descending
new_records.sort(key=lambda x: x['new_date'], reverse=True)

os.makedirs('docs', exist_ok=True)
with open('docs/latest_records.json', 'w') as f:
    json.dump(new_records, f, indent=4)

print(f"Found {len(new_records)} new records for {current_year}. Saved to docs/latest_records.json")
