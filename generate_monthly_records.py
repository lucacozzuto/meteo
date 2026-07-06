import pandas as pd
import os
import json

def get_monthly_records(data_dir):
    records = {}
    if not os.path.exists(data_dir):
        return records
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            city = file.replace('.csv', '')
            filepath = os.path.join(data_dir, file)
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.month

            monthly_max = df.groupby('month')['temperature_2m_max'].max().round(1).tolist()
            records[city] = monthly_max
    return dict(sorted(records.items())) # Sort alphabetically

def main():
    europe_records = get_monthly_records('data')
    italy_records = get_monthly_records('data_italy')

    all_records = {
        'Europe': europe_records,
        'Italy': italy_records
    }

    os.makedirs('docs', exist_ok=True)
    with open('docs/monthly_records.json', 'w') as f:
        json.dump(all_records, f)

    print("docs/monthly_records.json generated successfully.")

if __name__ == "__main__":
    main()
