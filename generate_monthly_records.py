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
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month

            # Filter from 1940 onwards
            df = df[df['year'] >= 1940]

            monthly_yearly_max = df.groupby(['month', 'year'])['temperature_2m_max'].max().reset_index()
            monthly_yearly_max = monthly_yearly_max.sort_values(by=['month', 'year'])
            monthly_yearly_max['prev_max'] = monthly_yearly_max.groupby('month')['temperature_2m_max'].transform(lambda x: x.cummax().shift(1))

            monthly_yearly_max['is_record'] = (monthly_yearly_max['temperature_2m_max'] > monthly_yearly_max['prev_max']).astype(int)
            monthly_yearly_max.loc[monthly_yearly_max['year'] < 1955, 'is_record'] = -1

            years = sorted(monthly_yearly_max['year'].unique().tolist())

            city_data = {
                "years": years,
                "records": [],
                "temps": []
            }

            for m in range(1, 13):
                m_data = monthly_yearly_max[monthly_yearly_max['month'] == m]
                m_data = m_data.set_index('year').reindex(years).reset_index()

                m_data['is_record'] = m_data['is_record'].fillna(0).astype(int)
                m_data['temperature_2m_max'] = m_data['temperature_2m_max'].fillna(0)

                city_data["records"].append(m_data['is_record'].tolist())
                city_data["temps"].append(m_data['temperature_2m_max'].round(1).tolist())

            records[city] = city_data

    return dict(sorted(records.items()))

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
