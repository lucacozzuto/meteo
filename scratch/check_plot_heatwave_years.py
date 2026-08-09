import glob
import pandas as pd
import numpy as np

def get_plot_sync_years(city_files, threshold_pct=0.25):
    all_events = []
    date_counts = {}
    
    for cfile in city_files:
        df = pd.read_csv(cfile)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df[df['date'].dt.year >= 1980].copy()
        
        # Restrict to June, July, August
        df = df[df['date'].dt.month.isin([6, 7, 8])].copy()
        
        # Calculate fixed 90th percentile based on 1991-2020 period
        baseline_data = df[(df['date'].dt.year >= 1991) & (df['date'].dt.year <= 2020)]['temperature_2m_max'].dropna()
        threshold = baseline_data.quantile(0.90) if not baseline_data.empty else 30
        
        df['is_hot'] = df['temperature_2m_max'] > threshold
        
        current_hw = []
        for i, row in df.iterrows():
            if row['is_hot']:
                if len(current_hw) > 0 and (row['date'] - current_hw[-1]['date']).days > 1:
                    if len(current_hw) >= 6:
                        for r in current_hw:
                            date_counts[r['date'].date()] = date_counts.get(r['date'].date(), 0) + 1
                    current_hw = []
                current_hw.append(row)
            else:
                if len(current_hw) >= 6:
                    for r in current_hw:
                        date_counts[r['date'].date()] = date_counts.get(r['date'].date(), 0) + 1
                current_hw = []

    # Sync events: dates where >= 25% of cities experience a heatwave simultaneously
    sync_dates = [d for d, count in date_counts.items() if count >= len(city_files) * threshold_pct]
    sync_years = sorted(list(set([d.year for d in sync_dates])))
    return sync_years

if __name__ == '__main__':
    cities = sorted(glob.glob('data/*.csv'))
    print("Europe Heatwave Sync Years (25% threshold):", get_plot_sync_years(cities, 0.25))
    print("Europe Heatwave Sync Years (33% threshold):", get_plot_sync_years(cities, 0.33))
