import glob
import pandas as pd
import numpy as np

def compute_heatwaves_years(city_files):
    all_events = []
    date_counts = {}
    
    for cfile in city_files:
        df = pd.read_csv(cfile)
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Summer months (June, July, August)
        df_summer = df[df['month'].isin([6, 7, 8])].copy()
        
        # Baseline 1991-2020 90th percentile per calendar day
        df_base = df_summer[(df_summer['year'] >= 1991) & (df_summer['year'] <= 2020)]
        p90 = df_base.groupby(df_base['date'].dt.strftime('%m-%d'))['temperature_2m_max'].transform(lambda x: np.percentile(x, 90))
        
        # Merge p90 back
        df_summer['p90'] = df_summer['date'].dt.strftime('%m-%d').map(
            df_base.groupby(df_base['date'].dt.strftime('%m-%d'))['temperature_2m_max'].apply(lambda x: np.percentile(x, 90))
        )
        
        df_summer['is_hot'] = df_summer['temperature_2m_max'] > df_summer['p90']
        
        # Find consecutive hot days >= 6
        df_summer['group'] = (~df_summer['is_hot']).cumsum()
        hot_periods = df_summer[df_summer['is_hot']].groupby('group')
        
        for _, grp in hot_periods:
            if len(grp) >= 6:
                for d in grp['date']:
                    date_counts[d.date()] = date_counts.get(d.date(), 0) + 1

    # Sync events: dates where >= 25% of cities experience a heatwave simultaneously
    sync_dates = [d for d, count in date_counts.items() if count >= len(city_files) * 0.20]
    sync_years = sorted(list(set([d.year for d in sync_dates])))
    return sync_years, date_counts

if __name__ == '__main__':
    cities = sorted(glob.glob('data/*.csv'))
    hw_years, d_counts = compute_heatwaves_years(cities)
    print("Heatwave Sync Years (1980-2026):", hw_years)
