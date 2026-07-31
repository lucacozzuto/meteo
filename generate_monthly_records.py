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

            # MAX TEMP RECORDS
            monthly_yearly_max = df.groupby(['month', 'year'])['temperature_2m_max'].max().reset_index()
            monthly_yearly_max = monthly_yearly_max.sort_values(by=['month', 'year'])
            monthly_yearly_max['prev_max'] = monthly_yearly_max.groupby('month')['temperature_2m_max'].transform(lambda x: x.cummax().shift(1))
            monthly_yearly_max['is_record'] = (monthly_yearly_max['temperature_2m_max'] > monthly_yearly_max['prev_max']).astype(int)
            monthly_yearly_max.loc[monthly_yearly_max['year'] < 1955, 'is_record'] = -1

            # MIN TEMP RECORDS (Warmest Nights)
            monthly_yearly_min = df.groupby(['month', 'year'])['temperature_2m_min'].max().reset_index()
            monthly_yearly_min = monthly_yearly_min.sort_values(by=['month', 'year'])
            monthly_yearly_min['prev_min'] = monthly_yearly_min.groupby('month')['temperature_2m_min'].transform(lambda x: x.cummax().shift(1))
            monthly_yearly_min['is_record'] = (monthly_yearly_min['temperature_2m_min'] > monthly_yearly_min['prev_min']).astype(int)
            monthly_yearly_min.loc[monthly_yearly_min['year'] < 1955, 'is_record'] = -1

            # MEAN TEMP (Daily Mean)
            df['temp_mean'] = (df['temperature_2m_max'] + df['temperature_2m_min']) / 2
            
            # MEAN TEMP PER MONTH
            monthly_yearly_mean = df.groupby(['month', 'year'])['temp_mean'].mean().reset_index()

            years = sorted(monthly_yearly_max['year'].unique().tolist())
            
            # ANNUAL ANOMALIES (1940-2025 full years + 2026 ongoing)
            annual_years_list = [y for y in years if y < 2026]
            annual_df = df[df['year'] < 2026]
            annual_mean = annual_df.groupby('year')['temp_mean'].mean().reindex(annual_years_list)
            baseline_mean = annual_mean.mean()
            annual_anomalies_list = (annual_mean - baseline_mean).round(2)

            import numpy as np
            annual_anomalies = annual_anomalies_list.replace({np.nan: None}).tolist()
            annual_years = annual_years_list

            # Include 2026 ongoing anomaly
            df_2026 = df[df['year'] == 2026]
            if not df_2026.empty:
                md_2026 = set(df_2026['date'].dt.strftime('%m-%d'))
                mean_2026 = df_2026['temp_mean'].mean()
                baseline_2026_period = df[(df['year'] < 2026) & (df['date'].dt.strftime('%m-%d').isin(md_2026))]['temp_mean'].mean()
                if pd.notna(mean_2026) and pd.notna(baseline_2026_period):
                    anom_2026 = float(round(mean_2026 - baseline_2026_period, 2))
                    annual_years.append(2026)
                    annual_anomalies.append(anom_2026)

            # Heatwaves calculation
            df_summer = df[df['month'].isin([6, 7, 8])].copy()
            df_base_summer = df[(df['date'].dt.year >= 1991) & (df['date'].dt.year <= 2020) & (df['date'].dt.month.isin([6, 7, 8]))]
            
            def calc_waves(col_name):
                threshold = df_base_summer[col_name].quantile(0.90) if not df_base_summer.empty else 30
                baseline_mean_summer = df_base_summer[col_name].mean() if not df_base_summer.empty else 25
                
                df_summer['is_hot'] = df_summer[col_name] > threshold
                waves = []
                current_hw = []
                
                def add_hw(hw_list):
                    if len(hw_list) >= 6:
                        hw_df = pd.DataFrame(hw_list)
                        start_date = hw_df['date'].min()
                        end_date = hw_df['date'].max()
                        center_date = start_date + (end_date - start_date)/2
                        waves.append({
                            'year': start_date.year,
                            'start': start_date.strftime('%Y-%m-%d'),
                            'end': end_date.strftime('%Y-%m-%d'),
                            'center': center_date.strftime('%Y-%m-%d'),
                            'duration': len(hw_list),
                            'max_temp': float(round(hw_df[col_name].max(), 1)),
                            'anomaly': float(round(max(0, hw_df[col_name].max() - baseline_mean_summer), 1)),
                            'baseline_mean': float(round(baseline_mean_summer, 1))
                        })

                for i, row in df_summer.iterrows():
                    if row['is_hot']:
                        if len(current_hw) > 0 and (row['date'] - current_hw[-1]['date']).days > 1:
                            add_hw(current_hw)
                            current_hw = []
                        current_hw.append(row)
                    else:
                        add_hw(current_hw)
                        current_hw = []
                add_hw(current_hw)
                return waves

            heatwaves = calc_waves('temperature_2m_max')
            night_heatwaves = calc_waves('temperature_2m_min')

            # Daily evolution 2026 vs baseline 1991-2020 (AEMET style)
            df['md'] = df['date'].dt.strftime('%m-%d')
            df_base_daily = df[(df['year'] >= 1991) & (df['year'] <= 2020) & (df['md'] != '02-29')]
            all_mds = [
                f"{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 32)
                if not (m == 2 and d > 28) and not (m in [4, 6, 9, 11] and d > 30)
            ]
            daily_normals_raw = df_base_daily.groupby('md')['temp_mean'].mean().reindex(all_mds)
            s_3 = pd.concat([daily_normals_raw, daily_normals_raw, daily_normals_raw])
            daily_normals_smoothed = s_3.rolling(window=21, center=True, min_periods=1).mean().iloc[365:730].round(2).tolist()
            dates_365 = [f"2026-{md}" for md in all_mds]
            md_to_norm = dict(zip(all_mds, daily_normals_smoothed))
            df_2026 = df[df['year'] == 2026].sort_values('date')
            dates_2026 = df_2026['date'].dt.strftime('%Y-%m-%d').tolist()
            temps_2026 = [round(x, 1) if pd.notna(x) else None for x in df_2026['temp_mean']]
            normals_2026 = [md_to_norm.get(md, None) for md in df_2026['md']]

            daily_2026 = {
                "dates": dates_2026,
                "temps": temps_2026,
                "normals": normals_2026,
                "dates_365": dates_365,
                "normals_365": daily_normals_smoothed
            }

            # TOP 10 HOTTEST SUMMERS (JJA Mean Temp)
            df_jja = df[df['month'].isin([6, 7, 8])].copy()
            summer_stats = []
            if not df_jja.empty:
                grouped_jja = df_jja.groupby('year')
                for y, group in grouped_jja:
                    n_days = len(group)
                    if n_days == 0:
                        continue
                    m_temp = float(round(group['temp_mean'].mean(), 2))
                    m_max = float(round(group['temperature_2m_max'].mean(), 1))
                    m_min = float(round(group['temperature_2m_min'].mean(), 1))
                    is_partial = (y == 2026 and n_days < 92)
                    summer_stats.append({
                        "year": int(y),
                        "mean_temp": m_temp,
                        "mean_max": m_max,
                        "mean_min": m_min,
                        "days": int(n_days),
                        "is_partial": is_partial
                    })
                
                # Sort descending by mean_temp
                summer_stats.sort(key=lambda x: x['mean_temp'], reverse=True)
                top_summers = []
                for rank_idx, s in enumerate(summer_stats[:10]):
                    s_copy = dict(s)
                    s_copy['rank'] = rank_idx + 1
                    top_summers.append(s_copy)
            else:
                top_summers = []

            city_data = {
                "years": years,
                "records": [],
                "temps": [],
                "records_min": [],
                "temps_min": [],
                "mean_temps": [],
                "annual_anomalies": annual_anomalies,
                "annual_years": annual_years,
                "heatwaves": heatwaves,
                "night_heatwaves": night_heatwaves,
                "daily_2026": daily_2026,
                "top_summers": top_summers
            }

            for m in range(1, 13):
                # Max Data
                m_data = monthly_yearly_max[monthly_yearly_max['month'] == m]
                m_data = m_data.set_index('year').reindex(years).reset_index()
                m_data['is_record'] = m_data['is_record'].fillna(-1).astype(int)
                m_data['temperature_2m_max'] = m_data['temperature_2m_max'].fillna(0)
                city_data["records"].append(m_data['is_record'].tolist())
                city_data["temps"].append(m_data['temperature_2m_max'].round(1).tolist())

                # Min Data
                m_data_min = monthly_yearly_min[monthly_yearly_min['month'] == m]
                m_data_min = m_data_min.set_index('year').reindex(years).reset_index()
                m_data_min['is_record'] = m_data_min['is_record'].fillna(-1).astype(int)
                m_data_min['temperature_2m_min'] = m_data_min['temperature_2m_min'].fillna(0)
                city_data["records_min"].append(m_data_min['is_record'].tolist())
                city_data["temps_min"].append(m_data_min['temperature_2m_min'].round(1).tolist())
                
                # Mean Data
                m_data_mean = monthly_yearly_mean[monthly_yearly_mean['month'] == m]
                m_data_mean = m_data_mean.set_index('year').reindex(years).reset_index()
                city_data["mean_temps"].append([round(x, 1) if pd.notna(x) else None for x in m_data_mean['temp_mean']])

            records[city] = city_data

    return dict(sorted(records.items()))

def compute_overall_top_summers(data_dir, specific_file=None):
    if specific_file and os.path.exists(os.path.join(data_dir, specific_file)):
        df_all = pd.read_csv(os.path.join(data_dir, specific_file))
    else:
        all_dfs = []
        if not os.path.exists(data_dir):
            return []
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                filepath = os.path.join(data_dir, file)
                df = pd.read_csv(filepath)
                all_dfs.append(df)
        if not all_dfs:
            return []
        df_all = pd.concat(all_dfs, ignore_index=True)

    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all['year'] = df_all['date'].dt.year
    df_all['month'] = df_all['date'].dt.month
    df_all['temp_mean'] = (df_all['temperature_2m_max'] + df_all['temperature_2m_min']) / 2
    
    df_jja = df_all[(df_all['year'] >= 1940) & (df_all['month'].isin([6, 7, 8]))].copy()
    if df_jja.empty:
        return []
        
    summer_stats = []
    grouped = df_jja.groupby('year')
    for y, group in grouped:
        n_days = group['date'].nunique()
        if n_days == 0:
            continue
        m_temp = float(round(group['temp_mean'].mean(), 2))
        m_max = float(round(group['temperature_2m_max'].mean(), 1))
        m_min = float(round(group['temperature_2m_min'].mean(), 1))
        is_partial = (y == 2026 and n_days < 92)
        summer_stats.append({
            "year": int(y),
            "mean_temp": m_temp,
            "mean_max": m_max,
            "mean_min": m_min,
            "days": int(n_days),
            "is_partial": is_partial
        })
        
    summer_stats.sort(key=lambda x: x['mean_temp'], reverse=True)
    top_summers = []
    for rank_idx, s in enumerate(summer_stats[:10]):
        s_copy = dict(s)
        s_copy['rank'] = rank_idx + 1
        top_summers.append(s_copy)
    return top_summers

def main():
    europe_records = get_monthly_records('data')
    italy_records = get_monthly_records('data_italy')

    europe_top_summers = compute_overall_top_summers('data', 'Europa.csv')
    italy_top_summers = compute_overall_top_summers('data_italy', 'Italia.csv')

    all_records = {
        'Europe': europe_records,
        'Italy': italy_records,
        'overall_top_summers': {
            'Europe': europe_top_summers,
            'Italy': italy_top_summers
        }
    }

    os.makedirs('docs', exist_ok=True)
    with open('docs/monthly_records.json', 'w') as f:
        json.dump(all_records, f)

    print("docs/monthly_records.json generated successfully with overall top summers from national datasets.")

if __name__ == "__main__":
    main()
