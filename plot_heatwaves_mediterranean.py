import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import timedelta

MEDITERRANEAN_CITY_FILES = [
    'data/Athens.csv',
    'data/Lisbon.csv',
    'data/Madrid.csv',
    'data/Paris.csv',
    'data/Rome.csv'
]

def compute_heatwaves(city_file):
    df = pd.read_csv(city_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df[df['date'].dt.year >= 1980].copy()
    
    # Restrict to June, July, August
    df = df[df['date'].dt.month.isin([6, 7, 8])].copy()
    
    # Calculate fixed 90th percentile based on 1991-2020 period
    baseline_data = df[(df['date'].dt.year >= 1991) & (df['date'].dt.year <= 2020) & (df['date'].dt.month.isin([6, 7, 8]))]['temperature_2m_max'].dropna()
    threshold = baseline_data.quantile(0.90) if not baseline_data.empty else 30
    baseline_mean = baseline_data.mean() if not baseline_data.empty else 25
    baseline_std = baseline_data.std() if not baseline_data.empty else 2
    
    df['threshold'] = threshold
    
    # Find heatwaves: days where max temp > threshold
    df['is_hot'] = df['temperature_2m_max'] > df['threshold']
    
    # Group consecutive hot days >= 6
    hws = []
    current_hw = []
    
    for i, row in df.iterrows():
        if row['is_hot']:
            if len(current_hw) > 0 and (row['date'] - current_hw[-1]['date']).days > 1:
                if len(current_hw) >= 6:
                    hw_df = pd.DataFrame(current_hw)
                    start_date = hw_df['date'].min()
                    end_date = hw_df['date'].max()
                    max_temp = hw_df['temperature_2m_max'].max()
                    center_date = start_date + (end_date - start_date)/2
                    
                    hws.append({
                        'start': start_date,
                        'end': end_date,
                        'duration': (end_date - start_date).days + 1,
                        'max_temp': max_temp,
                        'center': center_date,
                        'baseline_mean': baseline_mean,
                        'baseline_std': baseline_std,
                        'dates': hw_df['date'].tolist()
                    })
                current_hw = []
            current_hw.append(row)
        else:
            if len(current_hw) >= 6:
                hw_df = pd.DataFrame(current_hw)
                start_date = hw_df['date'].min()
                end_date = hw_df['date'].max()
                max_temp = hw_df['temperature_2m_max'].max()
                center_date = start_date + (end_date - start_date)/2
                
                hws.append({
                    'start': start_date,
                    'end': end_date,
                    'duration': (end_date - start_date).days + 1,
                    'max_temp': max_temp,
                    'center': center_date,
                    'baseline_mean': baseline_mean,
                    'baseline_std': baseline_std,
                    'dates': hw_df['date'].tolist()
                })
            current_hw = []
            
    if len(current_hw) >= 6:
        hw_df = pd.DataFrame(current_hw)
        start_date = hw_df['date'].min()
        end_date = hw_df['date'].max()
        max_temp = hw_df['temperature_2m_max'].max()
        center_date = start_date + (end_date - start_date)/2
        
        hws.append({
            'start': start_date,
            'end': end_date,
            'duration': (end_date - start_date).days + 1,
            'max_temp': max_temp,
            'center': center_date,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'dates': hw_df['date'].tolist()
        })
        
    return hws

def plot_waves(city_files, output_file):
    plt.figure(figsize=(24, 12), dpi=200)
    ax = plt.gca()
    ax.set_facecolor('#1a1a2e')
    plt.gcf().patch.set_facecolor('#1a1a2e')
    
    cities_data = []
    for f in city_files:
        cname = os.path.basename(f).replace('.csv', '')
        hws = compute_heatwaves(f)
        cities_data.append((cname, hws))
        
    # Sort cities by mean heatwave count or latitude
    cities_data.sort(key=lambda x: x[0])
    
    # Timeline: consecutive summer days 1980-2026
    timeline_days = []
    for y in range(1980, 2027):
        start = pd.Timestamp(f'{y}-06-01')
        end = pd.Timestamp(f'{y}-08-31')
        timeline_days.extend(pd.date_range(start, end).date)
        
    date_to_x = {d: i for i, d in enumerate(timeline_days)}
    max_x = len(timeline_days)
    
    y_ticks = []
    y_labels = []
    
    date_counts = {}
    
    for i, (cname, hws) in enumerate(cities_data):
        y_base = i * 2.5
        y_ticks.append(y_base)
        y_labels.append(cname)
        
        # Draw base line
        ax.plot([0, max_x], [y_base, y_base], color='#4a4e69', lw=1, alpha=0.5)
        
        # Plot waves
        for hw in hws:
            for d in hw['dates']:
                dt = d.date()
                date_counts[dt] = date_counts.get(dt, 0) + 1
                
            x_vals = [date_to_x[d.date()] for d in hw['dates'] if d.date() in date_to_x]
            if len(x_vals) > 0:
                h = hw['duration']
                y_vals = [y_base + (h / 6.0) * np.sin(np.pi * (x - x_vals[0]) / (x_vals[-1] - x_vals[0] + 1e-5)) for x in x_vals]
                color = plt.cm.inferno(min(1.0, h / 20))
                ax.fill_between(x_vals, y_base, y_vals, color=color, alpha=0.85, zorder=i+2)
                ax.plot(x_vals, y_vals, color='white', lw=0.5, alpha=0.5, zorder=i+2)

    # Sync events logic (>= 25% of cities in heatwave simultaneously)
    sync_dates = [d for d, count in date_counts.items() if count >= len(city_files) * 0.25]
    sync_events = []
    if sync_dates:
        sync_dates.sort()
        current_event = [sync_dates[0]]
        for d in sync_dates[1:]:
            if (d - current_event[-1]).days == 1:
                current_event.append(d)
            else:
                sync_events.append(current_event)
                current_event = [d]
        sync_events.append(current_event)
        
    sync_color = '#00ffcc'
    sync_years = set()
    sync_events_x = []
    for event in sync_events:
        center_day = event[len(event)//2]
        x_idx = date_to_x[center_day]
        ax.axvline(x=x_idx, color=sync_color, linestyle='--', lw=1.5, zorder=1, alpha=0.8)
        sync_years.add(center_day.year)
        sync_events_x.append((center_day.year, x_idx))

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, color='white', fontsize=12)
    
    # Custom x-axis ticks
    xticks = []
    xticklabels = []
    xtick_colors = []
    
    for y in range(1980, 2027, 5):
        if y not in sync_years:
            d = pd.Timestamp(f'{y}-07-15').date()
            if d in date_to_x:
                xticks.append(date_to_x[d])
                xticklabels.append(str(y))
                xtick_colors.append('white')
                
    for y, x_idx in sync_events_x:
        xticks.append(x_idx)
        xticklabels.append(str(y))
        xtick_colors.append(sync_color)
            
    ax.set_xticks(xticks)
    labels = ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
    for label, color in zip(labels, xtick_colors):
        label.set_color(color)
        if color == sync_color:
            label.set_weight('bold')
            
    ax.set_xlim(0, max_x)
    
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#4a4e69')
    ax.spines['bottom'].set_color('#4a4e69')
    
    plt.title("Ondate di Calore Estive nei Paesi del Mediterraneo (Giugno-Luglio-Agosto, 1980-2026)", color='white', fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, facecolor='#1a1a2e')
    plt.close()
    
    med_years = sorted(list(sync_years))
    print(f"Mediterranean Heatwaves Plot saved to {output_file}")
    print("Mediterranean Heatwave Sync Years:", med_years)
    return med_years

if __name__ == '__main__':
    plot_waves(MEDITERRANEAN_CITY_FILES, 'docs/heatwaves_mediterranean.png')
