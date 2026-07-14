import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import timedelta

def compute_heatwaves(city_file):
    df = pd.read_csv(city_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df[df['date'].dt.year >= 1940].copy()
    
    # Restrict to June, July, August
    df = df[df['date'].dt.month.isin([6, 7, 8])].copy()
    
    # Calculate fixed 99th percentile based on 1975-2000 period
    baseline_data = df[(df['date'].dt.year >= 1975) & (df['date'].dt.year <= 2000)]['temperature_2m_max'].dropna()
    threshold = baseline_data.quantile(0.99) if not baseline_data.empty else 30
    baseline_mean = baseline_data.mean() if not baseline_data.empty else 25
    baseline_std = baseline_data.std() if not baseline_data.empty else 2
    
    df['threshold'] = threshold
    
    # Find heatwaves: days where max temp > threshold
    df['is_hot'] = df['temperature_2m_max'] > df['threshold']
    
    # Group consecutive hot days
    hws = []
    current_hw = []
    
    for i, row in df.iterrows():
        if row['is_hot']:
            if len(current_hw) > 0 and (row['date'] - current_hw[-1]['date']).days > 1:
                if len(current_hw) >= 3:
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
                        'baseline_std': baseline_std
                    })
                current_hw = []
            current_hw.append(row)
        else:
            if len(current_hw) >= 3:
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
                    'baseline_std': baseline_std
                })
            current_hw = []
                
    if len(current_hw) >= 3:
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
            'baseline_std': baseline_std
        })
        
    return hws

def plot_waves(cities, output_file, region_title):
    fig, ax = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Sort cities for vertical stacking
    cities.reverse()
    
    y_labels = []
    y_ticks = []
    
    # Create summer-only continuous axis mapping
    all_dates = pd.date_range('1975-01-01', '2026-12-31', freq='D')
    summer_dates = all_dates[all_dates.month.isin([6, 7, 8])]
    date_to_x = {d.date(): i for i, d in enumerate(summer_dates)}
    max_x = len(date_to_x)
    
    date_counts = {d.date(): 0 for d in summer_dates}
    
    for i, city in enumerate(cities):
        city_name = os.path.basename(city).replace('.csv', '')
        y_base = i * 20  # Base y-level for the city
        
        y_labels.append(city_name)
        y_ticks.append(y_base)
        
        # Plot baseline line
        ax.plot([0, max_x], [y_base, y_base], color='#4a4e69', lw=1, zorder=1)
        
        hws = compute_heatwaves(city)
        hws = [hw for hw in hws if hw['start'].year >= 1975] # Filter to 1975+
        
        for hw in hws:
            hw_dates = pd.date_range(hw['start'], hw['end'], freq='D')
            for d in hw_dates:
                if d.date() in date_counts:
                    date_counts[d.date()] += 1
        
        # Plot each wave (altezza = anomalia assoluta in gradi rispetto alla media 1975-2000)
        for hw in hws:
            h = max(0, hw['max_temp'] - hw['baseline_mean'])
            
            width_days = max(hw['duration'], 2)
            sigma = width_days / 2
            
            dates = pd.date_range(hw['center'] - pd.Timedelta(days=int(width_days*3)), 
                                  hw['center'] + pd.Timedelta(days=int(width_days*3)), freq='D')
            
            x_vals = []
            y_vals = []
            
            for d in dates:
                if d.date() in date_to_x:
                    x_idx = date_to_x[d.date()]
                    diff_days = (d - hw['center']).days
                    y_wave = y_base + h * np.exp(- (diff_days**2) / (2 * sigma**2))
                    x_vals.append(x_idx)
                    y_vals.append(y_wave)
            
            if len(x_vals) > 0:
                color = plt.cm.inferno(min(1.0, h / 20)) 
                ax.fill_between(x_vals, y_base, y_vals, color=color, alpha=0.8, zorder=i+2)
                ax.plot(x_vals, y_vals, color='white', lw=0.5, alpha=0.5, zorder=i+2)

    # Sync events logic
    sync_dates = [d for d, count in date_counts.items() if count >= len(cities) * 0.5]
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
    for event in sync_events:
        center_day = event[len(event)//2]
        x_idx = date_to_x[center_day]
        ax.axvline(x=x_idx, color=sync_color, linestyle='--', lw=1.5, zorder=1, alpha=0.8)
        sync_years.add(center_day.year)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, color='white', fontsize=12)
    
    # Custom x-axis ticks for summer-only timeline
    xticks = []
    xticklabels = []
    xtick_colors = []
    for y in range(1975, 2030, 5):
        d = pd.Timestamp(f'{y}-06-01').date()
        if d in date_to_x:
            xticks.append(date_to_x[d])
            xticklabels.append(str(y))
            if y in sync_years:
                xtick_colors.append(sync_color)
            else:
                xtick_colors.append('white')
                
    for y in sync_years:
        if y % 5 != 0:
            d = pd.Timestamp(f'{y}-06-01').date()
            if d in date_to_x:
                xticks.append(date_to_x[d])
                xticklabels.append(str(y))
                xtick_colors.append(sync_color)
            
    ax.set_xticks(xticks)
    labels = ax.set_xticklabels(xticklabels, rotation=45, ha='right')
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
    
    plt.title(f"Ondate di Calore in Estate - {region_title} (Giugno-Luglio-Agosto, 1975-2026)", color='white', fontsize=20, pad=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, facecolor='#1a1a2e')
    plt.close()

if __name__ == "__main__":
    cities = sorted(glob.glob('data/*.csv'))
    out_path = 'docs/heatwaves_europe.png'
    plot_waves(cities, out_path, 'Europa')
    print(f"Plot saved to {out_path}")
