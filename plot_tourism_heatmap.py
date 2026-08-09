#!/usr/bin/env python3
"""
Fetch monthly tourism data (overnight stays) from Eurostat for Mediterranean countries
and generate intensity heatmaps (months x years) with record highlighting.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import requests
import os
import json

# ── Configuration ──────────────────────────────────────────────────────────────
COUNTRIES = {
    'ES': 'Spagna',
    'FR': 'Francia',
    'IT': 'Italia',
    'EL': 'Grecia',
    'PT': 'Portogallo'
}

MONTH_NAMES = {
    1: 'Gen', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mag', 6: 'Giu',
    7: 'Lug', 8: 'Ago', 9: 'Set', 10: 'Ott', 11: 'Nov', 12: 'Dic'
}

OUTPUT_DIR = 'docs'


def download_tourism_dataset(dataset_code):
    """Download monthly tourism dataset (tour_occ_nim or tour_occ_arm) from Eurostat SDMX-CSV API for Foreign Tourists (FOR)."""
    countries_str = '+'.join(COUNTRIES.keys())
    # c_resid = FOR (Foreign / International tourists only)
    url = (
        f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
        f"{dataset_code}/M.FOR.NR.I551-I553.{countries_str}"
        f"?format=SDMX-CSV&compressed=false"
    )
    
    print(f"Downloading {dataset_code} from Eurostat...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    print(f"Downloaded {len(df)} rows for {dataset_code}")
    return df


def process_data(df):
    """Process raw Eurostat data into a clean DataFrame with year, month, country, value."""
    df = df.copy()
    
    df['year'] = df['TIME_PERIOD'].str[:4].astype(int)
    df['month'] = df['TIME_PERIOD'].str[5:7].astype(int)
    
    df['country'] = df['geo'].map(COUNTRIES)
    df['value'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['value'])
    df['value_millions'] = df['value'] / 1_000_000
    
    return df[['country', 'geo', 'year', 'month', 'value', 'value_millions']]


def generate_intensity_heatmap(df, country_code, country_name, metric_key='stays', metric_title='Pernottamenti Turistici'):
    """Generate two stacked heatmaps in the same figure: monthly values (ax1) and yearly total (ax2)."""
    cdf = df[df['geo'] == country_code].copy()
    if cdf.empty:
        print(f"No data for {country_name} ({metric_key})")
        return
    
    # 1. Pivot months (1 to 12)
    pivot_months = cdf.pivot_table(index='month', columns='year', values='value_millions', aggfunc='first')
    pivot_months = pivot_months.sort_index()
    
    # Calculate yearly total
    yearly_totals = pivot_months.sum(axis=0, skipna=True)
    month_counts = pivot_months.notna().sum(axis=0)
    yearly_totals[month_counts < 10] = np.nan
    
    pivot_total = pd.DataFrame([yearly_totals], index=['TOTALE'], columns=pivot_months.columns)
    
    # Rename month index to Gen-Dic
    pivot_months.index = [MONTH_NAMES.get(m, str(m)) for m in pivot_months.index]
    
    # Records per month
    records_months = {}
    for month_label in pivot_months.index:
        row = pivot_months.loc[month_label].dropna()
        if len(row) > 0:
            records_months[month_label] = row.idxmax()
            
    # Record year for total
    record_total_year = yearly_totals.dropna().idxmax() if len(yearly_totals.dropna()) > 0 else None

    # Annotations
    annot_months = pivot_months.round(1).astype(str).replace('nan', '')
    for col in annot_months.columns:
        annot_months[col] = annot_months[col].apply(lambda x: '' if x == 'nan' else x)

    annot_total = pivot_total.round(1).astype(str).replace('nan', '')
    for col in annot_total.columns:
        annot_total[col] = annot_total[col].apply(lambda x: '' if x == 'nan' else x)

    # 2 Subplots in the same figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(max(24, len(pivot_months.columns) * 0.55), 8.5), 
        gridspec_kw={'height_ratios': [12, 1.3], 'hspace': 0.15}
    )

    # Top Heatmap: Monthly values
    sns.heatmap(
        pivot_months,
        cmap='YlOrRd', ax=ax1,
        annot=annot_months, fmt="",
        annot_kws={"size": 6},
        linewidths=0.5, linecolor='lightgray',
        xticklabels=False,
        cbar_kws={'label': f'{metric_title} Mensili (milioni)'}
    )
    
    # Draw green rectangles for monthly records
    col_list = list(pivot_months.columns)
    row_list = list(pivot_months.index)
    for month_label, record_year in records_months.items():
        if record_year in col_list and month_label in row_list:
            col_idx = col_list.index(record_year)
            row_idx = row_list.index(month_label)
            ax1.add_patch(plt.Rectangle(
                (col_idx, row_idx), 1, 1,
                fill=False, edgecolor='#00cc44', lw=2.5, zorder=10
            ))

    ax1.set_title(f'{metric_title} Internazionali – {country_name}\n(milioni · bordo verde = record assoluto del mese)', fontsize=14)
    ax1.set_xlabel('')
    ax1.set_ylabel('Mese', fontsize=12)

    # Bottom Heatmap: Yearly Total
    sns.heatmap(
        pivot_total,
        cmap='YlOrRd', ax=ax2,
        annot=annot_total, fmt="",
        annot_kws={"size": 7, "weight": "bold"},
        linewidths=0.5, linecolor='lightgray',
        xticklabels=True,
        cbar_kws={'label': 'Totale Anno (milioni)'}
    )

    # Draw green rectangle for total year record
    if record_total_year in col_list:
        col_idx = col_list.index(record_total_year)
        ax2.add_patch(plt.Rectangle(
            (col_idx, 0), 1, 1,
            fill=False, edgecolor='#00cc44', lw=3, zorder=10
        ))

    ax2.set_xlabel('Anno', fontsize=12)
    ax2.set_ylabel('', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save with specific metric name and also backward compatible name for stays
    output_path = os.path.join(OUTPUT_DIR, f'tourism_heatmap_{metric_key}_{country_code.lower()}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    
    if metric_key == 'stays':
        compat_path = os.path.join(OUTPUT_DIR, f'tourism_heatmap_{country_code.lower()}.png')
        plt.savefig(compat_path, dpi=200, bbox_inches='tight')
        
    plt.close()
    print(f"Saved: {output_path}")


def save_json_data(df_stays, df_arrivals):
    """Save processed data as JSON for potential web visualization."""
    result = {'stays': {}, 'arrivals': {}}
    
    for metric_name, df in [('stays', df_stays), ('arrivals', df_arrivals)]:
        for country_code, country_name in COUNTRIES.items():
            cdf = df[df['geo'] == country_code].copy()
            if cdf.empty:
                continue
            
            country_data = {}
            for _, row in cdf.iterrows():
                year = int(row['year'])
                month = int(row['month'])
                if year not in country_data:
                    country_data[year] = {}
                country_data[year][month] = round(row['value_millions'], 2)
            
            result[metric_name][country_code] = {
                'name': country_name,
                'data': country_data
            }
    
    output_path = os.path.join(OUTPUT_DIR, 'tourism_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Download datasets
    raw_stays = download_tourism_dataset('tour_occ_nim')
    raw_arrivals = download_tourism_dataset('tour_occ_arm')
    
    # 2. Process
    df_stays = process_data(raw_stays)
    df_arrivals = process_data(raw_arrivals)
    
    # 3. Save JSON
    save_json_data(df_stays, df_arrivals)
    
    # 4. Generate intensity heatmaps for both metrics
    for country_code, country_name in COUNTRIES.items():
        generate_intensity_heatmap(df_stays, country_code, country_name, metric_key='stays', metric_title='Pernottamenti Turistici')
        generate_intensity_heatmap(df_arrivals, country_code, country_name, metric_key='arrivals', metric_title='Arrivi / Visitatori Turistici')
    
    print("\n✅ All tourism heatmaps (stays & arrivals) generated!")


if __name__ == '__main__':
    main()
