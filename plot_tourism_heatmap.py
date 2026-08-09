#!/usr/bin/env python3
"""
Fetch monthly tourism OVERNIGHT STAYS data from Eurostat (dataset tour_occ_nim - Notti trascorse)
for Mediterranean countries and generate:
1. Total Overnight Stays Heatmap (months x years + yearly total in millions of nights)
2. Line Chart comparing Foreign Stays (Red Line) vs Domestic Stays (Blue Line) over time (1990-2026),
   computing DOM = TOTAL - FOR when DOM is missing.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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


def download_tourism_category(category_code):
    """Download monthly overnight stays (tour_occ_nim) for a specific c_resid category from Eurostat."""
    countries_str = '+'.join(COUNTRIES.keys())
    url = (
        f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
        f"tour_occ_nim/M.{category_code}.NR.I551-I553.{countries_str}"
        f"?format=SDMX-CSV&compressed=false"
    )
    
    print(f"Downloading overnight stays (tour_occ_nim) for c_resid={category_code} from Eurostat...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    print(f"Downloaded {len(df)} rows for {category_code}")
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


def generate_total_heatmap(df_total, country_code, country_name):
    """Generate two stacked heatmaps in the same figure: monthly total nights (ax1) and yearly total (ax2)."""
    cdf = df_total[df_total['geo'] == country_code].copy()
    if cdf.empty:
        print(f"No total data for {country_name}")
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
        cbar_kws={'label': 'Notti Trascorse Mensili (milioni)'}
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

    ax1.set_title(f'Pernottamenti Turistici Totali Mensili – {country_name}\n(milioni di notti trascorse · bordo verde = record assoluto del mese)', fontsize=14)
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
        cbar_kws={'label': 'Totale Anno (milioni notti)'}
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
    
    output_path = os.path.join(OUTPUT_DIR, f'tourism_heatmap_total_{country_code.lower()}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    
    # Also save as backward compatible tourism_heatmap_{country}.png
    compat_path = os.path.join(OUTPUT_DIR, f'tourism_heatmap_{country_code.lower()}.png')
    plt.savefig(compat_path, dpi=200, bbox_inches='tight')
    
    plt.close()
    print(f"Saved: {output_path}")


def generate_line_chart(df_total, df_foreign, df_domestic, country_code, country_name):
    """Generate a single line chart comparing Foreign Stays (Red Line) vs Domestic Stays (Blue Line) in millions of nights.
       Computes DOM = TOTAL - FOR when Eurostat DOM is unpopulated.
    """
    cdf_tot = df_total[df_total['geo'] == country_code].copy()
    cdf_for = df_foreign[df_foreign['geo'] == country_code].copy()
    cdf_dom = df_domestic[df_domestic['geo'] == country_code].copy()
    
    if cdf_tot.empty or cdf_for.empty:
        print(f"Missing line chart data for {country_name}")
        return

    # Pivot annual totals
    piv_tot = cdf_tot.pivot_table(index='month', columns='year', values='value_millions', aggfunc='first')
    tot_tot = piv_tot.sum(axis=0, skipna=True)
    cnt_tot = piv_tot.notna().sum(axis=0)
    tot_tot[cnt_tot < 10] = np.nan

    piv_for = cdf_for.pivot_table(index='month', columns='year', values='value_millions', aggfunc='first')
    tot_for = piv_for.sum(axis=0, skipna=True)
    cnt_for = piv_for.notna().sum(axis=0)
    tot_for[cnt_for < 10] = np.nan

    piv_dom = cdf_dom.pivot_table(index='month', columns='year', values='value_millions', aggfunc='first')
    tot_dom = piv_dom.sum(axis=0, skipna=True)
    cnt_dom = piv_dom.notna().sum(axis=0)
    tot_dom[cnt_dom < 10] = np.nan

    # Fill DOM = TOTAL - FOR whenever DOM is missing
    years = sorted(list(set(tot_tot.dropna().index).union(set(tot_for.dropna().index))))
    dom_filled = []
    for y in years:
        d_val = tot_dom.get(y, np.nan)
        if pd.isna(d_val):
            t_val = tot_tot.get(y, np.nan)
            f_val = tot_for.get(y, np.nan)
            if pd.notna(t_val) and pd.notna(f_val):
                d_val = t_val - f_val
        dom_filled.append(d_val)

    df_line = pd.DataFrame({
        'Stranieri': [tot_for.get(y, np.nan) for y in years],
        'Locali': dom_filled
    }, index=years)

    fig, ax = plt.subplots(figsize=(max(16, len(years) * 0.45), 6.5))

    # Plot Red Line (Stranieri / Esteri)
    ax.plot(
        df_line.index, df_line['Stranieri'],
        color='#ef4444', linewidth=3, marker='o', markersize=6,
        label='Notti Stranieri / Esteri (FOR)'
    )

    # Plot Blue Line (Locali / Residenti)
    ax.plot(
        df_line.index, df_line['Locali'],
        color='#2563eb', linewidth=3, marker='s', markersize=6,
        label='Notti Locali / Residenti (DOM)'
    )

    # Annotate numbers on point markers for key years or all complete years
    for y in years:
        val_for = df_line.loc[y, 'Stranieri']
        val_dom = df_line.loc[y, 'Locali']

        if pd.notna(val_for):
            ax.annotate(
                f'{val_for:.1f}', (y, val_for),
                textcoords="offset points", xytext=(0, 8), ha='center',
                fontsize=7.5, fontweight='bold', color='#dc2626'
            )
        if pd.notna(val_dom):
            ax.annotate(
                f'{val_dom:.1f}', (y, val_dom),
                textcoords="offset points", xytext=(0, -12), ha='center',
                fontsize=7.5, fontweight='bold', color='#1d4ed8'
            )

    ax.set_title(f'Confronto Pernottamenti Stranieri vs Residenti – {country_name}\n(Milioni di notti trascorse annue · Eurostat tour_occ_nim)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Anno', fontsize=12)
    ax.set_ylabel('Notti Trascorse Annue (milioni)', fontsize=12)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha='right')

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=11, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f'tourism_line_{country_code.lower()}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def save_json_data(df_total, df_foreign, df_domestic):
    """Save processed data as JSON for web visualization."""
    result = {'total': {}, 'foreign': {}, 'domestic': {}}
    
    for cat_key, df in [('total', df_total), ('foreign', df_foreign), ('domestic', df_domestic)]:
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
            
            result[cat_key][country_code] = {
                'name': country_name,
                'data': country_data
            }
    
    output_path = os.path.join(OUTPUT_DIR, 'tourism_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Download datasets (tour_occ_nim for overnight stays)
    raw_total = download_tourism_category('TOTAL')
    raw_foreign = download_tourism_category('FOR')
    raw_domestic = download_tourism_category('DOM')
    
    # 2. Process
    df_total = process_data(raw_total)
    df_foreign = process_data(raw_foreign)
    df_domestic = process_data(raw_domestic)
    
    # 3. Save JSON
    save_json_data(df_total, df_foreign, df_domestic)
    
    # 4. Generate Total Heatmap + Line Chart per country
    for country_code, country_name in COUNTRIES.items():
        generate_total_heatmap(df_total, country_code, country_name)
        generate_line_chart(df_total, df_foreign, df_domestic, country_code, country_name)
    
    print("\n✅ Total Overnight Stays Heatmaps & Foreign vs Domestic Line Charts generated successfully!")


if __name__ == '__main__':
    main()
