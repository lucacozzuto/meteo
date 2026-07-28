import requests
import io
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# EFFIS official historical annual burnt area (1980-2001) in hectares for 5 Mediterranean countries
HIST_SOUTHERN = {
    'Spain': {
        1980: 256000, 1981: 289000, 1982: 152000, 1983: 125000, 1984: 165000,
        1985: 484000, 1986: 262000, 1987: 145000, 1988: 138000, 1989: 426000,
        1990: 203000, 1991: 260000, 1992: 105000, 1993: 89000,  1994: 437000,
        1995: 143000, 1996: 60000,  1997: 98000,  1998: 133000, 1999: 82000,
        2000: 188000, 2001: 93000
    },
    'Italy': {
        1980: 144200, 1981: 265000, 1982: 130000, 1983: 212600, 1984: 75000,
        1985: 190600, 1986: 119000, 1987: 120600, 1988: 186400, 1989: 95000,
        1990: 195300, 1991: 99800,  1992: 105400, 1993: 203700, 1994: 116400,
        1995: 47800,  1996: 58200,  1997: 111400, 1998: 155500, 1999: 71100,
        2000: 114600, 2001: 76400
    },
    'Greece': {
        1980: 33000,  1981: 81000,  1982: 24000,  1983: 19000,  1984: 31000,
        1985: 105000, 1986: 24000,  1987: 46000,  1988: 110000, 1989: 41000,
        1990: 39000,  1991: 23000,  1992: 71000,  1993: 54000,  1994: 58000,
        1995: 27000,  1996: 25000,  1997: 52000,  1998: 93000,  1999: 83000,
        2000: 145000, 2001: 28000
    },
    'Portugal': {
        1980: 44000,  1981: 89000,  1982: 39000,  1983: 47000,  1984: 53000,
        1985: 146000, 1986: 98000,  1987: 76000,  1988: 22000,  1989: 126000,
        1990: 137000, 1991: 182000, 1992: 57000,  1993: 49000,  1994: 77000,
        1995: 169000, 1996: 88000,  1997: 30000,  1998: 158000, 1999: 70000,
        2000: 159000, 2001: 111000
    },
    'France': {
        1980: 22000,  1981: 27000,  1982: 54000,  1983: 53000,  1984: 27000,
        1985: 49000,  1986: 52000,  1987: 20000,  1988: 13000,  1989: 75000,
        1990: 76000,  1991: 10000,  1992: 17000,  1993: 26000,  1994: 25000,
        1995: 18000,  1996: 11000,  1997: 22000,  1998: 18000,  1999: 16000,
        2000: 24000,  2001: 21000
    }
}

MEDITERRANEAN_5 = {
    'France': 46.2, 'Italy': 41.9, 'Spain': 40.4, 'Portugal': 39.4, 'Greece': 39.0
}

def generate_custom_p80_p90_heatmap(matrix_ha, title, subtitle, output_path, figsize=(30, 8.5)):
    # Calculate 80th percentile for individual countries (rows 0..N-2)
    p80_countries = matrix_ha.iloc[:-1].apply(lambda row: np.percentile(row.dropna(), 80), axis=1)
    
    # Calculate 90th percentile for TOTALE MEDITERRANEO (last row)
    p90_total = np.percentile(matrix_ha.loc['TOTALE MEDITERRANEO'].dropna(), 90)

    heatmap_data = pd.DataFrame(0, index=matrix_ha.index, columns=matrix_ha.columns)
    annot_data = pd.DataFrame("", index=matrix_ha.index, columns=matrix_ha.columns, dtype=object)

    # 1. Fill country rows based on 80th percentile threshold
    for idx in matrix_ha.index[:-1]:
        thresh = p80_countries[idx]
        for col in matrix_ha.columns:
            val = matrix_ha.loc[idx, col]
            if val >= thresh and val > 0:
                heatmap_data.loc[idx, col] = 1
                if val >= 1000000:
                    annot_data.loc[idx, col] = f"{val/1000000:.2f}M"
                elif val >= 1000:
                    annot_data.loc[idx, col] = f"{int(round(val/1000))}k"
                else:
                    annot_data.loc[idx, col] = f"{int(round(val))}ha"
            else:
                heatmap_data.loc[idx, col] = 0

    # 2. Fill TOTALE MEDITERRANEO row based on 90th percentile threshold
    for col in matrix_ha.columns:
        val = matrix_ha.loc['TOTALE MEDITERRANEO', col]
        if val >= p90_total:
            heatmap_data.loc['TOTALE MEDITERRANEO', col] = 1
            annot_data.loc['TOTALE MEDITERRANEO', col] = f"{val/1000000:.2f}M" if val >= 1000000 else f"{int(round(val/1000))}k"
        else:
            heatmap_data.loc['TOTALE MEDITERRANEO', col] = 0

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # 0 = White (Normal), 1 = Red/Orange (>= threshold)
    cmap = mcolors.ListedColormap(['#ffffff', '#ff5722'])

    sns.heatmap(
        heatmap_data, 
        cmap=cmap, 
        ax=ax, 
        annot=annot_data.values, 
        fmt="", 
        annot_kws={"size": 8.0, "color": "black", "weight": "bold"},
        linewidths=0.3, 
        linecolor='lightgray', 
        xticklabels=True, 
        yticklabels=True,
        cbar=False,
        vmin=0, 
        vmax=1
    )

    # Draw horizontal separating line above the TOTALE MEDITERRANEO row
    n_rows = matrix_ha.shape[0]
    ax.axhline(n_rows - 1, color='cyan', linewidth=3)

    # Highlight COLUMNS (years) ONLY when TOTALE MEDITERRANEO is >= 90th percentile
    column_highlight = [col for col in matrix_ha.columns if matrix_ha.loc['TOTALE MEDITERRANEO', col] >= p90_total]

    for i, year in enumerate(matrix_ha.columns):
        if year in column_highlight:
            ax.add_patch(plt.Rectangle((i, 0), 1, matrix_ha.shape[0], fill=False, edgecolor='blue', lw=3.0))
            labels = ax.get_xticklabels()
            if i < len(labels):
                labels[i].set_weight("bold")
                labels[i].set_color("blue")

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(subtitle, fontsize=12, labelpad=10)
    ax.set_ylabel('Paese Mediterraneo (Da Nord a Sud)', fontsize=12, labelpad=10)

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Custom P80/P90 Mediterranean wildfire heatmap saved to {output_path}")

def main():
    print("--- DOWNLOADING GWIS AND OWID DATASETS ---")
    url_gwis = "https://ourworldindata.org/grapher/annual-area-burnt-by-wildfires-gwis.csv"
    url_weekly = "https://ourworldindata.org/grapher/weekly-area-burnt-by-wildfires.csv"

    r_g = requests.get(url_gwis, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    df_g = pd.read_csv(io.StringIO(r_g.text))

    r_w = requests.get(url_weekly, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    df_w = pd.read_csv(io.StringIO(r_w.text))
    df_w['Day'] = pd.to_datetime(df_w['Day'])
    df_w['year'] = df_w['Day'].dt.year

    print("\n--- GENERATING MEDITERRANEAN HEATMAP (80th Pct Countries / 90th Pct Total Columns) ---")
    years_1980 = list(range(1980, 2027))
    matrix_med = pd.DataFrame(index=list(MEDITERRANEAN_5.keys()), columns=years_1980, dtype=float).fillna(0.0)

    for country in MEDITERRANEAN_5.keys():
        if country in HIST_SOUTHERN:
            for y, val in HIST_SOUTHERN[country].items():
                matrix_med.loc[country, y] = float(val)
                
        df_c_g = df_g[df_g['Entity'] == country]
        for _, row in df_c_g.iterrows():
            y = int(row['Year'])
            val = float(row['Yearly burned area across all land types'])
            if 2002 <= y <= 2024:
                matrix_med.loc[country, y] = val

        df_c_w = df_w[df_w['Entity'] == country]
        for y in [2025, 2026]:
            val = df_c_w[df_c_w['year'] == y]['Area burnt by wildfires'].sum()
            matrix_med.loc[country, y] = float(val)

    matrix_med['lat'] = matrix_med.index.map(MEDITERRANEAN_5)
    matrix_med = matrix_med.sort_values('lat', ascending=False)
    formatted_index = [f"{c} ({lat:.1f}°N)" for c, lat in zip(matrix_med.index, matrix_med['lat'])]
    matrix_med.drop(columns=['lat'], inplace=True)
    matrix_med.index = formatted_index

    # Add TOTAL MEDITERRANEAN Row at the bottom
    total_row = matrix_med.sum(axis=0)
    matrix_med.loc['TOTALE MEDITERRANEO'] = total_row

    generate_custom_p80_p90_heatmap(
        matrix_ha=matrix_med,
        title='Superficie Forestale Bruciata nei Paesi del Mediterraneo (1980–2026) [80° Pct Paesi — 90° Pct Totale]',
        subtitle='Anno (Evidenziate in Rosso le Annate > 80° Percentile del Paese — Rettangolo Blu sulla Colonna se il Totale > 90° Percentile)',
        output_path='docs/record_heatmap_wildfires_mediterranean.png',
        figsize=(30, 8.5)
    )

if __name__ == "__main__":
    main()
