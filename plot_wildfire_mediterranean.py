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

def main():
    print("--- GENERATING MEDITERRANEAN WILDFIRE RECORD HEATMAP (1980-2026) ---")
    url_gwis = "https://ourworldindata.org/grapher/annual-area-burnt-by-wildfires-gwis.csv"
    url_weekly = "https://ourworldindata.org/grapher/weekly-area-burnt-by-wildfires.csv"

    r_g = requests.get(url_gwis, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    df_g = pd.read_csv(io.StringIO(r_g.text))

    r_w = requests.get(url_weekly, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    df_w = pd.read_csv(io.StringIO(r_w.text))
    if 'Day' in df_w.columns:
        df_w['Day'] = pd.to_datetime(df_w['Day'])
    elif 'Week' in df_w.columns:
        df_w['Day'] = pd.to_datetime(df_w['Week'] + '-1', format='%G-W%V-%u')
    elif 'Date' in df_w.columns:
        df_w['Day'] = pd.to_datetime(df_w['Date'])
    elif 'date' in df_w.columns:
        df_w['Day'] = pd.to_datetime(df_w['date'])
    df_w['year'] = df_w['Day'].dt.year

    years = list(range(1980, 2027))
    matrix_ha = pd.DataFrame(index=list(MEDITERRANEAN_5.keys()), columns=years, dtype=float).fillna(0.0)

    for country in MEDITERRANEAN_5.keys():
        # 1. 1980-2001 EFFIS
        if country in HIST_SOUTHERN:
            for y, val in HIST_SOUTHERN[country].items():
                matrix_ha.loc[country, y] = float(val)
                
        # 2. 2002-2024 GWIS MODIS
        df_c_g = df_g[df_g['Entity'] == country]
        for _, row in df_c_g.iterrows():
            y = int(row['Year'])
            val = float(row['Yearly burned area across all land types'])
            if 2002 <= y <= 2024:
                matrix_ha.loc[country, y] = val

        # 3. 2025-2026 VIIRS
        df_c_w = df_w[df_w['Entity'] == country]
        for y in [2025, 2026]:
            val = df_c_w[df_c_w['year'] == y]['Area burnt by wildfires'].sum()
            matrix_ha.loc[country, y] = float(val)

    # Sort matrix by Latitude (North to South)
    matrix_ha['lat'] = matrix_ha.index.map(MEDITERRANEAN_5)
    matrix_ha = matrix_ha.sort_values('lat', ascending=False)
    formatted_index = [f"{c} ({lat:.1f}°N)" for c, lat in zip(matrix_ha.index, matrix_ha['lat'])]
    matrix_ha.drop(columns=['lat'], inplace=True)
    matrix_ha.index = formatted_index

    # Calculate 90th percentile in hectares per country over 1980-2026
    p90 = matrix_ha.apply(lambda row: np.percentile(row.dropna(), 90), axis=1)

    # Heatmap binary state: 0 = White (Normal), 1 = Red (>= 90th percentile)
    heatmap_data = pd.DataFrame(0, index=matrix_ha.index, columns=matrix_ha.columns)
    annot_data = pd.DataFrame("", index=matrix_ha.index, columns=matrix_ha.columns, dtype=object)

    for idx in matrix_ha.index:
        thresh = p90[idx]
        for col in matrix_ha.columns:
            val = matrix_ha.loc[idx, col]
            if val >= thresh and val > 0:
                heatmap_data.loc[idx, col] = 1
                if val >= 1000:
                    annot_data.loc[idx, col] = f"{int(round(val/1000))}k"
                else:
                    annot_data.loc[idx, col] = f"{int(round(val))}ha"
            else:
                heatmap_data.loc[idx, col] = 0

    fig, ax = plt.subplots(figsize=(28, 7.5), dpi=300)
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

    # Highlight years with >= 2 countries exceeding 90th percentile
    records_per_year = (heatmap_data == 1).sum(axis=0)
    years_to_highlight = records_per_year[records_per_year >= 2].index

    for i, year in enumerate(heatmap_data.columns):
        if year in years_to_highlight:
            ax.add_patch(plt.Rectangle((i, 0), 1, heatmap_data.shape[0], fill=False, edgecolor='blue', lw=2.5))
            labels = ax.get_xticklabels()
            if i < len(labels):
                labels[i].set_weight("bold")
                labels[i].set_color("blue")

    ax.set_title('Superficie Forestale Bruciata nei Paesi del Mediterraneo (1980–2026) [Ettari — Annate > 90° Percentile]', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Anno (Evidenziati in Rosso gli Anni Record > 90° Percentile in Ettari)', fontsize=12, labelpad=10)
    ax.set_ylabel('Paese Mediterraneo (Da Nord a Sud)', fontsize=12, labelpad=10)

    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = 'docs/record_heatmap_wildfires_mediterranean.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mediterranean record heatmap saved to {output_path}")

    # Remove Europe plot if exists to keep repo clean
    if os.path.exists("docs/record_heatmap_wildfires_europe.png"):
        os.remove("docs/record_heatmap_wildfires_europe.png")

if __name__ == "__main__":
    main()
