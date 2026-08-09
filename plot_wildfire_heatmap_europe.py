import requests
import io
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# EFFIS official historical annual burnt area (1961-2001) in hectares
HIST_SOUTHERN = {
    'Spain': {
        1961: 32000, 1962: 18000, 1963: 15000, 1964: 22000, 1965: 35000,
        1966: 41000, 1967: 78000, 1968: 92000, 1969: 108000, 1970: 125000,
        1971: 72000, 1972: 68000, 1973: 135000, 1974: 228000, 1975: 241000,
        1976: 152000, 1977: 114000, 1978: 435000, 1979: 264000, 1980: 256000,
        1981: 289000, 1982: 152000, 1983: 125000, 1984: 165000, 1985: 484000,
        1986: 262000, 1987: 145000, 1988: 138000, 1989: 426000, 1990: 203000,
        1991: 260000, 1992: 105000, 1993: 89000,  1994: 437000, 1995: 143000,
        1996: 60000,  1997: 98000,  1998: 133000, 1999: 82000,  2000: 188000, 2001: 93000
    },
    'Italy': {
        1961: 35000, 1962: 42000, 1963: 28000, 1964: 31000, 1965: 25000,
        1966: 39000, 1967: 62000, 1968: 71000, 1969: 84000, 1970: 69000,
        1971: 112000, 1972: 78000, 1973: 95000, 1974: 124000, 1975: 118000,
        1976: 89000, 1977: 92000, 1978: 142000, 1979: 168000, 1980: 144200,
        1981: 265000, 1982: 130000, 1983: 212600, 1984: 75000,  1985: 190600,
        1986: 119000, 1987: 120600, 1988: 186400, 1989: 95000,  1990: 195300,
        1991: 99800,  1992: 105400, 1993: 203700, 1994: 116400, 1995: 47800,
        1996: 58200,  1997: 111400, 1998: 155500, 1999: 71100,  2000: 114600, 2001: 76400
    },
    'Greece': {
        1965: 18000, 1966: 15000, 1967: 22000, 1968: 31000, 1969: 28000,
        1970: 19000, 1971: 25000, 1972: 21000, 1973: 34000, 1974: 52000,
        1975: 41000, 1976: 38000, 1977: 74000, 1978: 51000, 1979: 45000,
        1980: 33000, 1981: 81000, 1982: 24000, 1983: 19000, 1984: 31000,
        1985: 105000, 1986: 24000, 1987: 46000, 1988: 110000, 1989: 41000,
        1990: 39000, 1991: 23000, 1992: 71000, 1993: 54000, 1994: 58000,
        1995: 27000, 1996: 25000, 1997: 52000, 1998: 93000, 1999: 83000,
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
        1973: 28000, 1974: 31000, 1975: 35000, 1976: 88000, 1977: 24000,
        1978: 52000, 1979: 78000, 1980: 22000, 1981: 27000, 1982: 54000,
        1983: 53000, 1984: 27000, 1985: 49000, 1986: 52000, 1987: 20000,
        1988: 13000, 1989: 75000, 1990: 76000, 1991: 10000, 1992: 17000,
        1993: 26000, 1994: 25000, 1995: 18000, 1996: 11000, 1997: 22000,
        1998: 18000, 1999: 16000, 2000: 24000, 2001: 21000
    }
}

# European countries sorted by Latitude (North to South)
countries_lat = {
    'Iceland': 64.9, 'Finland': 61.9, 'Norway': 60.4, 'Sweden': 60.1, 'Estonia': 58.6,
    'Latvia': 56.9, 'Lithuania': 55.2, 'Denmark': 56.3, 'Ireland': 53.4,
    'United Kingdom': 55.3, 'Belarus': 53.7, 'Netherlands': 52.1,
    'Poland': 51.9, 'Germany': 51.2, 'Belgium': 50.5, 'Czechia': 49.8, 'Luxembourg': 49.8,
    'Slovakia': 48.7, 'Ukraine': 48.3, 'Austria': 47.5, 'Moldova': 47.4, 'Hungary': 47.2,
    'Switzerland': 46.8, 'France': 46.2, 'Slovenia': 46.1, 'Croatia': 45.1, 'Romania': 45.9,
    'Bosnia and Herzegovina': 43.9, 'Serbia': 44.0, 'Bulgaria': 42.7,
    'Montenegro': 42.7, 'North Macedonia': 41.6, 'Albania': 41.1,
    'Italy': 41.9, 'Spain': 40.4, 'Portugal': 39.4, 'Greece': 39.0, 'Turkey': 38.9,
    'Malta': 35.9, 'Cyprus': 35.1
}

def main():
    print("--- GENERATING WILDFIRE EXTREME HEATMAP WITH GRAY NO-DATA CELLS (1961-2026) ---")
    url_gwis = "https://ourworldindata.org/grapher/annual-area-burnt-by-wildfires-gwis.csv"
    url_weekly = "https://ourworldindata.org/grapher/weekly-area-burnt-by-wildfires.csv"

    print("Downloading GWIS and OWID datasets...")
    r_g = requests.get(url_gwis, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    df_g = pd.read_csv(io.StringIO(r_g.text))

    r_w = requests.get(url_weekly, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    df_w = pd.read_csv(io.StringIO(r_w.text))
    df_w['Day'] = pd.to_datetime(df_w['Day'])
    df_w['year'] = df_w['Day'].dt.year

    years = list(range(1961, 2027))
    matrix = pd.DataFrame(np.nan, index=list(countries_lat.keys()), columns=years, dtype=float)

    for country in countries_lat.keys():
        # 1. 1961-2001 EFFIS / National Forestry Archives
        if country in HIST_SOUTHERN:
            for y, val in HIST_SOUTHERN[country].items():
                matrix.loc[country, y] = float(val)
                
        # 2. 2002-2024 GWIS MODIS
        df_c_g = df_g[df_g['Entity'] == country]
        for _, row in df_c_g.iterrows():
            y = int(row['Year'])
            val = float(row['Yearly burned area across all land types'])
            if 2002 <= y <= 2024:
                matrix.loc[country, y] = val

        # 3. 2025-2026 VIIRS
        df_c_w = df_w[df_w['Entity'] == country]
        for y in [2025, 2026]:
            val = df_c_w[df_c_w['year'] == y]['Area burnt by wildfires'].sum()
            if val > 0 or y in df_c_w['year'].values:
                matrix.loc[country, y] = float(val)

    # Sort matrix by Latitude (North to South)
    matrix['lat'] = matrix.index.map(countries_lat)
    matrix = matrix.sort_values('lat', ascending=False)
    
    formatted_index = [f"{c} ({lat:.1f}°N)" for c, lat in zip(matrix.index, matrix['lat'])]
    matrix = matrix.drop(columns=['lat'])
    matrix.index = formatted_index

    # Calculate 90th percentile per country over valid recorded values
    p90 = matrix.apply(lambda row: np.percentile(row.dropna(), 90) if len(row.dropna()) > 0 else 0, axis=1)

    # State encoding for Heatmap:
    # -1 = No Data (Gray)
    #  0 = Data Available, Normal Year (< 90th percentile) (White)
    #  1 = Data Available, Extreme Fire Year (>= 90th percentile) (Red/Orange)
    heatmap_data = pd.DataFrame(-1, index=matrix.index, columns=matrix.columns)
    annot_data = pd.DataFrame("", index=matrix.index, columns=matrix.columns, dtype=object)

    for idx in matrix.index:
        thresh = p90[idx]
        for col in matrix.columns:
            val = matrix.loc[idx, col]
            if pd.isna(val):
                heatmap_data.loc[idx, col] = -1
            else:
                if thresh > 0 and val >= thresh:
                    heatmap_data.loc[idx, col] = 1
                    if val >= 1000:
                        annot_data.loc[idx, col] = f"{int(round(val/1000))}k"
                    else:
                        annot_data.loc[idx, col] = f"{int(round(val))}ha"
                else:
                    heatmap_data.loc[idx, col] = 0

    fig, ax = plt.subplots(figsize=(34, 16), dpi=300)
    
    # Custom colormap matching temperature records style:
    # -1: Light Gray (#d9d9d9) -> No Data Available
    #  0: White (#ffffff)      -> Normal Year (< 90th percentile)
    #  1: Red/Orange (#ff5722) -> Extreme Fire Year (>= 90th percentile)
    cmap = mcolors.ListedColormap(['#d9d9d9', '#ffffff', '#ff5722'])

    sns.heatmap(
        heatmap_data, 
        cmap=cmap, 
        ax=ax, 
        annot=annot_data.values, 
        fmt="", 
        annot_kws={"size": 7.0, "color": "black", "weight": "bold"},
        linewidths=0.2, 
        linecolor='#b0b0b0', 
        xticklabels=True, 
        yticklabels=True,
        cbar=False,
        vmin=-1, 
        vmax=1
    )

    # Highlight years with >= 4 countries having extreme fire events
    records_per_year = (heatmap_data == 1).sum(axis=0)
    years_to_highlight = records_per_year[records_per_year >= 4].index

    for i, year in enumerate(heatmap_data.columns):
        if year in years_to_highlight:
            ax.add_patch(plt.Rectangle((i, 0), 1, heatmap_data.shape[0], fill=False, edgecolor='blue', lw=2.5))
            labels = ax.get_xticklabels()
            if i < len(labels):
                labels[i].set_weight("bold")
                labels[i].set_color("blue")

    ax.set_title('Anni Estremi per Superficie Forestale Bruciata in Europa (1961–2026) [Valori Oltre il 90° Percentile]', fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('Anno (Grigio = Dati non disponibili prima dell\'era satellitare 2002)', fontsize=14, labelpad=10)
    ax.set_ylabel('Paese Europeo (Da Nord a Sud)', fontsize=14, labelpad=10)

    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = 'docs/record_heatmap_wildfires_europe.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Wildfire 1961-2026 record heatmap with gray no-data cells saved to {output_path}")

if __name__ == "__main__":
    main()
