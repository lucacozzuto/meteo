import requests
import io
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Official EFFIS (European Forest Fire Information System) Ground Statistics (1980-2026) in hectares
EFFIS_GROUND = {
    'Spain': {
        1980: 256000, 1981: 289000, 1982: 152000, 1983: 125000, 1984: 165000,
        1985: 484000, 1986: 262000, 1987: 145000, 1988: 138000, 1989: 426000,
        1990: 203000, 1991: 260000, 1992: 105000, 1993: 89000,  1994: 437000,
        1995: 143000, 1996: 60000,  1997: 98000,  1998: 133000, 1999: 82000,
        2000: 188000, 2001: 93000,  2002: 107476, 2003: 148121, 2004: 134195,
        2005: 188697, 2006: 155362, 2007: 86180,  2008: 50304,  2009: 110800,
        2010: 54770,  2011: 102161, 2012: 216894, 2013: 58984,  2014: 48163,
        2015: 103200, 2016: 65817,  2017: 178234, 2018: 25164,  2019: 83963,
        2020: 65906,  2021: 87925,  2022: 310378, 2023: 89068,  2024: 42000,
        2025: 98000,  2026: 22000
    },
    'Italy': {
        1980: 144200, 1981: 265000, 1982: 130000, 1983: 212600, 1984: 75000,
        1985: 190600, 1986: 119000, 1987: 120600, 1988: 186400, 1989: 95000,
        1990: 195300, 1991: 99800,  1992: 105400, 1993: 203700, 1994: 116400,
        1995: 47800,  1996: 58200,  1997: 111400, 1998: 155500, 1999: 71100,
        2000: 114600, 2001: 76400,  2002: 46686,  2003: 91804,  2004: 60078,
        2005: 47575,  2006: 39946,  2007: 227729, 2008: 66327,  2009: 73355,
        2010: 46526,  2011: 72036,  2012: 130814, 2013: 29076,  2014: 36780,
        2015: 41506,  2016: 27402,  2017: 140445, 2018: 19441,  2019: 36000,
        2020: 56000,  2021: 159000, 2022: 68500,  2023: 61000,  2024: 45000,
        2025: 72000,  2026: 18000
    },
    'Greece': {
        1980: 33000,  1981: 81000,  1982: 24000,  1983: 19000,  1984: 31000,
        1985: 105000, 1986: 24000,  1987: 46000,  1988: 110000, 1989: 41000,
        1990: 39000,  1991: 23000,  1992: 71000,  1993: 54000,  1994: 58000,
        1995: 27000,  1996: 25000,  1997: 52000,  1998: 93000,  1999: 83000,
        2000: 145000, 2001: 28000,  2002: 6000,   2003: 3500,   2004: 10000,
        2005: 6400,   2006: 12600,  2007: 270000, 2008: 30000,  2009: 35000,
        2010: 8900,   2011: 29000,  2012: 59000,  2013: 25000,  2014: 20000,
        2015: 13000,  2016: 32000,  2017: 22000,  2018: 12000,  2019: 10000,
        2020: 15000,  2021: 130000, 2022: 22000,  2023: 175000, 2024: 40000,
        2025: 55000,  2026: 12000
    },
    'Portugal': {
        1980: 44000,  1981: 89000,  1982: 39000,  1983: 47000,  1984: 53000,
        1985: 146000, 1986: 98000,  1987: 76000,  1988: 22000,  1989: 126000,
        1990: 137000, 1991: 182000, 1992: 57000,  1993: 49000,  1994: 77000,
        1995: 169000, 1996: 88000,  1997: 30000,  1998: 158000, 1999: 70000,
        2000: 159000, 2001: 111000, 2002: 124411, 2003: 425726, 2004: 129539,
        2005: 338262, 2006: 75510,  2007: 31450,  2008: 17270,  2009: 87416,
        2010: 133083, 2011: 73813,  2012: 110231, 2013: 152756, 2014: 19929,
        2015: 64443,  2016: 161522, 2017: 539921, 2018: 44578,  2019: 42000,
        2020: 67000,  2021: 28000,  2022: 110000, 2023: 35000,  2024: 140000,
        2025: 145000, 2026: 25000
    },
    'France': {
        1980: 22000,  1981: 27000,  1982: 54000,  1983: 53000,  1984: 27000,
        1985: 49000,  1986: 52000,  1987: 20000,  1988: 13000,  1989: 75000,
        1990: 76000,  1991: 10000,  1992: 17000,  1993: 26000,  1994: 25000,
        1995: 18000,  1996: 11000,  1997: 22000,  1998: 18000,  1999: 16000,
        2000: 24000,  2001: 21000,  2002: 30168,  2003: 73278,  2004: 13700,
        2005: 22100,  2006: 8500,   2007: 8600,   2008: 6000,   2009: 17000,
        2010: 10000,  2011: 9000,   2012: 8500,   2013: 4000,   2014: 8000,
        2015: 11000,  2016: 14000,  2017: 26000,  2018: 5000,   2019: 23000,
        2020: 14000,  2021: 15000,  2022: 66000,  2023: 13000,  2024: 12000,
        2025: 32000,  2026: 8000
    }
}

MEDITERRANEAN_5 = {
    'France': 46.2, 'Italy': 41.9, 'Spain': 40.4, 'Portugal': 39.4, 'Greece': 39.0
}

CITY_MAP = {
    'Spain': 'data/Madrid.csv',
    'Italy': 'data/Rome.csv',
    'Greece': 'data/Athens.csv',
    'Portugal': 'data/Lisbon.csv',
    'France': 'data/Paris.csv'
}

def get_country_heatwave_flags():
    country_hw_matrix = {}
    for country, fpath in CITY_MAP.items():
        hw_set = set()
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            df = df[df['date'].dt.year >= 1980].copy()
            df = df[df['date'].dt.month.isin([6, 7, 8])].copy()
            
            df_base = df[(df['date'].dt.year >= 1991) & (df['date'].dt.year <= 2020)]['temperature_2m_max'].dropna()
            thresh = df_base.quantile(0.90) if not df_base.empty else 30
            
            df['is_hot'] = df['temperature_2m_max'] > thresh
            
            current_hw = []
            for i, row in df.iterrows():
                if row['is_hot']:
                    if len(current_hw) > 0 and (row['date'] - current_hw[-1]['date']).days > 1:
                        if len(current_hw) >= 6:
                            for r in current_hw:
                                hw_set.add(r['date'].year)
                        current_hw = []
                    current_hw.append(row)
                else:
                    if len(current_hw) >= 6:
                        for r in current_hw:
                            hw_set.add(r['date'].year)
                    current_hw = []
            if len(current_hw) >= 6:
                for r in current_hw:
                    hw_set.add(r['date'].year)
                    
        country_hw_matrix[country] = hw_set
    return country_hw_matrix

def generate_wildfire_heatmap_with_cell_asterisks(matrix_ha, title, subtitle, output_path, figsize=(30, 8.5)):
    hw_matrix = get_country_heatwave_flags()
    
    # Calculate 90th percentile for individual countries
    p90_countries = matrix_ha.iloc[:-1].apply(lambda row: np.percentile(row.dropna(), 90), axis=1)
    p90_total = np.percentile(matrix_ha.loc['TOTALE'].dropna(), 90)

    heatmap_data = pd.DataFrame(0, index=matrix_ha.index, columns=matrix_ha.columns)
    annot_data = pd.DataFrame("", index=matrix_ha.index, columns=matrix_ha.columns, dtype=object)

    # 1. Country rows (90th percentile threshold + cell asterisk for heatwave years)
    for idx in matrix_ha.index[:-1]:
        c_name = idx.split(' (')[0]
        thresh = p90_countries[idx]
        hw_years = hw_matrix.get(c_name, set())
        
        for col in matrix_ha.columns:
            val = matrix_ha.loc[idx, col]
            has_hw = (col in hw_years)
            ast = "*" if has_hw else ""
            
            if val >= thresh and val > 0:
                heatmap_data.loc[idx, col] = 1
                if val >= 1000000:
                    val_str = f"{val/1000000:.2f}M"
                elif val >= 1000:
                    val_str = f"{int(round(val/1000))}k"
                else:
                    val_str = f"{int(round(val))}ha"
                annot_data.loc[idx, col] = f"{val_str}{ast}"
            else:
                heatmap_data.loc[idx, col] = 0
                annot_data.loc[idx, col] = ast

    # 2. TOTALE row (90th percentile threshold + asterisk if any country had heatwave)
    for col in matrix_ha.columns:
        val = matrix_ha.loc['TOTALE', col]
        any_hw = any(col in hw_matrix[c] for c in hw_matrix)
        ast = "*" if any_hw else ""
        
        if val >= p90_total:
            heatmap_data.loc['TOTALE', col] = 1
            val_str = f"{val/1000000:.2f}M" if val >= 1000000 else f"{int(round(val/1000))}k"
            annot_data.loc['TOTALE', col] = f"{val_str}{ast}"
        else:
            heatmap_data.loc['TOTALE', col] = 0
            annot_data.loc['TOTALE', col] = ast

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # 0 = White (Normal), 1 = Red/Orange (>= 90th percentile threshold)
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

    # Draw horizontal separating line above TOTALE row
    n_rows = matrix_ha.shape[0]
    ax.axhline(n_rows - 1, color='cyan', linewidth=3)

    # Highlight COLUMNS (years) ONLY when TOTALE is >= 90th percentile
    column_highlight = [col for col in matrix_ha.columns if matrix_ha.loc['TOTALE', col] >= p90_total]

    # Clean X-axis tick labels without asterisk on year labels
    ax.set_xticklabels(matrix_ha.columns, rotation=45, ha='right')

    for i, year in enumerate(matrix_ha.columns):
        if year in column_highlight:
            ax.add_patch(plt.Rectangle((i, 0), 1, matrix_ha.shape[0], fill=False, edgecolor='blue', lw=3.0))
            labels = ax.get_xticklabels()
            if i < len(labels):
                labels[i].set_weight("bold")
                labels[i].set_color("blue")

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(subtitle, fontsize=11, labelpad=10)
    ax.set_ylabel('Paese Mediterraneo (Da Nord a Sud)', fontsize=12, labelpad=10)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Wildfire heatmap with cell-level heatwave asterisks saved to {output_path}")

def main():
    print("--- GENERATING WILDFIRE HEATMAP WITH CELL-LEVEL HEATWAVE ASTERISKS (1980-2026) ---")
    years_1980 = list(range(1980, 2027))
    matrix_med = pd.DataFrame(index=list(MEDITERRANEAN_5.keys()), columns=years_1980, dtype=float).fillna(0.0)

    for country in MEDITERRANEAN_5.keys():
        if country in EFFIS_GROUND:
            for y, val in EFFIS_GROUND[country].items():
                matrix_med.loc[country, y] = float(val)

    matrix_med['lat'] = matrix_med.index.map(MEDITERRANEAN_5)
    matrix_med.sort_values('lat', ascending=False, inplace=True)
    formatted_index = [f"{c} ({lat:.1f}°N)" for c, lat in zip(matrix_med.index, matrix_med['lat'])]
    matrix_med.drop(columns=['lat'], inplace=True)
    matrix_med.index = formatted_index

    # Add TOTAL Row at the bottom
    total_row = matrix_med.sum(axis=0)
    matrix_med.loc['TOTALE'] = total_row

    generate_wildfire_heatmap_with_cell_asterisks(
        matrix_ha=matrix_med,
        title='Superficie Forestale Bruciata nel Mediterraneo (1980–2026) [* = Anno con Ondata di Calore nel Paese]',
        subtitle='Anno (Evidenziate in Rosso le Annate Incendio > 90° Percentile — L\'asterisco * indica la presenza di un\'Ondata di Calore Estiva in quel Paese/Anno)',
        output_path='docs/record_heatmap_wildfires_mediterranean.png',
        figsize=(30, 8.5)
    )

if __name__ == "__main__":
    main()
