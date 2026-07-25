import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set overall style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def load_and_process_data(data_dir="data_italy"):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Trovati {len(csv_files)} file CSV in {data_dir}")
    
    all_dfs = []
    for file in csv_files:
        city = os.path.basename(file).replace('.csv', '')
        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        df['city'] = city
        # Calcolo temperatura media giornaliera
        df['mean_temp'] = (df['temperature_2m_max'] + df['temperature_2m_min']) / 2.0
        # Giorno del mese/anno per climatologia (es. 07-15)
        df['month_day'] = df['date'].dt.strftime('%m-%d')
        df['year'] = df['date'].dt.year
        
        all_dfs.append(df[['date', 'year', 'month_day', 'city', 'mean_temp', 'temperature_2m_max', 'temperature_2m_min']])
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Rimuoviamo anni incompleti (come il 2026) per evitare bias stagionali nelle distribuzioni annuali
    max_year = full_df['year'].max()
    # Controlliamo se max_year ha 365 giorni per almeno una città
    days_in_last_year = full_df[full_df['year'] == max_year]['date'].dt.dayofyear.max()
    if days_in_last_year < 360:
        print(f"Anno {max_year} incompleto (giorni max: {days_in_last_year}), esclusione dall'analisi distributiva annuale.")
        full_df = full_df[full_df['year'] < max_year]
        
    print("Calcolo climatologia di riferimento (media storica 1940-2025 per giorno di calendario)...")
    # Calcolo della media storica per città e per giorno del calendario
    climatology = full_df.groupby(['city', 'month_day'])['mean_temp'].transform('mean')
    full_df['anomaly'] = full_df['mean_temp'] - climatology
    
    return full_df

def analyze_distributions(df):
    years = sorted(df['year'].unique())
    print(f"Analisi statistica delle distribuzioni per {len(years)} anni ({years[0]}-{years[-1]})...")
    
    # Baseline distribution per Wasserstein (primi 30 anni: 1940-1969)
    baseline_anomalies = df[df['year'].isin(range(1940, 1970))]['anomaly'].dropna().values
    
    stats_list = []
    kde_grid = np.linspace(-12.0, 12.0, 200)
    kde_curves = {}
    
    for y in years:
        y_anom = df[df['year'] == y]['anomaly'].dropna().values
        if len(y_anom) < 100:
            continue
            
        mean_val = float(np.mean(y_anom))
        std_val = float(np.std(y_anom))
        p90_val = float(np.percentile(y_anom, 90))
        p95_val = float(np.percentile(y_anom, 95))
        skew_val = float(stats.skew(y_anom))
        kurt_val = float(stats.kurtosis(y_anom)) # excess kurtosis (0 per Gaussiana)
        
        # Test di normalità di D'Agostino K^2
        stat_k2, pval_k2 = stats.normaltest(y_anom)
        
        # Distanza di Wasserstein (Earth Mover's Distance)
        w_dist = float(stats.wasserstein_distance(y_anom, baseline_anomalies))
        
        # Calcolo KDE per il grafico interattivo / sovrapposto
        kde = stats.gaussian_kde(y_anom, bw_method=0.25)
        kde_density = kde(kde_grid)
        
        stats_list.append({
            'year': int(y),
            'mean': round(mean_val, 3),
            'std': round(std_val, 3),
            'p90': round(p90_val, 3),
            'p95': round(p95_val, 3),
            'skewness': round(skew_val, 3),
            'kurtosis': round(kurt_val, 3),
            'k2_stat': round(float(stat_k2), 2),
            'k2_pval': float(pval_k2),
            'wasserstein': round(w_dist, 3)
        })
        
        kde_curves[int(y)] = [round(float(v), 4) for v in kde_density]
        
    stats_df = pd.DataFrame(stats_list)
    return stats_df, kde_grid, kde_curves

def generate_superimposed_plot(stats_df, kde_grid, kde_curves, out_path="docs/daily_anomalies_superimposed_italy.png"):
    print("Generazione grafico KDE sovrapposte...")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    years = stats_df['year'].values
    min_y, max_y = years[0], years[-1]
    
    # Colormap cronologica da blu (1940) a rosso/magenta (2025)
    cmap = plt.cm.coolwarm
    
    for y in years:
        norm_y = (y - min_y) / (max_y - min_y)
        color = cmap(norm_y)
        alpha = 0.35 if y < 2000 else 0.7
        lw = 0.8 if y < 2000 else 1.5
        
        # Evidenziamo il primo e l'ultimo anno e alcuni decenni
        if y in [1940, 1970, 2000, max_y]:
            lw = 2.5
            alpha = 1.0
            label = f"Anno {y}"
        else:
            label = None
            
        ax.plot(kde_grid, kde_curves[y], color=color, alpha=alpha, linewidth=lw, label=label)
        
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7, label="Zero Anomalia (Normale Storica)")
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_y, vmax=max_y))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('Anno di Osservazione', fontsize=12, fontweight='bold', labelpad=10)
    
    ax.set_title("Evoluzione delle Distribuzioni delle Anomalie Termiche Giornaliere in Italia (1940-2025)", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Anomalia Termica Giornaliera (°C rispetto a media storica di quel giorno)", fontsize=13, labelpad=10)
    ax.set_ylabel("Densità di Probabilità", fontsize=13, labelpad=10)
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 0.22)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Salvato {out_path}")

def generate_ridgeline_plot(df, out_path="docs/daily_anomalies_ridgeline_italy.png"):
    print("Generazione grafico Ridgeline per decenni...")
    # Add decade
    df['decade'] = (df['year'] // 10) * 10
    decades = sorted(df['decade'].unique())
    
    # Creiamo un plot ridgeline con seaborn FacetGrid
    pal = sns.color_palette("coolwarm", len(decades))
    
    g = sns.FacetGrid(df, row="decade", hue="decade", aspect=12, height=0.75, palette=pal)
    
    # Disegnamo le curve di densità
    g.map(sns.kdeplot, "anomaly", bw_adjust=0.8, clip_on=False, fill=True, alpha=0.8, linewidth=1.5)
    g.map(sns.kdeplot, "anomaly", clip_on=False, color="w", lw=2, bw_adjust=0.8)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    
    # Aggiungiamo etichette dei decenni
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0.02, 0.2, f"Anni '{str(label)[2:]}0", fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes, fontsize=13)
        # Linea dello zero
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
    g.map(label, "anomaly")
    
    # Rimuoviamo assi sovrapposti e miglioriamo estetica
    g.figure.subplots_adjust(hspace=-0.4)
    g.set_titles("")
    g.set(yticks=[], xlabel="Anomalia Termica Giornaliera (°C)", xlim=(-10, 10))
    g.despine(bottom=True, left=True)
    
    g.figure.suptitle("Distribuzione Decennale delle Anomalie Termiche Giornaliere (Ridgeline Plot)", fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Salvato {out_path}")

def generate_statistics_plot(stats_df, out_path="docs/daily_anomalies_statistics_italy.png"):
    print("Generazione grafico metriche statistiche di deviazione...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300)
    
    years = stats_df['year']
    
    # 1. Media e 90° Percentile
    ax = axes[0, 0]
    ax.plot(years, stats_df['mean'], color='#d62728', linewidth=2, label="Media Anomalia (°C)")
    ax.plot(years, stats_df['p90'], color='#ff7f0e', linewidth=1.5, linestyle='--', label="90° Percentile Estremo (°C)")
    ax.axhline(0, color='black', linestyle=':', alpha=0.7)
    ax.set_title("1. Spostamento del Centro e degli Estremi Caldi", fontsize=13, fontweight='bold')
    ax.set_ylabel("Anomalia (°C)")
    ax.legend(loc='upper left', frameon=True)
    
    # 2. Asimmetria (Skewness)
    ax = axes[0, 1]
    ax.plot(years, stats_df['skewness'], color='#2ca02c', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.7, label="Zero (Simmetria Normale perfetta)")
    # Trendline
    z = np.polyfit(years, stats_df['skewness'], 1)
    p = np.poly1d(z)
    ax.plot(years, p(years), color="darkgreen", linestyle=":", linewidth=2, label="Trend Asimmetria")
    ax.set_title("2. Asimmetria (Skewness) della Distribuzione", fontsize=13, fontweight='bold')
    ax.set_ylabel("Skewness (>0: coda calda allungata)")
    ax.legend(loc='upper left', frameon=True)
    
    # 3. Curtosi (Excess Kurtosis)
    ax = axes[1, 0]
    ax.plot(years, stats_df['kurtosis'], color='#9467bd', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.7, label="Zero (Curtosi Gaussiana)")
    z = np.polyfit(years, stats_df['kurtosis'], 1)
    p = np.poly1d(z)
    ax.plot(years, p(years), color="indigo", linestyle=":", linewidth=2, label="Trend Curtosi")
    ax.set_title("3. Curtosi in Eccesso (Peso delle Code / Estremi)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Kurtosis (>0: code più pesanti del Normale)")
    ax.legend(loc='upper left', frameon=True)
    
    # 4. Distanza di Wasserstein (Shift totale vs 1940-1969)
    ax = axes[1, 1]
    ax.plot(years, stats_df['wasserstein'], color='#e377c2', linewidth=2.5, label="Distanza di Wasserstein")
    ax.set_title("4. Distanza di Wasserstein (Deformazione Totale dal 1940-1969)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Distanza Geometrica (Earth Mover)")
    ax.legend(loc='upper left', frameon=True)
    
    for a in axes.flat:
        a.set_xlabel("Anno")
        a.grid(True, linestyle='--', alpha=0.5)
        
    plt.suptitle("Analisi Statistica di Deviazione dalla Normalità Storica e Gaussiana (Italia 1940-2025)", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Salvato {out_path}")

def export_json(stats_df, kde_grid, kde_curves, out_path="docs/daily_anomalies.json"):
    print("Esportazione dati per il web...")
    data = {
        'grid': [round(float(x), 3) for x in kde_grid],
        'years': stats_df['year'].tolist(),
        'stats': stats_df.to_dict(orient='records'),
        'curves': kde_curves
    }
    with open(out_path, 'w') as f:
        json.dump(data, f)
    print(f"Salvato {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")

if __name__ == '__main__':
    df = load_and_process_data()
    stats_df, kde_grid, kde_curves = analyze_distributions(df)
    
    generate_superimposed_plot(stats_df, kde_grid, kde_curves)
    generate_ridgeline_plot(df)
    generate_statistics_plot(stats_df)
    export_json(stats_df, kde_grid, kde_curves)
    print("Elaborazione e analisi terminate con successo!")
