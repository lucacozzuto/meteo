import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.stats.diagnostic as smd

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
        
        # Test di normalità di Kolmogorov-Smirnov (con correzione di Lilliefors) sulle medie giornaliere nazionali di quell'anno (N=365)
        df_daily_y = df[df['year'] == y].groupby('date')['anomaly'].mean().dropna().values
        stat_lilliefors, pval_lilliefors = smd.lilliefors(df_daily_y, dist='norm', pvalmethod='table')
        is_normal = bool(pval_lilliefors >= 0.001)
        
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
            'shapiro_stat': round(float(stat_lilliefors), 4),
            'shapiro_pval': float(pval_lilliefors),
            'is_normal_shapiro': is_normal,
            'wasserstein': round(w_dist, 3)
        })
        
        kde_curves[int(y)] = [round(float(v), 4) for v in kde_density]
        
    stats_df = pd.DataFrame(stats_list)
    return stats_df, kde_grid, kde_curves

def generate_superimposed_plot(stats_df, kde_grid, kde_curves, out_path="docs/daily_anomalies_superimposed_italy.png"):
    print("Generazione grafico KDE sovrapposte (Grigio = Normali, Rosso = Non Normali Lilliefors KS 0.001)...")
    fig, ax = plt.subplots(figsize=(13, 8), dpi=300)
    
    years = stats_df['year'].values
    
    # Per evitare sovrapposizioni eccessive di testo sui picchi, alterniamo leggermente le etichette
    normal_count = 0
    non_normal_count = 0
    
    for y in years:
        is_normal = stats_df[stats_df['year'] == y]['is_normal_shapiro'].values[0]
        curve = kde_curves[y]
        
        if is_normal:
            color = '#94a3b8'  # Grigio / Argento
            alpha = 0.35
            lw = 0.8
            ax.plot(kde_grid, curve, color=color, alpha=alpha, linewidth=lw)
            normal_count += 1
        else:
            color = '#ef4444'  # Rosso acceso
            alpha = 0.75
            lw = 1.6
            ax.plot(kde_grid, curve, color=color, alpha=alpha, linewidth=lw)
            non_normal_count += 1
            
            # Troviamo il picco della curva per posizionare l'etichetta dell'anno
            peak_idx = np.argmax(curve)
            peak_x = kde_grid[peak_idx]
            peak_y = curve[peak_idx]
            
            ax.text(peak_x, peak_y + 0.002, str(y), fontsize=7.5, color='#991b1b', 
                    fontweight='bold', ha='center', va='bottom', alpha=0.9)
        
    ax.axvline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7, label="Zero Anomalia (Normale Storica)")
    
    # Voci legenda
    ax.plot([], [], color='#94a3b8', linewidth=1.5, alpha=0.8, label=f"Distribuzione Normale ({normal_count} anni, p ≥ 0.001)")
    ax.plot([], [], color='#ef4444', linewidth=2.0, alpha=0.9, label=f"Non Normale (Lilliefors KS, {non_normal_count} anni, p < 0.001)")
    
    ax.set_title("Evoluzione delle Distribuzioni Termiche (Italia 1940-2025): Test di Lilliefors (KS)", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Anomalia Termica Giornaliera (°C rispetto a media storica di quel giorno)", fontsize=13, labelpad=10)
    ax.set_ylabel("Densità di Probabilità", fontsize=13, labelpad=10)
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 0.23)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=11)
    
    plt.tight_layout()
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

def generate_ventenni_plots(df, kde_grid, out_dir="docs"):
    print("Generazione grafici per ventenni (Confronto aggregato, Griglia 5 pannelli, e Singoli ventenni)...")
    ventenni = [
        (1940, 1959, "1940-1959", "#1f77b4"),  # Blu profondo
        (1960, 1979, "1960-1979", "#2ca02c"),  # Verde
        (1980, 1999, "1980-1999", "#ff7f0e"),  # Arancione
        (2000, 2019, "2000-2019", "#d62728"),  # Rosso scuro
        (2020, 2025, "2020-2025 (Attuale)", "#9467bd")   # Viola
    ]
    
    # 1. CONFRONTO DIRETTO CURVE AGGREGATE DEI VENTENNI
    fig, ax = plt.subplots(figsize=(13, 8), dpi=300)
    for start_y, end_y, label, color in ventenni:
        v_anom = df[(df['year'] >= start_y) & (df['year'] <= end_y)]['anomaly'].dropna().values
        if len(v_anom) == 0: continue
        kde = stats.gaussian_kde(v_anom, bw_method=0.25)
        density = kde(kde_grid)
        mean_val = np.mean(v_anom)
        ax.plot(kde_grid, density, label=f"Ventennio {label} (Media: {mean_val:+.2f}°C)", color=color, linewidth=2.8)
        ax.axvline(mean_val, color=color, linestyle=':', alpha=0.8, linewidth=1.5)
        
    ax.axvline(0, color='black', linestyle='--', label="Zero Anomalia (Normale Storica)", alpha=0.7, linewidth=1.5)
    ax.set_title("Confronto delle Distribuzioni Termiche per Ventennio (Italia 1940-2025)", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Anomalia Termica Giornaliera (°C rispetto a media storica di quel giorno)", fontsize=13, labelpad=10)
    ax.set_ylabel("Densità di Probabilità Aggregata", fontsize=13, labelpad=10)
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 0.23)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=11)
    plt.tight_layout()
    comp_path = os.path.join(out_dir, "daily_anomalies_ventenni_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Salvato {comp_path}")
    
    # 2. GRIGLIA ESPLORATIVA DEI 5 VENTENNI (3x2)
    fig, axes = plt.subplots(3, 2, figsize=(16, 15), dpi=300)
    axes_flat = axes.flat
    base_anom = df[(df['year'] >= 1940) & (df['year'] <= 1959)]['anomaly'].dropna().values
    base_kde = stats.gaussian_kde(base_anom, bw_method=0.25)(kde_grid)
    
    for idx, (start_y, end_y, label, color) in enumerate(ventenni):
        ax = axes_flat[idx]
        ax.fill_between(kde_grid, 0, base_kde, color='#cbd5e1', alpha=0.4, label="Baseline 1940-1959")
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        
        v_df = df[(df['year'] >= start_y) & (df['year'] <= end_y)]
        v_years = sorted(v_df['year'].unique())
        norm_c = 0
        non_norm_c = 0
        for y in v_years:
            y_anom = v_df[v_df['year'] == y]['anomaly'].dropna().values
            if len(y_anom) < 100: continue
            
            df_daily_y = v_df[v_df['year'] == y].groupby('date')['anomaly'].mean().dropna().values
            _, pval = smd.lilliefors(df_daily_y, dist='norm', pvalmethod='table')
            is_norm = bool(pval >= 0.001)
            
            curve = stats.gaussian_kde(y_anom, bw_method=0.25)(kde_grid)
            if is_norm:
                ax.plot(kde_grid, curve, color='#64748b', alpha=0.5, linewidth=1.0)
                norm_c += 1
            else:
                ax.plot(kde_grid, curve, color='#ef4444', alpha=0.85, linewidth=1.8)
                non_norm_c += 1
                peak_idx = np.argmax(curve)
                ax.text(kde_grid[peak_idx], curve[peak_idx] + 0.005, str(y), fontsize=8, color='#991b1b', fontweight='bold', ha='center', va='bottom')
                
        ax.set_title(f"Ventennio {label} (Normali: {norm_c} | Non Normali: {non_norm_c})", fontsize=13, fontweight='bold', color=color)
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 0.25)
        ax.grid(True, linestyle=':', alpha=0.6)
        if idx in [3, 4]: ax.set_xlabel("Anomalia Termica Giornaliera (°C)")
        if idx in [0, 2, 4]: ax.set_ylabel("Densità di Probabilità")
        
    axes_flat[5].axis('off')
    axes_flat[5].text(0.08, 0.5, "Legenda & Sintesi Ventenni:\n\n"
                                 "• Sfondo Grigio: Baseline Storica (1940-1959)\n"
                                 "• Linee Grigie: Anni Normali (Lilliefors KS p ≥ 0.001)\n"
                                 "• Linee Rosse: Anni Non Normali (Lilliefors KS p < 0.001)\n\n"
                                 "Nota Estremi: Si osserva un progressivo e drastico\n"
                                 "spostamento della campana verso destra (calore)\n"
                                 "e l'esplosione di anni non normali dal 2000 in poi.", 
                      fontsize=12.5, bbox=dict(facecolor='#f8fafc', edgecolor='#cbd5e1', boxstyle='round,pad=1.2'))
    
    plt.suptitle("Evoluzione delle Distribuzioni Termiche Italiane per Ventenni (1940-2025)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    grid_path = os.path.join(out_dir, "daily_anomalies_ventenni_grid.png")
    plt.savefig(grid_path, dpi=300)
    plt.close()
    print(f"Salvato {grid_path}")
    
    # 3. GRAFICI SINGOLI PER CIASCUN VENTENNIO
    for start_y, end_y, label, color in ventenni:
        fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
        v_df = df[(df['year'] >= start_y) & (df['year'] <= end_y)]
        v_years = sorted(v_df['year'].unique())
        norm_c = 0
        non_norm_c = 0
        for y in v_years:
            y_anom = v_df[v_df['year'] == y]['anomaly'].dropna().values
            if len(y_anom) < 100: continue
            df_daily_y = v_df[v_df['year'] == y].groupby('date')['anomaly'].mean().dropna().values
            _, pval = smd.lilliefors(df_daily_y, dist='norm', pvalmethod='table')
            is_norm = bool(pval >= 0.001)
            curve = stats.gaussian_kde(y_anom, bw_method=0.25)(kde_grid)
            if is_norm:
                ax.plot(kde_grid, curve, color='#94a3b8', alpha=0.5, linewidth=1.2)
                norm_c += 1
            else:
                ax.plot(kde_grid, curve, color='#ef4444', alpha=0.9, linewidth=2.0)
                non_norm_c += 1
                peak_idx = np.argmax(curve)
                ax.text(kde_grid[peak_idx], curve[peak_idx] + 0.003, str(y), fontsize=9, color='#991b1b', fontweight='bold', ha='center', va='bottom')
                
        ax.axvline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7, label="Zero Anomalia (Normale Storica)")
        ax.plot([], [], color='#94a3b8', linewidth=1.5, alpha=0.8, label=f"Anni Normali ({norm_c} anni, p ≥ 0.001)")
        ax.plot([], [], color='#ef4444', linewidth=2.0, alpha=0.9, label=f"Anni Non Normali ({non_norm_c} anni, p < 0.001)")
        
        ax.set_title(f"Distribuzioni Termiche Giornaliere: Ventennio {label}", fontsize=15, fontweight='bold', pad=15)
        ax.set_xlabel("Anomalia Termica Giornaliera (°C)", fontsize=13)
        ax.set_ylabel("Densità di Probabilità", fontsize=13)
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 0.25)
        ax.legend(loc='upper right', frameon=True, fontsize=11)
        plt.tight_layout()
        fname = f"daily_anomalies_ventennio_{start_y}_{end_y}.png"
        fpath = os.path.join(out_dir, fname)
        plt.savefig(fpath, dpi=300)
        plt.close()
        print(f"Salvato {fpath}")

def export_json(df, stats_df, kde_grid, kde_curves, out_path="docs/daily_anomalies.json"):
    print("Esportazione dati per il web...")
    ventenni = [
        (1940, 1959, "1940-1959", "#1f77b4"),
        (1960, 1979, "1960-1979", "#2ca02c"),
        (1980, 1999, "1980-1999", "#ff7f0e"),
        (2000, 2019, "2000-2019", "#d62728"),
        (2020, 2025, "2020-2025", "#9467bd")
    ]
    ventenni_curves = {}
    ventenni_stats = []
    for start_y, end_y, label, color in ventenni:
        v_df = df[(df['year'] >= start_y) & (df['year'] <= end_y)]
        v_anom = v_df['anomaly'].dropna().values
        if len(v_anom) == 0: continue
        kde = stats.gaussian_kde(v_anom, bw_method=0.25)(kde_grid)
        ventenni_curves[label] = [round(float(v), 4) for v in kde]
        ventenni_stats.append({
            'label': label,
            'start_year': start_y,
            'end_year': end_y,
            'color': color,
            'mean': round(float(np.mean(v_anom)), 3),
            'std': round(float(np.std(v_anom)), 3),
            'p90': round(float(np.percentile(v_anom, 90)), 3),
            'p95': round(float(np.percentile(v_anom, 95)), 3)
        })

    data = {
        'grid': [round(float(x), 3) for x in kde_grid],
        'years': stats_df['year'].tolist(),
        'stats': stats_df.to_dict(orient='records'),
        'curves': kde_curves,
        'ventenni_stats': ventenni_stats,
        'ventenni_curves': ventenni_curves
    }
    with open(out_path, 'w') as f:
        json.dump(data, f)
    print(f"Salvato {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")

if __name__ == '__main__':
    df = load_and_process_data()
    stats_df, kde_grid, kde_curves = analyze_distributions(df)
    
    generate_superimposed_plot(stats_df, kde_grid, kde_curves)
    generate_statistics_plot(stats_df)
    generate_ventenni_plots(df, kde_grid)
    export_json(df, stats_df, kde_grid, kde_curves)
    os.system("rm -f docs/weekly_anomalies_*")
    print("Elaborazione e analisi terminate con successo!")
