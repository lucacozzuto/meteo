import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

data_dir = "data_italy"
all_data = pd.DataFrame()

latitudes = {
    'Aosta': 45.7373, 'Torino': 45.0703, 'Genova': 44.4056, 'Milano': 45.4642,
    'Trento': 46.0697, 'Venezia': 45.4408, 'Trieste': 45.6495, 'Bologna': 44.4949,
    'Firenze': 43.7696, 'Perugia': 43.1107, 'Ancona': 43.6158, 'Roma': 41.9028,
    'LAquila': 42.3498, 'Campobasso': 41.5603, 'Napoli': 40.8518, 'Bari': 41.1171,
    'Potenza': 40.6404, 'Catanzaro': 38.9098, 'Palermo': 38.1157, 'Cagliari': 39.2238
}

dfs = []
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        city = file.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, file))
        df['city'] = city
        df['temp_mean'] = (df['temperature_2m_max'] + df['temperature_2m_min']) / 2
        dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
all_data['date'] = pd.to_datetime(all_data['date'])
all_data['year'] = all_data['date'].dt.year
all_data['month'] = all_data['date'].dt.month

all_data = all_data[(all_data['year'] >= 1940) & (all_data['month'] == 6)]

# Absolute maximum June temp per year per city
yearly_june = all_data.groupby(['city', 'year'])['temperature_2m_max'].max().reset_index()

# Calculate records
yearly_june = yearly_june.sort_values(by=['city', 'year'])
yearly_june['prev_max'] = yearly_june.groupby('city')['temperature_2m_max'].transform(lambda x: x.cummax().shift(1))

# A record is when the current temp is strictly greater than all previous ones
yearly_june['is_record'] = (yearly_june['temperature_2m_max'] > yearly_june['prev_max']).astype(int)

# Ignore the first 15 years (1940-1954) as a baseline training period
yearly_june.loc[yearly_june['year'] < 1955, 'is_record'] = -1

heatmap_data = yearly_june.pivot(index='city', columns='year', values='is_record').astype(int)
heatmap_temps = yearly_june.pivot(index='city', columns='year', values='temperature_2m_max')

heatmap_data['latitude'] = heatmap_data.index.map(latitudes)
heatmap_data = heatmap_data.sort_values('latitude', ascending=False)
heatmap_temps['latitude'] = heatmap_temps.index.map(latitudes)
heatmap_temps = heatmap_temps.sort_values('latitude', ascending=False)

display_cities = [c.replace('LAquila', "L'Aquila") for c in heatmap_data.index]
heatmap_data.index = [f"{city} ({lat:.1f}°N)" for city, lat in zip(display_cities, heatmap_data['latitude'])]
heatmap_data = heatmap_data.drop(columns=['latitude'])
heatmap_temps = heatmap_temps.drop(columns=['latitude'])

annot_data = np.where(heatmap_data == 1, heatmap_temps.round(1).astype(str), "")

fig, ax = plt.subplots(figsize=(30, 12))
# Custom colormap: -1=LightGray, 0=White, 1=Less intense red (salmon/light coral)
cmap = mcolors.ListedColormap(['lightgray', 'white', '#ff7f7f'])

sns.heatmap(heatmap_data, cmap=cmap, ax=ax, annot=annot_data, fmt="", annot_kws={"size": 7, "color": "black", "weight": "bold"},
            linewidths=0.1, linecolor='lightgray', xticklabels=True, cbar=False, vmin=-1, vmax=1)

# Highlight columns (years) with records >= 25% of cities
threshold = int(heatmap_data.shape[0] / 4)
for i, col in enumerate(heatmap_data.columns):
    num_records = (heatmap_data[col] == 1).sum()
    if num_records >= threshold:
        ax.add_patch(plt.Rectangle((i, 0), 1, heatmap_data.shape[0], fill=False, edgecolor='blue', lw=3))
        ax.get_xticklabels()[i].set_color('blue')
        ax.get_xticklabels()[i].set_weight('bold')

ax.set_title('Nuovi Record di Temperatura Massima ASSOLUTA a GIUGNO nei Capoluoghi Italiani (1940-2026) [Primi 15 anni usati come storico]', fontsize=18)
ax.set_xlabel('Anno', fontsize=14)
ax.set_ylabel('Città (Da Nord a Sud)', fontsize=14)

plt.figtext(0.5, 0.01, "* Nota: i dati dal 27 al 30 giugno 2026 sono provvisori (modello operativo), in attesa della ri-analisi definitiva.", ha="center", fontsize=12, color="gray", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])

plt.xticks(rotation=45)

artifact_dir = '/Users/lcozzuto/.gemini/antigravity/brain/e9a954cd-73d4-4826-91e2-4e6263f6d002'
output_path1 = os.path.join(artifact_dir, 'record_heatmap_june_italy.png')
output_path2 = '/Users/lcozzuto/git/meteo/record_heatmap_june_italy.png'
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"Records heatmap saved to {output_path1} and {output_path2}")
