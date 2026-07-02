import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

data_dir = "data_italy"
dfs = []

latitudes = {
    'Aosta': 45.7373, 'Torino': 45.0703, 'Genova': 44.4056, 'Milano': 45.4642,
    'Trento': 46.0697, 'Venezia': 45.4408, 'Trieste': 45.6495, 'Bologna': 44.4949,
    'Firenze': 43.7696, 'Perugia': 43.1107, 'Ancona': 43.6158, 'Roma': 41.9028,
    'LAquila': 42.3498, 'Campobasso': 41.5603, 'Napoli': 40.8518, 'Bari': 41.1171,
    'Potenza': 40.6404, 'Catanzaro': 38.9098, 'Palermo': 38.1157, 'Cagliari': 39.2238
}

for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        city = file.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, file))
        df['city'] = city
        dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
all_data['date'] = pd.to_datetime(all_data['date'])
all_data['year'] = all_data['date'].dt.year

all_data = all_data.sort_values(by=['city', 'date'])

# Filter from 1940 onwards
all_data = all_data[all_data['year'] >= 1940]

# Calculate CDD
all_data['is_dry'] = all_data['precipitation_sum'] < 1.0
all_data['dry_streak'] = all_data.groupby(['city', 'year', (~all_data['is_dry']).cumsum()])['is_dry'].cumsum()
yearly_cdd = all_data.groupby(['city', 'year'])['dry_streak'].max().reset_index()
yearly_cdd.rename(columns={'dry_streak': 'cdd'}, inplace=True)

# Calculate records
yearly_cdd = yearly_cdd.sort_values(by=['city', 'year'])
yearly_cdd['prev_max'] = yearly_cdd.groupby('city')['cdd'].transform(lambda x: x.cummax().shift(1))

yearly_cdd['is_record'] = (yearly_cdd['cdd'] > yearly_cdd['prev_max']).astype(int)

# 15 years baseline
yearly_cdd.loc[yearly_cdd['year'] < 1955, 'is_record'] = -1

heatmap_data = yearly_cdd.pivot(index='city', columns='year', values='is_record').astype(int)
heatmap_cdd = yearly_cdd.pivot(index='city', columns='year', values='cdd')

heatmap_data['latitude'] = heatmap_data.index.map(latitudes)
heatmap_data = heatmap_data.sort_values('latitude', ascending=False)
heatmap_cdd['latitude'] = heatmap_cdd.index.map(latitudes)
heatmap_cdd = heatmap_cdd.sort_values('latitude', ascending=False)

display_cities = [c.replace('LAquila', "L'Aquila") for c in heatmap_data.index]
heatmap_data.index = [f"{city} ({lat:.1f}°N)" for city, lat in zip(display_cities, heatmap_data['latitude'])]
heatmap_data = heatmap_data.drop(columns=['latitude'])
heatmap_cdd = heatmap_cdd.drop(columns=['latitude'])

annot_data = np.where(heatmap_data == 1, heatmap_cdd.astype(int).astype(str), "")

fig, ax = plt.subplots(figsize=(30, 12))

# Colormap for drought: -1=LightGray, 0=White, 1=Burnt Orange
cmap = mcolors.ListedColormap(['lightgray', 'white', '#cc6600'])

sns.heatmap(heatmap_data, cmap=cmap, ax=ax, annot=annot_data, fmt="", annot_kws={"size": 7, "color": "white", "weight": "bold"},
            linewidths=0.1, linecolor='lightgray', xticklabels=True, cbar=False, vmin=-1, vmax=1)

# Highlight columns (years) with records >= 25% of cities
threshold = int(heatmap_data.shape[0] / 4)
records_per_year = (heatmap_data == 1).sum(axis=0)
years_to_highlight = records_per_year[records_per_year >= threshold].index

for i, year in enumerate(heatmap_data.columns):
    if year in years_to_highlight:
        ax.add_patch(plt.Rectangle((i, 0), 1, heatmap_data.shape[0], fill=False, edgecolor='blue', lw=3))
        labels = ax.get_xticklabels()
        if i < len(labels):
            labels[i].set_weight("bold")
            labels[i].set_color("blue")

ax.set_title('Nuovi Record di Giorni Secchi Consecutivi (CDD) nei Capoluoghi Italiani (1940-2026) [Primi 15 anni usati come storico]', fontsize=18)
ax.set_xlabel('Anno', fontsize=14)
ax.set_ylabel('Città (Da Nord a Sud)', fontsize=14)

plt.xticks(rotation=45)
plt.tight_layout()

artifact_dir = '/Users/lcozzuto/.gemini/antigravity/brain/e9a954cd-73d4-4826-91e2-4e6263f6d002'
output_path1 = os.path.join(artifact_dir, 'record_heatmap_cdd_italy.png')
output_path2 = '/Users/lcozzuto/git/meteo/record_heatmap_cdd_italy.png'
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"Records heatmap saved to {output_path1} and {output_path2}")
