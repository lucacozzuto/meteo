import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

data_dir = "data"
dfs = []

latitudes = {
    'London': 51.5074, 'Paris': 48.8566, 'Berlin': 52.5200,
    'Madrid': 40.4168, 'Rome': 41.9028, 'Amsterdam': 52.3676,
    'Brussels': 50.8503, 'Vienna': 48.2082, 'Prague': 50.0755,
    'Warsaw': 52.2297, 'Budapest': 47.4979, 'Stockholm': 59.3293,
    'Oslo': 59.9127, 'Copenhagen': 55.6761, 'Helsinki': 60.1695,
    'Dublin': 53.3498, 'Athens': 37.9838, 'Lisbon': 38.7223,
    'Reykjavik': 64.1466, 'Moscow': 55.7558, 'Kyiv': 50.4501,
    'Bucharest': 44.4268, 'Sofia': 42.6977, 'Belgrade': 44.8125,
    'Zagreb': 45.8150, 'Sarajevo': 43.8563, 'Skopje': 42.0000,
    'Tirana': 41.3275, 'Podgorica': 42.4411, 'Pristina': 42.6629,
    'Bratislava': 48.1486, 'Ljubljana': 46.0569, 'Tallinn': 59.4370,
    'Riga': 56.9496, 'Vilnius': 54.6872, 'Chisinau': 47.0105,
    'Minsk': 53.9006, 'Bern': 46.9480
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

heatmap_data.index = [f"{city} ({lat:.1f}°N)" for city, lat in zip(heatmap_data.index, heatmap_data['latitude'])]
heatmap_data = heatmap_data.drop(columns=['latitude'])
heatmap_cdd = heatmap_cdd.drop(columns=['latitude'])

annot_data = np.where(heatmap_data == 1, heatmap_cdd.astype(int).astype(str), "")

fig, ax = plt.subplots(figsize=(30, 16))

# Colormap for drought: -1=LightGray, 0=White, 1=Burnt Orange
cmap = mcolors.ListedColormap(['lightgray', 'white', '#cc6600'])

sns.heatmap(heatmap_data, cmap=cmap, ax=ax, annot=annot_data, fmt="", annot_kws={"size": 7, "color": "white", "weight": "bold"},
            linewidths=0.1, linecolor='lightgray', xticklabels=True, cbar=False, vmin=-1, vmax=1)

# Highlight columns (years) with records >= 5 cities
threshold = 5
records_per_year = (heatmap_data == 1).sum(axis=0)
years_to_highlight = records_per_year[records_per_year >= threshold].index

for i, year in enumerate(heatmap_data.columns):
    if year in years_to_highlight:
        ax.add_patch(plt.Rectangle((i, 0), 1, heatmap_data.shape[0], fill=False, edgecolor='blue', lw=3))
        labels = ax.get_xticklabels()
        if i < len(labels):
            labels[i].set_weight("bold")
            labels[i].set_color("blue")

ax.set_title('Nuovi Record di Giorni Secchi Consecutivi (CDD) in Europa (1940-2026) [Primi 15 anni usati come storico]', fontsize=18)
ax.set_xlabel('Anno', fontsize=14)
ax.set_ylabel('Città (Da Nord a Sud)', fontsize=14)

plt.xticks(rotation=45)
plt.tight_layout()

artifact_dir = '/Users/lcozzuto/.gemini/antigravity/brain/e9a954cd-73d4-4826-91e2-4e6263f6d002'
output_path1 = os.path.join(artifact_dir, 'record_heatmap_cdd_europe.png')
output_path2 = '/Users/lcozzuto/git/meteo/record_heatmap_cdd_europe.png'
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"Records heatmap saved to {output_path1} and {output_path2}")
