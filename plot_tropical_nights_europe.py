import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

data_dir = "data"
all_data = pd.DataFrame()

# Latitudes for sorting (European Capitals)
latitudes = {
    'London': 51.5074, 'Paris': 48.8566, 'Berlin': 52.5200,
    'Madrid': 40.4168, 'Barcelona': 41.35, 'Rome': 41.9028, 'Amsterdam': 52.3676,
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

dfs = []
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        city = file.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, file))
        df['city'] = city
        dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
all_data['date'] = pd.to_datetime(all_data['date'])
all_data['year'] = all_data['date'].dt.year

# Filter from 1940 onwards
all_data = all_data[all_data['year'] >= 1940]

# Notti tropicali: temperatura minima >= 20 C
all_data['is_tropical'] = (all_data['temperature_2m_min'] >= 20).astype(int)
yearly_data = all_data.groupby(['city', 'year'])['is_tropical'].sum().reset_index()

heatmap_data = yearly_data.pivot(index='city', columns='year', values='is_tropical').fillna(0).astype(int)

heatmap_data['latitude'] = heatmap_data.index.map(latitudes)
heatmap_data = heatmap_data.sort_values('latitude', ascending=False)
heatmap_data.index = [f"{city} ({lat:.1f}°N)" for city, lat in zip(heatmap_data.index, heatmap_data['latitude'])]
heatmap_data = heatmap_data.drop(columns=['latitude'])

annot_data = np.where(heatmap_data == 0, "", heatmap_data.astype(str))

fig, ax = plt.subplots(figsize=(30, 16))

# Use a purplish colormap for nights
sns.heatmap(heatmap_data, cmap=mcolors.LinearSegmentedColormap.from_list('WhitePurples', ['white', 'plum', 'purple', 'indigo']), ax=ax, annot=annot_data, fmt="", annot_kws={"size": 8},
            linewidths=0.1, linecolor='lightgray', xticklabels=True, cbar_kws={'label': 'Notti >= 20°C'})

ax.set_title('Numero di Notti Tropicali (Minima >= 20°C) per anno in Europa', fontsize=18)
ax.set_xlabel('Anno', fontsize=14)
ax.set_ylabel('Città (Da Nord a Sud)', fontsize=14)

plt.xticks(rotation=45)
plt.tight_layout()

output_path = 'docs/tropical_nights_europe.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Tropical nights heatmap saved to {output_path}")
