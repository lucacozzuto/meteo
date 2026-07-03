import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

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

display_cities = [c.replace('LAquila', "L'Aquila") for c in heatmap_data.index]
heatmap_data.index = [f"{city} ({lat:.1f}°N)" for city, lat in zip(display_cities, heatmap_data['latitude'])]
heatmap_data = heatmap_data.drop(columns=['latitude'])

fig, ax = plt.subplots(figsize=(30, 12))

# Use a purplish colormap for nights
sns.heatmap(heatmap_data, cmap='Purples', ax=ax, annot=True, fmt="d", annot_kws={"size": 8},
            linewidths=0.1, linecolor='lightgray', xticklabels=True, cbar_kws={'label': 'Notti >= 20°C'})

ax.set_title('Numero di Notti Tropicali (Minima >= 20°C) per anno nei Capoluoghi Italiani', fontsize=18)
ax.set_xlabel('Anno', fontsize=14)
ax.set_ylabel('Città (Da Nord a Sud)', fontsize=14)

plt.xticks(rotation=45)
plt.tight_layout()

output_path = '/Users/lcozzuto/git/meteo/docs/tropical_nights_italy.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Tropical nights heatmap saved to {output_path}")
