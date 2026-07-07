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
        df['temp_mean'] = (df['temperature_2m_min'] + df['temperature_2m_min']) / 2
        dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
all_data['date'] = pd.to_datetime(all_data['date'])
all_data['year'] = all_data['date'].dt.year

# Filter from 1940 onwards (include 2026)
all_data = all_data[all_data['year'] >= 1940]

# Absolute maximum temperature per year per city
yearly_data = all_data.groupby(['city', 'year'])['temperature_2m_min'].max().reset_index()

# Calculate records
yearly_data = yearly_data.sort_values(by=['city', 'year'])
yearly_data['prev_min'] = yearly_data.groupby('city')['temperature_2m_min'].transform(lambda x: x.cummax().shift(1))

# A record is when the current temp is strictly greater than all previous ones
yearly_data['is_record'] = (yearly_data['temperature_2m_min'] > yearly_data['prev_min']).astype(int)

# Ignore the first 15 years (1940-1954) as a baseline training period
yearly_data.loc[yearly_data['year'] < 1955, 'is_record'] = -1

heatmap_data = yearly_data.pivot(index='city', columns='year', values='is_record').astype(int)
heatmap_temps = yearly_data.pivot(index='city', columns='year', values='temperature_2m_min')

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
records_per_year = (heatmap_data == 1).sum(axis=0)
years_to_highlight = records_per_year[records_per_year >= threshold].index

for i, year in enumerate(heatmap_data.columns):
    if year in years_to_highlight:
        # Draw a rectangle around the column
        rect = plt.Rectangle((i, 0), 1, heatmap_data.shape[0], fill=False, edgecolor='blue', lw=3, zorder=10)
        ax.add_patch(rect)
        
        # Highlight the x-axis tick label
        labels = ax.get_xticklabels()
        if i < len(labels):
            labels[i].set_weight("bold")
            labels[i].set_color("blue")

ax.set_title('Nuovi Record di Temperatura Minima Più Alta nei Capoluoghi Italiani (1940-2026) [Primi 15 anni usati come storico]', fontsize=18)
ax.set_xlabel('Anno', fontsize=14)
ax.set_ylabel('Città (Da Nord a Sud)', fontsize=14)

plt.xticks(rotation=45)
plt.tight_layout()

output_path = 'docs/record_heatmap_yearly_min_italy.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Records heatmap saved to {output_path}")
