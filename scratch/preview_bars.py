import json
import pandas as pd
import matplotlib.pyplot as plt
import os

with open('docs/monthly_records.json', 'r') as f:
    data = json.load(f)

city_data = data['Italy']['Roma']
heatwaves = city_data['heatwaves']

dates = [pd.to_datetime(hw['start']) for hw in heatwaves]
anomalies = [hw['anomaly'] for hw in heatwaves]

plt.figure(figsize=(10, 5))
plt.bar(dates, anomalies, color='#e34a33', width=10) # 10 days width just for visibility
plt.title('Ondate di Calore Estreme - Roma_1940 (Barre Isolate)')
plt.xlabel('Data')
plt.ylabel('Anomalia Termica (°C)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('scratch/heatwave_bars.png', dpi=150)
plt.close()
print("Plot saved to scratch/heatwave_bars.png")
