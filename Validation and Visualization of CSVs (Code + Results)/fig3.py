import pandas as pd
import matplotlib.pyplot as plt

#histogram data
df = pd.read_csv('histogram_sequential.csv')

#aggregate by time bin
temporal_data = df.groupby('time_bin')['count'].sum().reset_index()

fig, ax = plt.subplots(figsize=(16, 6), facecolor='white')

#plot
ax.plot(temporal_data['time_bin'], temporal_data['count'], 
        linewidth=2.5, color='#2E86AB', marker='o', markersize=5,
        markerfacecolor='white', markeredgecolor='#2E86AB', 
        markeredgewidth=2)
ax.fill_between(temporal_data['time_bin'], temporal_data['count'], 
                alpha=0.3, color='#2E86AB')

#labels and title
ax.set_xlabel('Time Bin (0-120, approximately 12 days per bin)', 
              fontsize=13, fontweight='bold')
ax.set_ylabel('Total Earthquake Count', fontsize=13, fontweight='bold')
ax.set_title('Temporal Distribution of Seismic Activity\nFebruary 2021 - January 2025', 
             fontsize=16, fontweight='bold', pad=20)

#stats
mean_count = temporal_data['count'].mean()
max_count = temporal_data['count'].max()
min_count = temporal_data['count'].min()

stats_text = f'Mean: {mean_count:.1f} events/bin\nMax: {max_count} events/bin\nMin: {min_count} events/bin'
ax.text(0.98, 0.97, stats_text, 
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                  edgecolor='#2E86AB', linewidth=2))

#grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_xlim(-2, 122)

plt.tight_layout()
plt.savefig('figure(3)_temporal_analysis.png', dpi=300, bbox_inches='tight', 
            facecolor='white')
plt.close()