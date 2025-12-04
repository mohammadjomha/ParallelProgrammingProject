import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#load histogram data
df = pd.read_csv('histogram_sequential.csv')

#create figure
fig, ax = plt.subplots(figsize=(18, 9), facecolor='white')

#create bubble plot
scatter = ax.scatter(
    df['lon_bin'], 
    df['lat_bin'],
    s=df['count'] * 5,  # Size based on event count
    c=df['max_magnitude'],  # Color based on magnitude
    cmap='YlOrRd',  # Yellow-Orange-Red colormap
    alpha=0.6,
    edgecolors='white',
    linewidth=0.5
)

#colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('Maximum Magnitude', fontsize=13, fontweight='bold')
cbar.ax.tick_params(labelsize=11)

#labels and title
ax.set_xlabel('Longitude Bin (0-360 maps to -180° to +180°)', 
              fontsize=13, fontweight='bold')
ax.set_ylabel('Latitude Bin (0-180 maps to -90° to +90°)', 
              fontsize=13, fontweight='bold')
ax.set_title('Seismic Events: Spatial Distribution with Magnitude\nUSGS Data (Feb 2021 - Jan 2025)', 
             fontsize=16, fontweight='bold', pad=20)

#grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_xlim(-5, 365)
ax.set_ylim(-5, 185)

ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5, linewidth=1, 
           label='Equator')
ax.axvline(x=180, color='gray', linestyle=':', alpha=0.5, linewidth=1, 
           label='Prime Meridian')

#legend
sizes = [1, 10, 50, 100]
labels = ['1 event', '10 events', '50 events', '100 events']
legend_elements = [plt.scatter([], [], s=s*5, c='gray', alpha=0.6, 
                              edgecolors='white', linewidth=0.5) 
                   for s in sizes]
legend1 = ax.legend(legend_elements, labels, 
                    loc='upper left', title='Event Count', 
                    frameon=True, fancybox=True, shadow=True,
                    fontsize=10, title_fontsize=11)
ax.add_artist(legend1)

plt.tight_layout()
plt.savefig('figure(1)_bubble_chart.png', dpi=300, bbox_inches='tight', 
            facecolor='white')
plt.close()