import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

#load the histogram data
df = pd.read_csv('histogram_sequential.csv')

#aggregate across time
geo_data = df.groupby(['lat_bin', 'lon_bin'])['count'].sum().reset_index()

#2D grid
grid = np.zeros((180, 360))
for _, row in geo_data.iterrows():
    grid[int(row['lat_bin']), int(row['lon_bin'])] = row['count']

#log scale for better visualization
grid_log = np.log10(grid + 1)  # +1 to avoid log(0)

fig, ax = plt.subplots(figsize=(18, 9), facecolor='white')


colors_list = ['#ffffff', '#d4e6f1', '#85c1e2', '#3498db', 
               '#2e86ab', '#8e44ad', '#c0392b']
n_bins = 256
cmap = LinearSegmentedColormap.from_list('custom_blue_purple_red', 
                                         colors_list, N=n_bins)

im = ax.imshow(grid_log, cmap=cmap, aspect='auto', 
               interpolation='bilinear', origin='lower')

#colorbar
cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.set_label('Earthquake Count (log scale)', fontsize=13, fontweight='bold')
cbar.ax.tick_params(labelsize=11)

#labels and title
ax.set_xlabel('Longitude Bin (0-360 maps to -180° to +180°)', 
              fontsize=13, fontweight='bold')
ax.set_ylabel('Latitude Bin (0-180 maps to -90° to +90°)', 
              fontsize=13, fontweight='bold')
ax.set_title('Global Earthquake Distribution - All Time Periods\nUSGS Data (Feb 2021 - Jan 2025)', 
             fontsize=16, fontweight='bold', pad=20)

#reference lines
ax.axhline(y=90, color='white', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=180, color='white', linestyle='--', alpha=0.5, linewidth=1)

#annotation
ax.text(0.02, 0.98, 'Pacific Ring of Fire clearly visible', 
        transform=ax.transAxes, fontsize=11, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('figure(2)_heatmap.png', dpi=300, bbox_inches='tight', 
            facecolor='white')
plt.close()