import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#load histogram data
df = pd.read_csv('histogram_sequential.csv')

print(f"Loaded {len(df):,} non-empty cells")
print("Generating 4-panel statistical analysis...")

#create figure with 2x2 grid
fig = plt.figure(figsize=(16, 12), facecolor='white')
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

#Panel A: Event Count Distribution (Log Scale)
ax1 = fig.add_subplot(gs[0, 0])
counts, bins, patches = ax1.hist(df['count'], bins=50, color='#2E86AB', 
                                  edgecolor='white', alpha=0.8, linewidth=1.5)

#color gradient for bars
cm = plt.cm.Blues
for i, patch in enumerate(patches):
    patch.set_facecolor(cm(0.3 + 0.7 * i / len(patches)))

ax1.set_xlabel('Events per Cell', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Cells (log scale)', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Events per Cell', fontsize=13, 
              fontweight='bold', pad=15)
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, axis='y')
ax1.text(0.97, 0.97, f'Most cells have 1-2 events\nHeavy-tailed distribution', 
         transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

#Panel B: Magnitude Distribution

ax2 = fig.add_subplot(gs[0, 1])
counts, bins, patches = ax2.hist(df['max_magnitude'], bins=40, 
                                  color='#D62246', edgecolor='white', 
                                  alpha=0.8, linewidth=1.5)

#color gradient
cm = plt.cm.Reds
for i, patch in enumerate(patches):
    patch.set_facecolor(cm(0.3 + 0.7 * i / len(patches)))

ax2.set_xlabel('Maximum Magnitude', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Cells', fontsize=12, fontweight='bold')
ax2.set_title('Distribution of Maximum Magnitudes', fontsize=13, 
              fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axvline(x=df['max_magnitude'].mean(), color='black', linestyle='--', 
            linewidth=2, label=f'Mean: {df["max_magnitude"].mean():.2f}')
ax2.legend(loc='upper right', fontsize=10)

#Panel C: Activity by Latitude

ax3 = fig.add_subplot(gs[1, 0])
lat_dist = df.groupby('lat_bin')['count'].sum().sort_values(ascending=True)
colors_gradient = plt.cm.viridis(np.linspace(0.2, 0.9, len(lat_dist)))
ax3.barh(range(len(lat_dist)), lat_dist.values, color=colors_gradient, 
         alpha=0.8, edgecolor='white', linewidth=0.5)
ax3.set_xlabel('Total Earthquake Count', fontsize=12, fontweight='bold')
ax3.set_ylabel('Latitude Bin', fontsize=12, fontweight='bold')
ax3.set_title('Seismic Activity by Latitude Band', fontsize=13, 
              fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3, axis='x')

#reference for equator
equator_idx = np.where(lat_dist.index == 90)[0]
if len(equator_idx) > 0:
    ax3.axhline(y=equator_idx[0], color='red', linestyle='--', 
                linewidth=2, alpha=0.5, label='Equator (lat=90)')
    ax3.legend(loc='lower right', fontsize=10)

#Panel D: Top 20 Hotspots

ax4 = fig.add_subplot(gs[1, 1])
top_cells = df.nlargest(20, 'count')[['lat_bin', 'lon_bin', 'count']].copy()
top_cells['location'] = top_cells.apply(
    lambda r: f"({int(r['lat_bin'])},{int(r['lon_bin'])})", axis=1
)
colors_gradient = plt.cm.Oranges(np.linspace(0.4, 0.9, len(top_cells)))
bars = ax4.barh(range(len(top_cells)), top_cells['count'].values, 
                color=colors_gradient, edgecolor='white', 
                linewidth=1.5, alpha=0.9)
ax4.set_yticks(range(len(top_cells)))
ax4.set_yticklabels(top_cells['location'].values, fontsize=9)
ax4.set_xlabel('Event Count', fontsize=12, fontweight='bold')
ax4.set_title('Top 20 Most Active Cells', fontsize=13, 
              fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3, axis='x')
ax4.invert_yaxis()

#value labels on bars
for i, (bar, val) in enumerate(zip(bars, top_cells['count'].values)):
    ax4.text(val + 5, i, str(int(val)), va='center', 
             fontsize=9, fontweight='bold')

#main title
fig.suptitle('Statistical Analysis of 3D Spatiotemporal Histogram', 
             fontsize=18, fontweight='bold', y=0.98)

plt.savefig('figure(4)_statistical_analysis.png', dpi=300, 
            bbox_inches='tight', facecolor='white')
plt.close()