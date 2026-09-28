"""Shared matplotlib settings. Import before pyplot."""
import matplotlib
matplotlib.use('Agg')
from cycler import cycler
import matplotlib.pyplot as plt  # noqa: E402, F401

# Material Design color palette
PALETTE = [
    '#2196F3',  # blue
    '#FF9800',  # orange
    '#4CAF50',  # green
    '#E91E63',  # pink
    '#9C27B0',  # purple
    '#00BCD4',  # cyan
    '#F44336',  # red
    '#8BC34A',  # light green
]

matplotlib.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'axes.titlesize': 12,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    # Color cycle
    'axes.prop_cycle': cycler(color=PALETTE),
    # Grid
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.6,
    'grid.color': '#CCCCCC',
    # Spines
    'axes.spines.top': False,
    'axes.spines.right': False,
    # Lines
    'lines.linewidth': 1.6,
    'lines.markersize': 5,
    # Legend
    'legend.framealpha': 0.85,
    'legend.edgecolor': '#CCCCCC',
})

ERRORBAR_KW = dict(fmt='none', ecolor='black', elinewidth=1.0,
                   capsize=3, capthick=1.0, zorder=5)

# ── Semantic role colors (use these everywhere, not raw hex) ──────────────────
C_VANILLA = '#888888'   # grey
C_FROZEN  = '#2196F3'   # blue  (Frozen-phi)
C_BILEVEL = '#E96E32'   # orange (Bilevel)
C_FILL_FP = '#BBDEFB'   # light blue fill band
C_FILL_BL = '#FFE0B2'   # light orange fill band
C_GAIN_POS = '#4CAF50'  # green  (improvement / gain < 0)
C_GAIN_NEG = '#E91E63'  # pink   (degradation / gain > 0)
C_GAIN_NEU = '#888888'  # grey   (neutral / near zero)
