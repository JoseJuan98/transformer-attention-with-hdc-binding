# -*- coding: utf-8 -*-
"""Visualization configuration used for a paper."""
# Third party imports
import matplotlib

# Define a modern, distinguishable color palette
modern_palette = [
    "#842e22",  # Modern Red
    "#339999",  # Modern Teal
    "#ff8773",  # Modern Coral
    "#336699",  # Modern Blue
    "#e6b800",  # Modern Yellow
    "#9966cc",  # Modern Purple
    "#66cc66",  # Modern Green
    "#cc6699",  # Modern Pink
    "#ffcc00",  # Modern Gold
    "#0099cc",  # Modern Sky Blue
    "#993333",  # Modern Maroon
    "#33cc33",  # Modern Lime
    "#cc33cc",  # Modern Magenta
]

rc_config = {
    # Figure and axes settings
    # 'figure.figsize': (12, 6),  # Default figure size
    "figure.figsize": (19, 10),
    "figure.dpi": 300,  # Resolution for modern clarity
    "figure.autolayout": True,  # Optimize figure for tight layout
    "axes.titlesize": 18,  # Title font size
    "axes.titleweight": "bold",  # Bold title
    "axes.labelsize": 22,  # Axis labels font size
    # "axes.labelweight": "bold",  # Bold axis labels
    "axes.edgecolor": "#333333",  # Dark gray axis lines
    "axes.linewidth": 1.5,  # Thicker axis lines
    # Grid settings
    "grid.color": "#eaeaea",  # Light gridlines
    "grid.linestyle": "-",  # Solid gridlines
    "grid.linewidth": 0.8,  # Thin gridlines
    "axes.grid": True,  # Enable grid by default
    # Ticks settings
    "xtick.labelsize": 20,  # X-axis tick font size
    "ytick.labelsize": 20,  # Y-axis tick font size
    "xtick.color": "#333333",  # Dark gray tick color
    "ytick.color": "#333333",  # Dark gray tick color
    "xtick.major.size": 6,  # Major tick size
    "ytick.major.size": 6,  # Major tick size
    # Line and Marker settings
    "lines.linewidth": 2.5,  # Thicker lines for visibility
    "lines.markersize": 8,  # Moderate marker size
    # Legend settings
    "legend.fontsize": 12,  # Legend font size
    "legend.loc": "best",  # Best location for legend
    "legend.frameon": False,  # No legend frame
    # Font and text settings
    "font.size": 16,  # General font size
    "font.family": "sans-serif",  # Use sans-serif fonts for modern look
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    # Color cycle (modern and visually appealing colors)
    "axes.prop_cycle": matplotlib.cycler(color=modern_palette),  # Use the palette for the color cycle
}
