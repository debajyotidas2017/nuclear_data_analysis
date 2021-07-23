# plot templates to be used in other files --------------------------------

import matplotlib.pyplot as plt

# Please see ** plt.rcParams.keys() ** ------------------------------------

plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['lines.markersize'] = 5
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 15

# xtick settings -----------------------------------
plt.rcParams['xtick.direction'] = "in"      # direction of xticks. For better visualization we will use inward direction. Default is outward.
plt.rcParams['xtick.major.size'] = 10       # size of major xtick
plt.rcParams['xtick.minor.visible'] = True  # shows minor xtick
plt.rcParams['xtick.minor.size'] = 5        # size of minor ytick

# ytick settings -----------------------------------
plt.rcParams['ytick.direction'] = "in"      # direction of yticks. For better visualization we will use inward direction. Default is outward.
plt.rcParams['ytick.major.size'] = 15       # size of major ytick
plt.rcParams['ytick.minor.visible'] = True  # shows minor ytick
plt.rcParams['ytick.minor.size'] = 7        # size of minor ytick

# Figure parameters to save the figure -------------------------------------------------

plt.rcParams['figure.figsize'] = (14, 9.8)     # figure size in inches

# plt.rcParams['figure.dpi'] = 300               # resolution of figure in dots per inch dpi

# plt.figure(figsize=(6.4,4.8), dpi=300)                 # another way of writing figure parameters
