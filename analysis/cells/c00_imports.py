# Packages used by the analysis.
import xarray as xr
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaflux

import matplotlib.colors as mcolors

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.rcParams['font.family'] = 'Arial'

from scipy.stats import linregress

import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MultipleLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime
import matplotlib.patches as mpatches
import seaborn as sns
