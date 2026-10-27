# Axis tick formatter.
from matplotlib.ticker import FuncFormatter

def custom_formatter(y, _):
    if y == 0:
        return '0'
    else:
        return f'{y:.1f}'  # one decimal
