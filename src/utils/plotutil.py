import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn import preprocessing

def corr_heatmap(df, save=False, *args, **kwargs):
    """Plot a heatmap of the correlation matrix of the dataframe."""
    sns.set_style(style='whitegrid')
    plt.figure(figsize=(25,10))
    sns.heatmap(df.corr(),vmin=-1,vmax=1,annot=True,cmap='BuPu')
    if save: plt.savefig(*args, **kwargs)
    plt.show()