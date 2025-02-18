import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
from sklearn import preprocessing
import mplhep as hep

class PlotUtil:
    def corr_discrete_conti_matrix(df, X, Y, num_bins=10, weight=None):
        """Plot a matrix of the counts of the discrete variable X and the binned continuous variable Y."""
        y_lower, y_upper = np.percentile(df[Y], [1, 99])
        df_filtered = df[(df[Y] > y_lower) & (df[Y] < y_upper)]
        y_min, y_max = np.floor(df_filtered[Y].min()), np.ceil(df_filtered[Y].max())

        approx_binsY = np.linspace(y_min, y_max, num_bins)
        binsY = np.round(approx_binsY).astype(int)
        binsY = np.unique(binsY)
        df_filtered['Y_binned'] = pd.cut(df_filtered[Y], bins=binsY, include_lowest=False)
        if weight is not None:
            counts = np.round(df_filtered.groupby([X, 'Y_binned'])[weight].sum()).unstack(fill_value=0)
        else:
            counts = df_filtered.groupby([X, 'Y_binned']).size().unstack(fill_value=0)
        print(counts)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(counts, cmap='BuPu', annot=True, fmt='g', ax=ax)
        plt.xlabel(Y)
        plt.ylabel(X)
        plt.title("Correlation Matrix")
        plt.show()
    
    @staticmethod
    def table_to_plot(df, save=False, savename='table.png', *args, **kwargs):
        # ! This currently produces ugly tables
        """Plot a table."""
        fig, ax = plt.subplots()
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', **kwargs)
        if save: plt.savefig(savename)
        else: plt.show()
    
 


def corr_heatmap(df, save=False, *args, **kwargs):
    """Plot a heatmap of the correlation matrix of the dataframe."""
    sns.set_style(style='whitegrid')
    plt.figure(figsize=(25,10))
    sns.heatmap(df.corr(),vmin=-1,vmax=1,annot=True,cmap='BuPu')
    if save: plt.savefig(*args, **kwargs)
    plt.show()
