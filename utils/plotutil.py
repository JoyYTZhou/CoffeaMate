import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

class PlotUtil:
    def corr_discrete(df, X, Y, num_bins=10, Y_range=None, weight=None):
        """Plot a 1D binned distribution of the ratio of the two types of values in column X along the Y axis."""
        """
        Plot two figures:
        1. A 1D binned distribution of the ratio of the two types of values in column X along the Y axis
        2. A 1D binned distribution of the percentage of one class across bins in Y
        """
        y_lower, y_upper = np.percentile(df[Y], [1, 99])
        df_filtered = df[(df[Y] > y_lower) & (df[Y] < y_upper)]
        if Y_range is not None:
            y_min, y_max = Y_range
        else:
            y_min, y_max = np.floor(df_filtered[Y].min()), np.ceil(df_filtered[Y].max())

        approx_binsY = np.linspace(y_min, y_max, num_bins, endpoint=False)
        binsY = np.round(approx_binsY).astype(int)
        binsY = np.unique(binsY)
        df_filtered['Y_binned'] = pd.cut(df_filtered[Y], bins=binsY, include_lowest=True)

        unique_values_X = df[X].unique()
        if len(unique_values_X) != 2:
            raise ValueError("Column X must contain exactly two types of values.")

        value1, value2 = unique_values_X
        value_counts = df_filtered.groupby('Y_binned')[X].value_counts(normalize=True).unstack().fillna(0)
        value_counts['ratio'] = value_counts[value1] / value_counts[value2]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Ratio distribution
        sns.barplot(x=value_counts.index.astype(str), y=value_counts['ratio'], ax=ax1)
        ax1.set_xlabel(Y)
        ax1.set_ylabel(f'{X} Ratio of {value1} to {value2}')
        ax1.set_title(f"1D Binned Distribution of {X} Ratio along {Y}")
        ax1.tick_params(axis='x', rotation=45)

        # Plot 2: Percentage distribution of value1
        sns.barplot(x=value_counts.index.astype(str), y=value_counts[value1], ax=ax2)
        ax2.set_xlabel(Y)
        ax2.set_ylabel(f'Percentage of {value1}')
        ax2.set_title(f"1D Binned Distribution of {value1} Percentage along {Y}")
        ax2.tick_params(axis='x', rotation=45)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()
        plt.xlabel(Y)
        plt.ylabel('{} Ratio of {} to {}'.format(X, value1, value2))
        plt.title(f"1D Binned Distribution of {X} Ratio along {Y}")
        plt.xticks(rotation=45)
        plt.show()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Ratio distribution
        sns.barplot(x=value_counts.index.astype(str), y=value_counts['ratio'], ax=ax1)
        ax1.set_xlabel(Y)
        ax1.set_ylabel(f'{X} Ratio of {value1} to {value2}')
        ax1.set_title(f"1D Binned Distribution of {X} Ratio along {Y}")
        ax1.tick_params(axis='x', rotation=45)

        # Plot 2: Percentage distribution of value1
        sns.barplot(x=value_counts.index.astype(str), y=value_counts[value1], ax=ax2)
        ax2.set_xlabel(Y)
        plt.show()
        ax2.set_ylabel(f'Percentage of {value1}')
        ax2.set_title(f"1D Binned Distribution of {value1} Percentage along {Y}")
        ax2.tick_params(axis='x', rotation=45)

        # Adjust layout to prevent overlap
        plt.tight_layout()
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
