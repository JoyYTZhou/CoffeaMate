import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import mplhep as hep
import matplotlib as mpl

class PlotStyle:
    """Handles basic plot styling and setup"""
    COLORS = list(mpl.colormaps['Set2'].colors)
    SIGNAL_COLORS = ['red', 'blue', 'forestgreen']
    
    @staticmethod
    def setup_cms_style(ax, lumi=None, year=None):
        """Apply CMS style to an axis"""
        hep.cms.label(ax=ax, loc=2, label='Work in Progress', 
                     fontsize=11, com=13.6, lumi=lumi, year=year)
        ax.tick_params(axis='both', which='major', labelsize=10, length=0)
    
    @staticmethod
    def setup_axis(ax, xlabel=None, ylabel=None, title=None, fontsize=12):
        """Setup axis labels and title"""
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        if title:
            ax.set_title(title, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=10, length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    @staticmethod
    def create_figure(n_row=1, n_col=1, figsize=None):
        """Create figure with default sizing"""
        if figsize is None:
            figsize = (8*n_col, 5*n_row)
        return plt.subplots(n_row, n_col, figsize=figsize)
    
    @staticmethod
    def create_ratio_figure(title: str, x_label, top_ylabel, bottom_ylabel, num_figure=1, set_log=False, lumi=None):
        """Create a figure with ratio panel"""
        fig = plt.figure(figsize=(8*num_figure, 6))
        fig.suptitle(title, fontsize=14)
        hep.style.use("CMS")
        
        gs = fig.add_gridspec(2, 1*num_figure, height_ratios=(4, 1))
        
        ax2s = []
        axs = []
        
        x_label = [x_label] if not isinstance(x_label, list) else x_label
        top_ylabel = [top_ylabel] if not isinstance(top_ylabel, list) else top_ylabel
        bottom_ylabel = [bottom_ylabel] if not isinstance(bottom_ylabel, list) else bottom_ylabel
        
        for i in range(num_figure):
            ax2 = fig.add_subplot(gs[1,i])
            ax = fig.add_subplot(gs[0,i], sharex=ax2)
            
            PlotStyle.setup_cms_style(ax, lumi=lumi)
            PlotStyle.setup_axis(ax, ylabel=top_ylabel[i], fontsize=12)
            ax.tick_params(labelbottom=False)
            
            PlotStyle.setup_axis(
                ax2, 
                xlabel=x_label[i],
                ylabel=bottom_ylabel[i],
                fontsize=11
            )
            ax2.yaxis.get_offset_text().set_fontsize(11)
            
            if set_log:
                ax2.set_yscale('log')
                
            axs.append(ax)
            ax2s.append(ax2)

        fig.subplots_adjust(hspace=0.15)
        return fig, axs, ax2s
    
class HistogramHelper:
    """Handles histogram operations"""
    @staticmethod
    def make_histogram(data, bins, range, weights=None, density=False):
        """Create histogram with proper overflow handling"""
        if isinstance(bins, int):
            bins = np.linspace(*range, bins+1)
            data = np.clip(data, bins[0], bins[-1])
        return np.histogram(data, bins=bins, weights=weights, density=density)

    @staticmethod
    def calc_ratio_and_errors(num, den, num_err, den_err):
        """Calculate ratio and its error"""
        ratio = np.nan_to_num(num / den, nan=0., posinf=0.)
        ratio_err = np.nan_to_num(
            ratio * np.sqrt((num_err/num)**2 + (den_err/den)**2), 
            nan=0., posinf=0.
        )
        return ratio, ratio_err

    @staticmethod
    def normalize_histogram(hist, weights, bin_width):
        """Normalize histogram and calculate errors"""
        norm_hist = hist / (np.sum(weights) * bin_width)
        norm_err = np.sqrt(hist) / (np.sum(weights) * bin_width)
        return norm_hist, norm_err
    
    @staticmethod
    def _prepare_data(df, Y, Y_range):
        """Prepare and filter data based on range."""
        if Y_range is not None:
            y_min, y_max = Y_range
            df_filtered = df.copy()
        else:
            y_lower, y_upper = np.percentile(df[Y], [1, 99])
            df_filtered = df.copy()[(df[Y] > y_lower) & (df[Y] < y_upper)]
            y_min, y_max = np.floor(df_filtered[Y].min()), np.ceil(df_filtered[Y].max())
        
        return df_filtered, y_min, y_max

class PlotUtil:
    @staticmethod
    def corr_discrete(df, X, Y, X_label, Y_label_1, Y_label_2, save_name=None, num_bins=10, Y_range=None, weight=None):
        """Plot a 1D binned distribution of the ratio of the two types of values in column X along the Y axis.
        
        Parameters
        - `X`: discrete column to be binned, contains boolean values
        - `Y`: continous column to be binned
        - `Y_label_1`: Y-axis label for the ratio distribution
        - `Y_label_2`: Y-axis label for the percentage distribution
        
        Plot two figures:
        1. A 1D binned distribution of the ratio of the two types of values in column X along the Y axis
        2. A 1D binned distribution of the percentage of one class across bins in Y
        """
        df_filtered, y_min, y_max = HistogramHelper._prepare_data(df, Y, Y_range)

        bins = np.linspace(y_min, y_max, num_bins, endpoint=False)

        # Separate data by X value
        data_true = df_filtered[df_filtered[X]][Y].values
        data_false = df_filtered[~df_filtered[X]][Y].values
        
        # Get weights if provided
        weights_true = df_filtered[df_filtered[X]][weight].values if weight else None
        weights_false = df_filtered[~df_filtered[X]][weight].values if weight else None

        # Create histograms using HistogramHelper
        hist_true, _ = HistogramHelper.make_histogram(data_true, bins=bins, range=(y_min, y_max), weights=weights_true)
        hist_false, _ = HistogramHelper.make_histogram(data_false, bins=bins, range=(y_min, y_max), weights=weights_false)

        # Calculate errors
        err_true = np.sqrt(hist_true) if weights_true is None else np.sqrt(np.histogram(data_true, bins=bins, weights=weights_true**2)[0])
        err_false = np.sqrt(hist_false) if weights_false is None else np.sqrt(np.histogram(data_false, bins=bins, weights=weights_false**2)[0])
        
        # Calculate ratio and percentage with errors
        ratio, ratio_err = HistogramHelper.calc_ratio_and_errors(hist_true, hist_false, err_true, err_false)
        total = hist_true + hist_false
        err_total = np.sqrt(total)
        percentage, percentage_err = HistogramHelper.calc_ratio_and_errors(hist_true, total, err_true, err_total)

        fig, (ax, ax2) = PlotStyle.create_figure(n_row=2, n_col=1, figsize=(12, 10))
        PlotStyle.setup_cms_style(ax, lumi=None)
        PlotStyle.setup_cms_style(ax2, lumi=None)
        
        hep.histplot(
            ratio,
            bins,
            yerr=ratio_err,
            ax=ax,
            label=Y_label_1,
            color=PlotStyle.COLORS[0],
            histtype='errorbar',
            alpha=0.7
        )
        PlotStyle.setup_axis(ax, X_label, Y_label_1, f"{Y_label_1} along {X_label}")

        # Bottom panel: Percentage plot
        hep.histplot(
            percentage,
            bins,
            yerr=percentage_err,
            ax=ax2,
            label=Y_label_2,
            color=PlotStyle.COLORS[1],
            histtype='errorbar',
            alpha=0.7
        )
        PlotStyle.setup_axis(ax2, X_label, Y_label_2, f"{Y_label_2} along {X_label}")
        
        # Set y-axis ranges with padding for error bars
        ax.set_ylim(min(ratio - ratio_err) / 1.2, max(ratio + ratio_err) * 1.2)
        ax2.set_ylim(min(percentage - percentage_err) / 1.2, min(1.2, max(percentage + percentage_err) * 1.2))
        
        plt.tight_layout()
        
        if save_name is not None:
            plt.savefig(save_name, bbox_inches='tight', dpi=300)
        
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
