import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import mplhep as hep
import matplotlib as mpl
from typing import List, Tuple, Optional, Union, Any

import logging

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
        ax.tick_params(axis='both', which='minor', length=3)
        ax.minorticks_on()
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
    def make_histogram(data: Union[np.ndarray, pd.Series], bins, range, weights=None, density=False):
        """Create histogram with proper overflow handling.

        Parameters:
        data (Union[np.ndarray, pd.Series]): Input data to histogram
        bins: Number of bins or array of bin edges
        range: Tuple of (min, max) range for the histogram
        weights: Optional weights for each data point
        density: If True, normalize histogram to form a probability density

        Returns:
        tuple: (histogram counts/density, bin edges)
        """
        # Convert bins to edges if number provided
        if isinstance(bins, int):
            bins = np.linspace(*range, bins+1)

        # Create mask for data within range
        in_range = (data >= bins[0]) & (data <= bins[-1])

        # Clip data and weights together
        clipped_data = data[in_range]
        clipped_weights = weights[in_range] if weights is not None else None

        # Create histogram with clipped data and weights
        return np.histogram(
            clipped_data,
            bins=bins,
            weights=clipped_weights,
            density=density
        )

    @staticmethod
    def calc_ratio_and_errors(num, den, num_err, den_err):
        """Calculate ratio and its error with proper handling of zero values.
        
        Parameters:
        num (array): Numerator values
        den (array): Denominator values
        num_err (array): Numerator errors
        den_err (array): Denominator errors
        
        Returns:
        tuple: (ratio, ratio_error) with zeros where undefined
        """
        valid_mask = (num != 0) & (den != 0)
        
        ratio = np.zeros_like(num, dtype=float)
        ratio_err = np.zeros_like(num, dtype=float)
        
        ratio[valid_mask] = num[valid_mask] / den[valid_mask]
        
        ratio_err[valid_mask] = ratio[valid_mask] * np.sqrt(
            (num_err[valid_mask]/num[valid_mask])**2 + 
            (den_err[valid_mask]/den[valid_mask])**2
        )
        
        return ratio, ratio_err

    @staticmethod
    def normalize_histogram(hist, total_wgt, bin_width):
        """Normalize histogram to get probability density and calculate errors.

        Parameters:
        hist (array): Raw histogram counts/weights
        total_wgt (float): Sum of all weights/events
        bin_width (float): Width of each histogram bin

        Returns:
        tuple: (normalized_histogram, errors)
            - normalized_histogram is in units of probability density (events/bin_width)
            - errors are propagated appropriately

        Note:
        To get a proper probability density function (PDF), we need to:
        1. Divide by total events/weights to get probability per bin
        2. Divide by bin width to convert to density (per unit x)
        This ensures the total integral of the PDF equals 1
        """
        # First normalize by total events/weights
        prob_hist = hist / total_wgt

        # Then convert to density by dividing by bin width
        density_hist = prob_hist / bin_width

        # Propagate errors, assuming Poisson statistics
        # Error = sqrt(N)/N * normalized value, where N is raw counts
        density_err = np.where(
            hist > 0,  # Only calculate errors for non-empty bins
            density_hist * (np.sqrt(hist)/hist),
            0
        )
        return density_hist, density_err
    
    @staticmethod
    def average_histogram(histograms: List[np.ndarray]) -> np.ndarray:
        """Calculate the bin-wise average of multiple histograms.

        Parameters:
        histograms (List[np.ndarray]): List of histograms with the same shape and binning

        Returns:
        np.ndarray: Average histogram with the same shape as input histograms

        Raises:
        ValueError: If the input list is empty or histograms have different shapes
        """
        if not histograms:
            raise ValueError("Input list of histograms is empty")

        # Check that all histograms have the same shape
        first_shape = histograms[0].shape
        if not all(hist.shape == first_shape for hist in histograms):
            raise ValueError("All histograms must have the same shape")

        # Convert list to numpy array for efficient computation
        hist_array = np.array(histograms)

        # Calculate mean along first axis (across histograms)
        average = np.mean(hist_array, axis=0)

        return average

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

        fig, (ax, ax2) = PlotStyle.create_figure(n_row=2, n_col=1, figsize=(8, 8))
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
            alpha=0.9
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
