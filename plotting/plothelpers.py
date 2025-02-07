from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import mplhep as hep

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