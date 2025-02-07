import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import json
from matplotlib.ticker import ScalarFormatter

from src.utils.datautil import arr_handler, iterwgt
from src.analysis.objutil import Object
from src.utils.filesysutil import FileSysHelper, pjoin, pdir, pbase
from src.utils.cutflowutil import load_csvs

class PlotStyle:
    """Handles basic plot styling and setup"""
    COLORS = list(mpl.colormaps['Set2'].colors)
    
    @staticmethod
    def setup_cms_style(ax, lumi=None, year=None):
        """Apply CMS style to an axis"""
        hep.cms.label(ax=ax, loc=2, label='Work in Progress', 
                     fontsize=11, com=13.6, lumi=lumi, year=year)
        ax.tick_params(axis='both', which='major', labelsize=10, length=0)
        
    @staticmethod
    def create_figure(n_row=1, n_col=1, figsize=None):
        """Create figure with default sizing"""
        if figsize is None:
            figsize = (8*n_col, 5*n_row)
        return plt.subplots(n_row, n_col, figsize=figsize)
    
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

class CSVPlotter:
    """Simplified plotter for CSV data"""
    def __init__(self, outdir):
        self.outdir = outdir
        FileSysHelper.checkpath(outdir)
        self.meta_dict = None
        self.data_dict = {}
        self.sig_group = None
        self.bkg_group = None
        
    def load_metadata(self, metadata_path):
        """Load metadata from JSON file"""
        with open(metadata_path, 'r') as f:
            self.meta_dict = json.load(f)
        self.labels = list(self.meta_dict.keys())
    
    def __set_group(self, sig_group, bkg_group):
        """Set the signal and background groups."""
        self.sig_group = sig_group
        self.bkg_group = bkg_group
    
    def __addextcf(self, cutflow: 'dict', df, ds, wgtname) -> None:
        """Add the cutflow to the dictionary to be udpated to the cutflow table.
        
        Parameters
        - `cutflow`: the cutflow dictionary to be updated"""
        cutflow[f'{ds}_raw'] = len(df)
        cutflow[f'{ds}_wgt'] = df[wgtname].sum()
    
    def __get_rwgt_fac(self, group, ds, signals, factor, luminosity) -> float:
        """Get the reweighting factor for the dataset (xsection * lumi)
        
        - `group`: the group of the dataset
        - `ds`: the dataset name
        - `signals`: the signal groups
        - `factor`: the factor to multiply to the flat weights
        - `luminosity`: the luminosity (in pb^-1)"""
        if group in signals: 
            multiply = factor
            print("Dataset is a signal.")
            print(f"Multiplying by {factor}")
        else:
            multiply = 1
        flat_wgt = 1/self.meta_dict[group][ds]['nwgt'] * multiply * self.meta_dict[group][ds]['xsection'] * luminosity
        return flat_wgt
    
    def process_datasets(self, datasource, metadata_path, postp_output, per_evt_wgt='Generator_weight', extraprocess=False, selname='Pass', signals=['ggF'], sig_factor=100, luminosity=41.5) -> pd.DataFrame:
        """Reweight the datasets to the desired xsection * luminosity by adding a column `weight` and save the processed dataframes to csv files.
        
        Parameters
        - `datasource`: the directory of the group subdirectories containing different csv output files.
        - `postp_output`: the directory to save the cutflows for post-processing
        - `per_evt_wgt`: the weight to be multiplied to the flat weights
        - `extraprocess`: additional processing to be done on the dataframe
        - `luminosity`: the luminosity in pb^-1
        
        Return 
        - `grouped`: the concatenated dataframe"""
        FileSysHelper.checkpath(postp_output)
        self.load_metadata(metadata_path)
        list_of_df = []

        for group in self.labels:
            load_dir = pjoin(datasource, group)
            if not FileSysHelper.checkpath(load_dir, createdir=False):
                continue
                
            cf_dict, cf_df = self.__process_group(
                group, load_dir, postp_output, per_evt_wgt,
                extraprocess, selname, signals, sig_factor, luminosity
            )
            
            self.__save_cutflow(cf_df, cf_dict, selname, postp_output, group)
            
            if cf_dict:
                list_of_df.extend([df for df in self.data_dict.get(group, {}).values() if df is not None])

        return pd.concat(list_of_df, axis=0).reset_index().drop('index', axis=1)
    
    def __process_group(self, group, load_dir, postp_output, per_evt_wgt, extraprocess, selname, signals, sig_factor, luminosity):
        """Process a single group of datasets."""
        cf_dict = {}
        cf_df = load_csvs(load_dir, f'{group}*cf.csv')[0]
        self.data_dict[group] = {}
        
        for ds in self.meta_dict[group].keys():
            df = self.__process_dataset(
                group, ds, load_dir, postp_output,
                per_evt_wgt, extraprocess, signals,
                sig_factor, luminosity
            )
            
            dsname = self.meta_dict[group][ds]['shortname']
            if df is not None:
                self.data_dict[group][dsname] = df
                self.__addextcf(cf_dict, df, dsname, per_evt_wgt)
            else:
                cf_dict[f'{dsname}_raw'] = 0
                cf_dict[f'{dsname}_wgt'] = 0
                
        return cf_dict, cf_df
    
    def __process_dataset(self, group, ds, load_dir, postp_output, per_evt_wgt, extraprocess, signals, sig_factor, luminosity):
        """Process a single dataset."""
        rwfac = self.__get_rwgt_fac(group, ds, signals, sig_factor, luminosity)
        dsname = self.meta_dict[group][ds]['shortname']
        
        def add_wgt(dfs, rwfac, ds, group):
            df = dfs[0]
            if df.empty:
                return None
            df['weight'] = df[per_evt_wgt] * rwfac
            df['dataset'] = ds
            df['group'] = group
            return extraprocess(df) if extraprocess else df
        
        FileSysHelper.checkpath(f'{postp_output}/{group}')
        return load_csvs(load_dir, f'{dsname}*out*', func=add_wgt, rwfac=rwfac, ds=dsname, group=group)
    
    def __save_cutflow(self, cf_df, cf_dict, selname, postp_output, group):
        """Save the cutflow dataframe to a CSV file."""
        if cf_dict:
            cf_df = pd.concat([cf_df, pd.DataFrame(cf_dict, index=[selname])])
            cf_df.to_csv(pjoin(postp_output, group, f'{group}_{selname.replace(" ", "")}_cf.csv'))

    @iterwgt
    def getdata(self, process, ds, file_type='.root'):
        """Returns the root files for the datasets."""
        result = FileSysHelper.glob_files(self._datadir, filepattern=f'{ds}*{file_type}')
        if result: 
            rootfile = result[0]
            if not process in self.data_dict: 
                self.data_dict[process] = {}
            if rootfile: self.data_dict[process][ds] = rootfile
        else:
            raise FileNotFoundError(f"Check if there are any files of specified pattern in {self._datadir}.")
   
    def get_hist(self, evts: 'pd.DataFrame', att, options, group: 'dict'=None, **kwargs) -> tuple[list, list, list[int, int], list, list]:
        """Histogram an attribute of the object for the given dataframe for groups of datasets. 
        Return sorted histograms based on the total counts.
        
        Parameters
        - `group`: the group of datasets to be plotted. {groupname: [list of datasets]}
        - `kwargs`: additional arguments for the `ObjectPlotter.hist_arr` function
        
        Returns
        - `hist_list`: a sorted list of histograms
        - `bins`: the bin edges
        - `bin_range`: the range of the histogram
        - `pltlabel`: the sorted labels of the datasets
        - `b_colors`: the colors of the datasets"""
        histopts = options['hist']
        bins = histopts.get('bins', 40)
        bin_range = histopts.get('range', (0,200))
        pltlabel = list(group.keys()) if group is not None else self.labels
        hist_list = []
        b_colors = colors[0:len(pltlabel)]
        for label in pltlabel:
            proc_list = group[label] if group is not None else [label]
            thisdf = evts[evts['group'].isin(proc_list)]
            thishist, bins = ObjectPlotter.hist_arr(thisdf[att], bins, bin_range, thisdf['weight'], **kwargs)
            hist_list.append(thishist)
        
        assert len(hist_list) == len(pltlabel), "The number of histograms and labels must be equal."
        assert len(hist_list) == len(b_colors), "The number of histograms and colors must be equal."

        return hist_list, bins, bin_range, pltlabel, b_colors
    
    @staticmethod
    def get_order(hist_list) -> list:
        """Order the histograms based on the total counts."""
        total_counts = [np.sum(hist) for hist in hist_list]
        sorted_indx = np.argsort(total_counts)[::-1]
        return sorted_indx
    
    @staticmethod
    def order_list(list_of_obj, order) -> list:
        """Order the list of lists based on the order."""
        return [list_of_obj[i] for i in order]
    
    @staticmethod
    def plot_shape(list_of_evts: list[pd.DataFrame], labels, attridict: dict, ratio_ylabel, outdir, hist_ylabel='Normalized', title='', save_suffix='') -> None:
        """Compare the (normalized) shape of the object attribute for two different dataframes, with ratio panel attached.
        
        Parameters
        - `list_of_evts`: the list of dataframes to be histogrammed and compared"""

        assert len(list_of_evts) >= 2, "The number of dataframes must be at least 2."
        assert 'weight' in list_of_evts[0].columns, "The weight column must be present in the dataframe."

        for att, options in attridict.items():
            pltopts = options['plot'].copy()
            fig, axs, ax2s = ObjectPlotter.set_style_with_ratio(title, pltopts.pop('xlabel', ''), hist_ylabel, ratio_ylabel)
            ax = axs[0]
            ax2 = ax2s[0]
            hist_list = []
            bins = options['hist']['bins']
            bin_range = options['hist']['range']
            wgt_list = []
            for evts in list_of_evts:
                thishist, bins = ObjectPlotter.hist_arr(evts[att], bins, bin_range, evts['weight'], density=False, keep_overflow=False)
                hist_list.append(thishist)
                wgt_list.append(evts['weight'])
            ObjectPlotter.plot_var_with_err(ax, ax2, hist_list, wgt_list, bins, labels, bin_range, **pltopts)

            fig.savefig(pjoin(outdir, f'{att}{save_suffix}.png'), dpi=400, bbox_inches='tight')
    
    def plot_SvBHist(self, ax, evts, att, attoptions, **kwargs) -> list:
        """Plot the signal and background histograms.
        
        Parameters
        - `ax`: matplotlib axis object
        - `att`: the attribute to be plotted
        - `attoptions`: the options for the attribute to be plotted
        
        Return
        - `order`: the order of the histograms"""
        b_hists, bins, x_range, blabels, _ = self.get_hist(evts, att, attoptions, self.bkg_group)
        order = kwargs.pop('order', CSVPlotter.get_order(b_hists))
        b_hists, blabels = CSVPlotter.order_list(b_hists, order), CSVPlotter.order_list(blabels, order)
        
        if self.sig_group is not None:
            s_hists, bins, x_range, slabels, _ = self.get_hist(evts, att, attoptions, self.sig_group, **kwargs)
            ObjectPlotter.plotSigVBkg(ax, s_hists, b_hists, bins, slabels, blabels, x_range, **kwargs)  
        else:
            ObjectPlotter.plot_var(ax, b_hists, bins, blabels, x_range, **kwargs)

        return order
    
    def plot_SvB(self, evts, attridict, bgroup, sgroup, title='', save_name='', lumi=220, **kwargs):
        """Plot the signal and background histograms."""
        self.__set_group(sgroup, bgroup)
        for att, options in attridict.items():
            xlabel = options['plot'].get('xlabel', '')
            fig, ax = ObjectPlotter.set_style('', xlabel, title, lumi=lumi)
            self.plot_SvBHist(ax[0], evts, att, options, **kwargs)
            fig.savefig(pjoin(self.outdir, f'{att}{save_name}.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)

    def plot_fourRegions(self, regionA, regionB, regionC, regionD, attridict, bgroup, sgroup, title='', save_name='', lumi=220, **kwargs):
        """Plot the signal and background histograms for the four regions."""
        self.__set_group(sgroup, bgroup)
        for att, options in attridict.items():
            xlabel = options['plot'].get('xlabel', '')
            subtitles = ['Region A (OS, 2b)', 'Region B (OS, 1b)', 'Region C (SS, 2b)', 'Region D (SS, 1b)']
            fig, axes = ObjectPlotter.set_style(title, [xlabel] * 4, subtitles, n_row=2, n_col=2, lumi=lumi)

            order = self.plot_SvBHist(ax=axes[0,0], evts=regionA, att=att, attoptions=options,  **kwargs)
            self.plot_SvBHist(ax=axes[0,1], evts=regionB, att=att, attoptions=options, order=order, **kwargs)
            self.plot_SvBHist(ax=axes[1,0], evts=regionC, att=att, attoptions=options, order=order, **kwargs)
            self.plot_SvBHist(ax=axes[1,1], evts=regionD, att=att, attoptions=options, order=order, **kwargs)
                

            fig.savefig(pjoin(self.outdir, f'{att}{save_name}.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
        
class ObjectPlotter():
    def __init__(self):
        pass
    
    @staticmethod
    def set_style_with_ratio(title: 'str', x_label, top_ylabel, bottom_ylabel, num_figure=1, set_log=False, lumi=None):
        """Set the style of the plot to CMS HEP with ratio plot as bottom panel.
        
        Parameters
        - `title`: the title of the plot
        - `x_label`: (str or list of str) the xlabel of the plot
        - `top_ylabel`: the ylabel of the top panel
        - `bottom_ylabel`: the ylabel of the bottom panel
        - `num_figure`: the number of figures (each with two panels top-down) to be plotted"""
        fig = plt.figure(figsize=(8*num_figure, 6))
        fig.suptitle(title, fontsize=14)
        hep.style.use("CMS")
        gs = fig.add_gridspec(2, 1*num_figure, height_ratios=(4, 1))
        ax2s = [None] * num_figure
        axs = [None] * num_figure
        x_label = [x_label] if not isinstance(x_label, list) else x_label
        top_ylabel = [top_ylabel] if not isinstance(top_ylabel, list) else top_ylabel
        bottom_ylabel = [bottom_ylabel] if not isinstance(bottom_ylabel, list) else bottom_ylabel
        for i in range(num_figure):
            ax2s[i] = fig.add_subplot(gs[1,i])
            axs[i] = fig.add_subplot(gs[0,i], sharex=ax2s[i])
            hep.cms.label(ax=axs[i], loc=2, label='Work in Progress', fontsize=11, com=13.6, lumi=lumi)
            axs[i].set_ylabel(top_ylabel[i], fontsize=12)
            axs[i].tick_params(axis='both', which='major', labelsize=10, length=0, labelbottom=False)
            ax2s[i].yaxis.get_offset_text().set_fontsize(11)  # Adjust the font size as needed
            ax2s[i].set_ylabel(bottom_ylabel[i], fontsize=11)
            ax2s[i].set_xlabel(x_label[i], fontsize=12)
            ax2s[i].tick_params(axis='both', which='major', labelsize=10, length=0)
            if set_log: ax2s[i].set_yscale('log')

        fig.subplots_adjust(hspace=0.15)
        
        return fig, axs, ax2s


    @staticmethod
    def set_style(title, xlabel='', subtitles='', n_row=1, n_col=1, year=None, lumi=220) -> tuple:
        """Set the style of the plot to CMS HEP.
        
        Return 
        - `fig`: the figure object
        - `axes`: the axes object"""

        fig, axes = plt.subplots(n_row, n_col, figsize=(8*n_col, 5*n_row))
        fig.suptitle(title, fontsize=14)
        hep.style.use("CMS")

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
            xlabel = [xlabel] if isinstance(xlabel, str) else xlabel
            subtitles = [subtitles] if isinstance(subtitles, str) else subtitles

        for i, ax in enumerate(axes.flat):
            hep.cms.label(ax=ax, loc=2, label='Work in Progress', fontsize=11, com=13.6, lumi=lumi, year=year)
            ax.set_xlabel(xlabel[i], fontsize=12)
            ax.set_ylabel("Events", fontsize=12)
            ax.set_prop_cycle('color', colors)
            ax.tick_params(axis='both', which='major', labelsize=10, length=0)
            ax.set_title(subtitles[i], fontsize=12)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
        return fig, axes

    @staticmethod
    def plot_var(ax, hist, bin_edges: np.ndarray, label, xrange, **kwargs):
        """A wrapper around the histplot function in mplhep to plot the histograms.
        
        Parameters
        - `hist`: np.ndarray object as histogram, or a list of histograms
        - `bin_edges`: the bin edges
        """
        hep.histplot(hist, bins=bin_edges, label=label, ax=ax, linewidth=1.5, **kwargs)
        ax.legend(fontsize=12, loc='upper right')
        ax.set_xlim(*xrange)
        ax.legend(fontsize=12, loc='upper right')
    
    @staticmethod
    def plot_var_with_err(ax, ax2, hist_list, wgt_list, bins, label, xrange, **kwargs):
        """Plot the histograms with error bars in a second panel.
        
        Parameters
        - `hist_list`: the list of histograms (un-normalized)
        - `wgt_list`: the list of weights array"""
        bin_width = bins[1] - bins[0]

        color = colors[0:len(hist_list)]

        norm_hist_list = []
        norm_err_list = []
        for hist, wgt in zip(hist_list, wgt_list):
            norm_hist_list.append(hist / (np.sum(wgt) * bin_width))
            norm_err_list.append(np.sqrt(hist) / (np.sum(wgt) * bin_width))
    
        ObjectPlotter.plot_var(ax, norm_hist_list, bins, label, xrange, histtype='step', alpha=1.0, color=color, **kwargs) 

        np.seterr(divide='ignore', invalid='ignore')

        error_x = (bins[:-1] + bins[1:])/2

        if len(hist_list) == 2:
            ratio = np.nan_to_num(hist_list[1] / hist_list[0], nan=0., posinf=0.)
            ratio_err = np.nan_to_num(ratio * np.sqrt((norm_err_list[0]/norm_hist_list[0])**2 + (norm_err_list[1]/norm_hist_list[1])**2), nan=0., posinf=0.)
            ax2.errorbar(error_x, ratio, yerr=ratio_err, markersize=3, fmt='o', color='black')
        else:
            for i in range(1, len(hist_list)):
                ratio = np.nan_to_num(hist_list[i] / hist_list[0], nan=0., posinf=0.)
                ratio_err = np.nan_to_num(ratio * np.sqrt((norm_err_list[i]/norm_hist_list[i])**2 + (norm_err_list[0]/norm_hist_list[0])**2), nan=0., posinf=0.)
                ax2.errorbar(error_x, ratio, yerr=ratio_err, markersize=3, fmt='o', color=color[i])
        ax2.axhline(1, color='gray', linestyle='--', linewidth=1)
        ax2.set_ylim(0.5, 1.5)
    
    @staticmethod
    def plotSigVBkg(ax, sig_hists, bkg_hists, bin_edges, sig_label, bkg_label, xrange, **kwargs):
        """Plot the signal and background histograms.
        
        Parameters
        - `ax`: the axis to be plotted
        - `sig_hists`: the signal histograms
        - `bkg_hists`: the background histograms
        - `bin_edges`: the bin edges
        """
        s_colors = ['red', 'blue', 'forestgreen']
        hep.histplot(bkg_hists, bins=bin_edges, label=bkg_label, ax=ax, histtype='fill', alpha=0.6, stack=True, linewidth=1)
        hep.histplot(sig_hists, bins=bin_edges, ax=ax, color=s_colors[0: len(sig_hists)], label=sig_label, stack=False, histtype='step', alpha=1.0, linewidth=1.5)
        ax.set_xlim(*xrange)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=12, loc='upper right')
        
    @staticmethod
    def hist_arr(arr, bins: int, range: list[int, int], weights=None, density=False, keep_overflow=True) -> tuple[np.ndarray, np.ndarray]:
        """Wrapper around numpy histogram function to deal with overflow.
        
        Parameters
        - `arr`: the array to be histogrammed
        - `bin_no`: number of bins
        - `range`: range of the histogram
        - `weights`: the weights of the array
        """
        if isinstance(bins, int):
            bins = np.linspace(*range, bins+1)
            min_edge = bins[0]
            max_edge = bins[-1]
            if keep_overflow: adjusted_data = np.clip(arr, min_edge, max_edge)
            else: adjusted_data = arr
        else:
            adjusted_data = arr
        hist, bin_edges = np.histogram(adjusted_data, bins=bins, weights=weights, density=density)
        return hist, bin_edges
            
    @staticmethod
    def sortobj(data, sort_by, sort_what, **kwargs):
        """Return an awkward array representation of the sorted attribute in data.
        
        Parameters
        - `sort_by`: the attribute to sort by
        - `sort_what`: the attribute to be sorted
        - `kwargs`: additional arguments for sorting
        """
        mask = Object.sortmask(data[sort_by], **kwargs)
        return arr_handler(data[sort_what])[mask]