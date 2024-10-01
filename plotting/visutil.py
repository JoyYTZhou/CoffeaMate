import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import json
from matplotlib.ticker import ScalarFormatter

from src.utils.datautil import arr_handler, iterwgt
from src.analysis.objutil import Object
from src.utils.filesysutil import FileSysHelper, pjoin
from src.utils.cutflowutil import load_csvs

colors = list(mpl.colormaps['Set2'].colors)

class CSVPlotter():
    """Plot the histograms and other visualizations from the csv files.
    
    Attributes
    - `meta_dict`: the metadata dictionary
    - `data_dict`: the dictionary of dataframes"""
    def __init__(self, outdir):
        self.outdir = outdir
        FileSysHelper.checkpath(self.outdir)
    
    def __set_meta(self, metadata_path):
        with open(metadata_path, 'r') as f:
            self.meta_dict = json.load(f)
        self.data_dict = {}
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
    
    def __get_rwgt_fac(self, group, ds, signals, factor):
        if group in signals: 
            multiply = factor
        else:
            multiply = 1
        flat_wgt = self.meta_dict[group][ds]['per_evt_wgt'] * multiply
        return flat_wgt
    
    def postprocess_csv(self, datasource, metadata_path, per_evt_wgt='Generator_weight', extraprocess=False, selname='Pass', signals=['ggF'], sig_factor=100) -> pd.DataFrame:
        """Post-process the datasets and save the processed dataframes to csv files.
        
        Parameters
        - `fitype`: the file type to be saved
        - `per_evt_wgt`: the weight to be multiplied to the flat weights
        - `extraprocess`: additional processing to be done on the dataframe
        
        Return 
        - `grouped`: the concatenated dataframe"""
        list_of_df = []
        new_outdir = f'{datasource}_extrasel'
        FileSysHelper.checkpath(new_outdir)
        self.__set_meta(metadata_path)
        def add_wgt(dfs, rwfac, ds, group):
            df = dfs[0]
            if df.empty: return None
            df['weight'] = df[per_evt_wgt] * rwfac
            df['dataset'] = ds
            df['group'] = group 
            if extraprocess: return extraprocess(df)
            else: return df
        for group in self.labels:
            load_dir = pjoin(datasource, group) 
            cf_dict = {}
            cf_df = load_csvs(load_dir, f'{group}_cf')[0]
            for ds in self.meta_dict[group].keys():
                rwfac = self.__get_rwgt_fac(group, ds, signals, sig_factor)
                dsname = self.meta_dict[group][ds]['shortname']
                df = load_csvs(load_dir, f'{dsname}_out', func=add_wgt, rwfac=rwfac, ds=dsname, group=group)
                FileSysHelper.checkpath(f'{new_outdir}/{group}')
                if df is not None: 
                    list_of_df.append(df)
                    self.__addextcf(cf_dict, df, dsname, per_evt_wgt)
                else:
                    cf_dict[f'{dsname}_raw'] = 0
                    cf_dict[f'{dsname}_wgt'] = 0
            cf_df = pd.concat([cf_df, pd.DataFrame(cf_dict, index=[selname])])
            cf_df.to_csv(pjoin(new_outdir, group, f'{group}_{selname.replace(" ", "")}_cf.csv'))
        grouped = pd.concat(list_of_df, axis=0).reset_index().drop('index', axis=1)
        return grouped

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

    def plot_shape(self, list_of_evts: list[pd.DataFrame], labels, attridict: dict, ratio_ylabel, hist_ylabel='Normalized', title='', save_name='') -> None:
        """Compare the (normalized) shape of the object attribute for two different dataframes, with ratio panel attached.
        
        Parameters
        - `list_of_evts`: the list of dataframes to be histogrammed and compared"""

        assert len(list_of_evts) == 2, "The number of dataframes must be 2."
        assert 'weight' in list_of_evts[0].columns, "The weight column must be present in the dataframe."

        for att, options in attridict.items():
            pltopts = options['plot'].copy()
            fig, ax, ax2 = ObjectPlotter.set_style_with_ratio(title, pltopts.pop('xlabel', ''), hist_ylabel, ratio_ylabel)
            hist_list = []
            bins = options['hist']['bins']
            bin_range = options['hist']['range']
            wgt_list = []
            for evts in list_of_evts:
                thishist, bins = ObjectPlotter.hist_arr(evts[att], bins, bin_range, evts['weight'], density=False, keep_overflow=False)
                hist_list.append(thishist)
                wgt_list.append(evts['weight'])
            ObjectPlotter.plot_var_with_err(ax, ax2, hist_list, wgt_list, bins, labels, bin_range, **pltopts)

            fig.savefig(pjoin(self.outdir, f'{att}{save_name}.png'), dpi=400, bbox_inches='tight')
    
    def plot_SvBHist(self, ax, evts, att, attoptions, title, **kwargs) -> list:
        """Plot the signal and background histograms.
        
        Parameters
        - `ax`: matplotlib axis object
        - `att`: the attribute to be plotted
        - `attoptions`: the options for the attribute to be plotted
        
        Return
        - `order`: the order of the histograms"""
        xlabel = attoptions['plot'].get('xlabel', '')
        b_hists, bins, range, blabels, _ = self.get_hist(evts, att, attoptions, self.bkg_group)
        order = kwargs.pop('order', CSVPlotter.get_order(b_hists))
        b_hists, blabels = CSVPlotter.order_list(b_hists, order), CSVPlotter.order_list(blabels, order)

        s_hists, bins, range, slabels, _ = self.get_hist(evts, att, attoptions, self.sig_group, **kwargs)
        ObjectPlotter.plotSigVBkg(ax, s_hists, b_hists, bins, slabels, blabels, range, title, xlabel=xlabel, **kwargs)  

        return order

    def plot_fourRegions(self, regionA, regionB, regionC, regionD, attridict, bgroup, sgroup, title='', save_name='', **kwargs):
        """Plot the signal and background histograms for the four regions."""
        self.__set_group(sgroup, bgroup)
        for att, options in attridict.items():
            fig, axes = ObjectPlotter.set_style(title, '', n_row=2, n_col=2)

            order = self.plot_SvBHist(ax=axes[0,0], evts=regionA, att=att, attoptions=options, title='Region A (OS, 2 b-tagged)', **kwargs)

            self.plot_SvBHist(ax=axes[0,1], evts=regionB, att=att, attoptions=options, title='Region B (OS, 1 b-tagged)', order=order, **kwargs)
            self.plot_SvBHist(ax=axes[1,0], evts=regionC, att=att, attoptions=options, title='Region C (SS, 2 b-tagged)', order=order, **kwargs)
            self.plot_SvBHist(ax=axes[1,1], evts=regionD, att=att, attoptions=options, title='Region D (SS, 1 b-tagged)', order=order, **kwargs)

            fig.savefig(pjoin(self.outdir, f'{att}{save_name}.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
        
class ObjectPlotter():
    def __init__(self):
        pass
    
    @staticmethod
    def set_style_with_ratio(title, x_label, top_ylabel, bottom_ylabel, set_log=False, lumi=220):
        """Set the style of the plot to CMS HEP with ratio plot as bottom panel.
        
        Parameters
        - `top_ylabel`: the ylabel of the top panel"""
        fig = plt.figure(figsize=(8, 6))
        fig.suptitle(title, fontsize=14, y=0.92)
        hep.style.use("CMS")
        gs = fig.add_gridspec(2, 1, height_ratios=(4, 1))
        ax2 = fig.add_subplot(gs[1,0])
        ax = fig.add_subplot(gs[0,0], sharex=ax2)
        fig.subplots_adjust(hspace=0.1)

        hep.cms.label(ax=ax, loc=2, label='Work in Progress', fontsize=11, com=13.6, lumi=lumi)

        ax.set_ylabel(top_ylabel, fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10, length=0, labelbottom=False)

        ax2.yaxis.get_offset_text().set_fontsize(11)  # Adjust the font size as needed
        ax2.set_ylabel(bottom_ylabel, fontsize=11)
        ax2.set_xlabel(x_label, fontsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=10, length=0)
        if set_log: ax2.set_yscale('log')

        return fig, ax, ax2

    @staticmethod
    def set_style(title, xlabel='', n_row=1, n_col=1, year=None, lumi=220) -> tuple:
        """Set the style of the plot to CMS HEP.
        
        Return 
        - `fig`: the figure object
        - `axes`: the axes object"""

        fig, axes = plt.subplots(n_row, n_col, figsize=(8*n_col, 5*n_row))
        fig.suptitle(title, fontsize=14)
        hep.style.use("CMS")

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for ax in axes.flat:
            hep.cms.label(ax=ax, loc=2, label='Work in Progress', fontsize=11, com=13.6, lumi=lumi, year=year)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel("Events", fontsize=12)
            ax.set_prop_cycle('color', colors)
            ax.tick_params(axis='both', which='major', labelsize=10, length=0)
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
    
    @staticmethod
    def plot_var_with_err(ax, ax2, hist_list, wgt_list, bins, label, xrange, **kwargs):
        """Plot the histograms with error bars in a second panel.
        
        Parameters
        - `hist_list`: the list of histograms (un-normalized)
        - `wgt_list`: the list of weights array"""
        bin_width = bins[1] - bins[0]

        norm_hist_list = []
        norm_err_list = []
        for hist, wgt in zip(hist_list, wgt_list):
            norm_hist_list.append(hist / (np.sum(wgt) * bin_width))
            norm_err_list.append(np.sqrt(hist) / (np.sum(wgt) * bin_width))
    
        ratio = np.nan_to_num(hist_list[0] / hist_list[1], nan=0., posinf=0.)
        ratio_err = np.nan_to_num(ratio * np.sqrt((norm_err_list[0]/norm_hist_list[0])**2 + (norm_err_list[1]/norm_hist_list[1])**2), nan=0., posinf=0.)

        ObjectPlotter.plot_var(ax, norm_hist_list, bins, label, xrange, histtype='step', alpha=1.0, **kwargs) 

        np.seterr(divide='ignore', invalid='ignore')
        ax2.errorbar((bins[:-1] + bins[1:])/2, ratio, yerr=ratio_err, markersize=4, fmt='o', color='black')
    
    @staticmethod
    def plotSigVBkg(ax, sig_hists, bkg_hists, bin_edges, sig_label, bkg_label, xrange, title, **kwargs):
        """Plot the signal and background histograms.
        
        Parameters
        - `ax`: the axis to be plotted
        - `sig_hists`: the signal histograms
        - `bkg_hists`: the background histograms
        - `bin_edges`: the bin edges
        """
        s_colors = ['red', 'blue', 'forestgreen']
        hep.histplot(bkg_hists, bins=bin_edges, label=bkg_label, ax=ax, histtype='fill', alpha=0.6, stack=True, linewidth=1)
        hep.histplot(sig_hists, bins=bin_edges, ax=ax, color=s_colors, label=sig_label, stack=False, histtype='step', alpha=1.0, linewidth=1.5)
        ax.set_xlim(*xrange)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(kwargs.get('xlabel', ''))
        ax.set_title(title, fontsize=12)
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