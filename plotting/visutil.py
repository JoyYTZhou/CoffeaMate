import mplhep as hep
import numpy as np
import pandas as pd
import json, logging
from matplotlib.ticker import ScalarFormatter

from src.utils.datautil import DataLoader, iterwgt, arr_handler
from src.analysis.objutil import ObjectProcessor
from src.utils.filesysutil import FileSysHelper, pjoin, pdir, pbase
from src.utils.plotutil import HistogramHelper, PlotStyle

luminosity = {"2022PreEE": 41.5/2 * 1000, "2022PostEE": 41.5 * 1000/2, "2023Summer": 32.7 * 1000}
regroup_dict = {"Others": ['WJets', 'WZ', 'WW', 'WWW', 'ZZZ', 'WZZ', 'WWZ'], 'HH': ['ggF']}

class CSVPlotter:
    """Simplified plotter for CSV data.
    
    Attributes
    - `sig_group`: dictionary of {signal group label: [list of datasets]}
    - `bkg_group`: dictionary of {background group label: [list of datasets]}"""
    def __init__(self, outdir):
        self.outdir = outdir
        FileSysHelper.checkpath(outdir)
        self.meta_dict = None
        self.data_dict = {}
        self.sig_group = {r"b$\bar{b} \tau \tau \times 100$": ["ggF"]}
        self.bkg_group = {"DYJets": ["DYJets"], r"$t\bar{t}$": ['TTbar'], "SingleH": ["SingleH"], "Others": ["WZ", "WWW", "WW", "WWZ", "WZZ", "WJets", "Others", "ZH", "ZZ"]}
        self.data_label = 'Data'
        
    def load_metadata(self, metadata_path):
        """Load metadata from JSON file. This initializes the `meta_dict` attribute which contains dataset groups, xsection values, etc."""
        with open(metadata_path, 'r') as f:
            self.meta_dict = json.load(f)
        self.labels = list(self.meta_dict.keys())
        logging.info(f"Loaded metadata from {metadata_path} with groups: {self.labels}")
    
    def __set_group(self, sig_group, bkg_group):
        """Set the signal and background groups."""
        self.sig_group = sig_group
        self.bkg_group = bkg_group
    
    def __addextcf(self, cutflow: 'dict', df, ds, wgtname) -> None:
        """Add the cutflow to the dictionary to be udpated to the cutflow table.
        
        Parameters
        - `cutflow`: the cutflow dictionary to be updated"""
        cutflow[f'{ds}_raw'] = len(df)
        if wgtname in df.columns:
            cutflow[f'{ds}_wgt'] = df[wgtname].sum()
    
    def __get_rwgt_fac(self, group, ds, luminosity) -> float:
        """Calculate reweighting factor."""
        if group == 'Data':
            return 1
        xsection = self.meta_dict[group][ds].get('xsection', 1)
        nwgt = self.meta_dict[group][ds].get('nwgt')
        if nwgt is None:
            logging.warning(f"No nwgt for {group}/{ds}. Using flat weight.")
            return xsection * luminosity
        return (xsection * luminosity) / nwgt
    
    def process_datasets(self, datasource, metadata_path, postp_output, 
                         per_evt_wgt='Generator_weight', extraprocess=lambda df: df, 
                         selname='Pass', signals=['ggF'], luminosity=41.5) -> pd.DataFrame:
        """Reweight the datasets to the desired xsection * luminosity by adding a column `weight` and save the processed dataframes to csv files.
        This also saves the added cutflows (not weighted by xsection * luminosity) to a csv file.
        
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
                
            list_of_df.append(self.__process_group(group, load_dir, per_evt_wgt,
                extraprocess, luminosity))
        
        return pd.concat(list_of_df, axis=0).reset_index(drop=True)
    
    def __process_group(self, group: str, load_dir: str, per_evt_wgt: str, 
                        extraprocess: callable = lambda df: df, luminosity: float = 1.0) -> pd.DataFrame:
        """
        Process a single group of datasets by applying weights, loading data, and updating counters.

        Args:
            group (str): Name of the dataset group to process.
            load_dir (str): Directory containing input CSV files.
            postp_output (str): Directory to save processed outputs.
            per_evt_wgt (str): Column name for per-event weights.
            extraprocess (callable, optional): Function to apply additional processing to the DataFrame.
            luminosity (float, optional): Luminosity value for weight calculation.

        Returns:
            dict: Dictionary containing raw and weighted counters for each dataset.
        """
        self.data_dict[group] = {}

        def add_wgt(df):
            if df.empty: 
                return None
            for ds, meta in self.meta_dict[group].items():
                dsname = meta['shortname']
                rwfac = self.__get_rwgt_fac(group, ds, luminosity)
                if group != 'Data':
                    df.loc[df.dataset == dsname, 'weight'] = df.loc[df.dataset == dsname, per_evt_wgt] * rwfac
                else:
                    df.loc[df.dataset == dsname, 'weight'] = 1.0
            return df

        output_df = DataLoader.load_csvs(load_dir, f'{group}*out*', func=lambda dfs: extraprocess(add_wgt(dfs[0])))
        
        return output_df

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
    
    def get_hist(self, evts: 'pd.DataFrame', att:'str', options, group: 'dict' = None, rescale=1, **kwargs) -> tuple[list, list, list[int, int], list, list]:
        """Returns histograms for the given attribute in the dataframe, each for the specified group.
        
        Parameters
        - `att`: attribute to histogram. column in the dataframe
        - `options`: dictionary containing histogram options
        - `group`: the groups to be plotted with specific labels. {group label: [list of datasets]}
        - `kwargs`: additional histogram parameters
        
        Returns
        - `hist_list`: a sorted list of histograms
        - `bins`: the bin edges
        - `bin_range`: the range of the histogram
        - `pltlabel`: the sorted labels of the datasets
        - `b_colors`: the colors of the datasets
        """
        histopts = options['hist']
        bins = histopts.get('bins', 40)
        bin_range = histopts.get('range', (0, 200))
        
        pltlabel = list(group.keys()) if group is not None else self.labels
        b_colors = PlotStyle.COLORS[:len(pltlabel)]
        
        hist_list = []
        for label in pltlabel:
            proc_list = group[label] if group is not None else [label]
            # Filter the DataFrame for the current group
            thisdf = evts[evts['group'].isin(proc_list)]
            if thisdf.empty:
                logging.warning(f"No data for group {label}. Skipping.")
                continue
            if thisdf[att].isna().any():
                logging.warning(f"Attribute {att} is nan for group {label}. Skipping.")
                continue
            counts, edges = HistogramHelper.make_histogram(data=thisdf[att], bins=bins, range=bin_range,
                weights=thisdf['weight']*rescale, density=kwargs.get('density', False))
            hist_list.append(counts)
            bins = edges  # Update bins in case they were modified
            
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
    def plot_shape(list_of_evts: list[pd.DataFrame], labels: list, attridict: dict, 
                ratio_ylabel: str, outdir: str, hist_ylabel: str = 'Normalized', normalize: bool = True, 
                title: str = '', save_suffix: str = '') -> None:
        """Compare normalized shapes of distributions with ratio panels.
        
        Parameters
        ----------
        labels : list
            Labels for each histogram in the comparison
        attridict : dict
            Dictionary of attributes to plot with their options
            Format: {
                'attribute_name': {
                    'hist': {'bins': int, 'range': tuple},
                    'plot': {'xlabel': str, ...}
                }
            }
        ratio_ylabel : str
            Label for ratio panel y-axis
        title : str, optional
            Plot title
        """
        # Input validation
        if len(list_of_evts) < 2:
            raise ValueError("Need at least 2 DataFrames to compare")
        if not all('weight' in df.columns for df in list_of_evts):
            raise ValueError("All DataFrames must have 'weight' column")

        styles = [{'histtype': 'fill', 'alpha': 0.4, 'linewidth': 1.5},
            {'histtype': 'step', 'alpha': 1.0, 'linewidth': 1},
            {'histtype': 'step', 'alpha': 1.0, 'linewidth': 1}]

        for attr, options in attridict.items():
            fig, axs, ax2s = PlotStyle.create_figure(ratio_panel=True, title=title, x_label=options['plot'].get('xlabel', ''),
                                                     top_ylabel=hist_ylabel, bottom_ylabel=ratio_ylabel)
            
            hist_list = []
            wgt_list = []
            for df in list_of_evts:
                counts, edges = HistogramHelper.make_histogram(
                    data=df[attr], bins=options['hist']['bins'],
                    range=options['hist']['range'], weights=df['weight'])
                hist_list.append(counts)
                wgt_list.append(df['weight'].sum())
            
            ObjectPlotter.plot_hist_with_err(
                ax=axs[0], ax2=ax2s[0], hist_list=hist_list, wgt_list=wgt_list, normalize=normalize,
                bins=edges, label=labels, xrange=options['hist']['range'], styles=styles)

            fig.savefig(
                pjoin(outdir, f'{attr}{save_suffix}.png'),
                dpi=400, bbox_inches='tight')
    
    def get_dataMinusMCHist(self, evts, att, options) -> tuple[np.ndarray, np.ndarray, list[int, int]]:
        """Return a np.ndarray histogram of data minus MC for the given attribute."""
        b_hists, b_bins, x_range, _, _ = self.get_hist(evts, att, options, self.bkg_group, rescale=-1) 
        total_bhist = b_hists[0]
        for b_hist in b_hists[1:]:
            total_bhist += b_hist
        if self.sig_group is not None: 
            s_hists, s_bins, s_range, slabels, _ = self.get_hist(evts, att, options, self.sig_group, rescale=-1) 
        for s_hist in s_hists:
            total_bhist += s_hist
        data_hist, _, _, _, _ = self.get_hist(evts, att, options, {"Data": ["Data"]})

        return data_hist+total_bhist, b_bins, x_range
            
    def plot_dataMinusMC(self, evts, attridict, bgroup=None, sgroup=None, minus_sig=True, title='', save_name='', lumi=220, **kwargs):
        """Plot the data minus MC histograms."""
        if bgroup is not None and sgroup is not None:
            self.__set_group(sgroup, bgroup)
        
        for att, options in attridict.items():
            xlabel = options['plot'].get('xlabel', '')
            fig, axes = PlotStyle.create_figure()
            PlotStyle.setup_cms_style(axes, lumi=lumi)
            PlotStyle.setup_axis(axes, xlabel=xlabel, title=title)
            data_hist, bins, x_range = self.get_dataMinusMCHist(evts, att, options, minus_sig=minus_sig)
            ObjectPlotter.plot_var(
                ax=axes, hists=data_hist, bin_edges=bins, label=['Data - MC'],
                xrange=x_range, yerr=True
            )
            fig.savefig(pjoin(self.outdir, f'{att}_{save_name}_DataMinusMC.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.1)

    def __plot_SvBHist(self, ax, evts, att, attoptions, include_data=True, include_sig=True, stack_all=False, rescale_sig=100, **kwargs) -> list:
        """Plot the signal and background histograms."""
        b_hists, bins, x_range, blabels, _ = self.get_hist(evts, att, attoptions, self.bkg_group)
        if include_data:
            data_hists, _, _, _, _ = self.get_hist(evts, att, attoptions, {"Data": ["Data"]}, **kwargs)
        else:
            data_hists = None
        
        order = kwargs.pop('order', CSVPlotter.get_order(b_hists))
        b_hists, blabels = CSVPlotter.order_list(b_hists, order), CSVPlotter.order_list(blabels, order)
        
        if self.sig_group is not None and include_sig:
            s_hists, bins, x_range, slabels, _ = self.get_hist(evts, att, attoptions, self.sig_group, rescale=rescale_sig, **kwargs)
            ObjectPlotter.plotSigWBkg(ax=ax, sig_hists=s_hists, bkg_hists=b_hists, data_hist=data_hists, bin_edges=bins, sig_label=slabels, bkg_label=blabels, xrange=x_range, stack_all=stack_all, **kwargs)
        else:
            ObjectPlotter.plot_var(ax=ax, hists=b_hists, bin_edges=bins, label=blabels, xrange=x_range, **kwargs)

        return order
    
    def plot_SvB(self, evts, attridict, bgroup=None, sgroup=None, title='', save_name='', lumi=220, **kwargs):
        """Plot the signal and background histograms."""
        if bgroup is not None and sgroup is not None:
            self.__set_group(sgroup, bgroup)
        
        for att, options in attridict.items():
            xlabel = options['plot'].get('xlabel', '')
            fig, axes = PlotStyle.create_figure()
            PlotStyle.setup_cms_style(axes, lumi=lumi)
            PlotStyle.setup_axis(axes, xlabel=xlabel, title=title)  # Added title parameter here
            self.__plot_SvBHist(axes, evts, att, options, **kwargs)
            
            if save_name:
                save_name = f'_{save_name}'
            fig.savefig(pjoin(self.outdir, f'{att}{save_name}.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    def plot_fourRegions(self, regionA, regionB, regionC, regionD, attridict, bgroup, sgroup, title='', save_name='', lumi=220, **kwargs):
        """Plot the signal and background histograms for the four regions."""
        self.__set_group(sgroup, bgroup)
        
        regions = [regionA, regionB, regionC, regionD]
        subtitles = ['Region A (OS, 2b)', 'Region B (OS, 1b)',
                    'Region C (SS, 2b)', 'Region D (SS, 1b)']
        
        for att, options in attridict.items():
            xlabel = options['plot'].get('xlabel', '')
            fig, axes = PlotStyle.create_figure(n_row=2, n_col=2)

            if title: fig.suptitle(title)
            
            for ax, subtitle in zip(axes.flat, subtitles):
                PlotStyle.setup_cms_style(ax, lumi=lumi)
                PlotStyle.setup_axis(ax, xlabel=xlabel, title=subtitle)
            
            order = None
            for idx, region in enumerate(regions):
                ax = axes.flat[idx]
                order = self.plot_SvBHist(
                    ax=ax,
                    evts=region,
                    att=att,
                    attoptions=options,
                    order=order,
                    **kwargs
                )
            
            fig.savefig(
                pjoin(self.outdir, f'{att}{save_name}.png'),
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1
            )
    
        
class ObjectPlotter():
    @staticmethod
    def plot_var(ax, hists, bin_edges: np.ndarray, label, xrange, stack=True, **kwargs):
        """Plot histograms on an axis"""
        if stack:
            histtype = 'fill'
        else:
            histtype = 'step'
        hep.histplot(hists, bins=bin_edges, label=label, ax=ax, linewidth=1.5, alpha=0.7, histtype=histtype, stack=stack, **kwargs)
        ax.legend(fontsize=12, loc='upper right')
        ax.set_xlim(*xrange)

    @staticmethod
    def plot_hist_with_err(ax, ax2, hist_list, wgt_list, bins, label, xrange, normalize=False, **kwargs):
        """Plot multiple histograms with error bars and ratio panel.
        
        Parameters
        ----------
        wgt_list : list
            List of total weights of events in each histogram
        bins : array-like
            Bin edges
        label : list
            Labels for each histogram
        **kwargs : dict
            Additional plotting parameters including 'styles' for individual histogram styling
        """
        bin_width = bins[1] - bins[0]
        colors = kwargs.pop('colors', PlotStyle.COLORS[:len(hist_list)])
        styles = kwargs.pop('styles', None) or [{'histtype': 'step', 'alpha': 1.0}] * len(hist_list)
        
        if normalize:
            normalized_data = [
                HistogramHelper.normalize_histogram(hist, wgt, bin_width) for hist, wgt in zip(hist_list, wgt_list)
            ]
            norm_hist_list, norm_err_list = zip(*normalized_data)
        else:
            norm_hist_list = hist_list
            norm_err_list = [np.sqrt(hist) for hist in hist_list]

        for hist, style, lbl, color in zip(norm_hist_list, styles, label, colors):
            hep.histplot(hist, bins=bins, label=lbl, ax=ax, color=color, **style, **kwargs)

        ax.legend(fontsize=12, loc='upper right')
        ax.set_xlim(*xrange)

        error_x = (bins[:-1] + bins[1:]) / 2
        ObjectPlotter._plot_ratio_panel(ax2, norm_hist_list, norm_err_list, error_x, colors)

    @staticmethod
    def _plot_ratio_panel(ax2, norm_hist_list, norm_err_list, error_x, colors):
        """Plot the ratio panel"""
        if len(norm_hist_list) == 2:
            ratio, ratio_err = HistogramHelper.calc_ratio_and_errors(
                norm_hist_list[1], norm_hist_list[0],
                norm_err_list[1], norm_err_list[0]
            )
            ax2.errorbar(error_x, ratio, yerr=ratio_err, markersize=3, fmt='o', color='black', elinewidth=0.9)
        else:
            for i in range(1, len(norm_hist_list)):
                ratio, ratio_err = HistogramHelper.calc_ratio_and_errors(
                    norm_hist_list[i], norm_hist_list[0],
                    norm_err_list[i], norm_err_list[0]
                )
                ax2.errorbar(error_x, ratio, yerr=ratio_err, markersize=3, fmt='o', color=colors[i], elinewidth=0.9)

        ax2.axhline(1, color='gray', linestyle='--', linewidth=1)
        ax2.set_ylim(0.5, 1.5)
    
    @staticmethod
    def plotSigWBkg(ax, sig_hists, bkg_hists, data_hist, bin_edges, sig_label, bkg_label, xrange, stack_all=False, **kwargs):
        """Plot signal and background histograms"""
        if stack_all:
            total_hists = bkg_hists + sig_hists
            total_label = bkg_label + sig_label
            hep.histplot(total_hists, bins=bin_edges, label=total_label, ax=ax, histtype='fill', alpha=0.6, stack=True, linewidth=1)
        else:
            hep.histplot(bkg_hists, bins=bin_edges, label=bkg_label,
                ax=ax, histtype='fill', alpha=0.6, stack=True, linewidth=1)
            hep.histplot(sig_hists, bins=bin_edges, ax=ax,
                color=PlotStyle.SIGNAL_COLORS[:len(sig_hists)],
                label=sig_label, stack=False,
                histtype='step', alpha=1.0, linewidth=1.5)

        if data_hist is not None:
            hep.histplot(data_hist, bins=bin_edges, ax=ax, color='black', histtype='errorbar', xerr=True, label='Data', linewidth=1.5, **kwargs)
            
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
        mask = ObjectProcessor.sortmask(data[sort_by], **kwargs)
        return arr_handler(data[sort_what])[mask]