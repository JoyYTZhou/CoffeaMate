import mplhep as hep
import numpy as np
import pandas as pd
import json
from matplotlib.ticker import ScalarFormatter

from src.utils.datautil import arr_handler, iterwgt
from src.analysis.objutil import Object
from src.utils.filesysutil import FileSysHelper, pjoin, pdir, pbase
from src.utils.cutflowutil import load_csvs
from src.utils.plotutil import HistogramHelper, PlotStyle

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
        - `evts`: DataFrame containing the events to histogram
        - `att`: attribute to histogram
        - `options`: dictionary containing histogram options
        - `group`: the group of datasets to be plotted. {groupname: [list of datasets]}
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
            thisdf = evts[evts['group'].isin(proc_list)]
            
            counts, edges = HistogramHelper.make_histogram(
                data=thisdf[att],
                bins=bins,
                range=bin_range,
                weights=thisdf['weight'],
                density=kwargs.get('density', False)
            )
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
                ratio_ylabel: str, outdir: str, hist_ylabel: str = 'Normalized', 
                title: str = '', save_suffix: str = '') -> None:
        """Compare normalized shapes of distributions with ratio panels.
        
        Parameters
        ----------
        list_of_evts : list[pd.DataFrame]
            List of DataFrames to compare
        labels : list
            Labels for each DataFrame in the comparison
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
        outdir : str
            Directory to save plots
        hist_ylabel : str, optional
            Label for histogram y-axis
        title : str, optional
            Plot title
        save_suffix : str, optional
            Suffix for saved files
        """
        # Input validation
        if len(list_of_evts) < 2:
            raise ValueError("Need at least 2 DataFrames to compare")
        if not all('weight' in df.columns for df in list_of_evts):
            raise ValueError("All DataFrames must have 'weight' column")

        styles = [
            {'histtype': 'fill', 'alpha': 0.4, 'linewidth': 1.5},
            {'histtype': 'step', 'alpha': 1.0, 'linewidth': 1},
            {'histtype': 'step', 'alpha': 1.0, 'linewidth': 1}
        ]

        for attr, options in attridict.items():
            fig, axs, ax2s = PlotStyle.create_ratio_figure(
                title=title,
                x_label=options['plot'].get('xlabel', ''),
                top_ylabel=hist_ylabel,
                bottom_ylabel=ratio_ylabel
            )
            
            hist_list = []
            wgt_list = []
            for df in list_of_evts:
                counts, edges = HistogramHelper.make_histogram(
                    data=df[attr],
                    bins=options['hist']['bins'],
                    range=options['hist']['range'],
                    weights=df['weight']
                )
                hist_list.append(counts)
                wgt_list.append(df['weight'])
            
            ObjectPlotter.plot_var_with_err(
                ax=axs[0],
                ax2=ax2s[0],
                hist_list=hist_list,
                wgt_list=wgt_list,
                bins=edges,
                label=labels,
                xrange=options['hist']['range'],
                styles=styles,
            )

            fig.savefig(
                pjoin(outdir, f'{attr}{save_suffix}.png'),
                dpi=400,
                bbox_inches='tight'
            )

    def plot_SvBHist(self, ax, evts, att, attoptions, **kwargs) -> list:
        """Plot the signal and background histograms."""
        b_hists, bins, x_range, blabels, _ = self.get_hist(
            evts, att, attoptions, self.bkg_group
        )
        
        order = kwargs.pop('order', CSVPlotter.get_order(b_hists))
        b_hists, blabels = CSVPlotter.order_list(b_hists, order), CSVPlotter.order_list(blabels, order)
        
        if self.sig_group is not None:
            # Get signal histograms
            s_hists, bins, x_range, slabels, _ = self.get_hist(
                evts, att, attoptions, self.sig_group, **kwargs
            )
            ObjectPlotter.plotSigVBkg(
                ax=ax,
                sig_hists=s_hists,
                bkg_hists=b_hists,
                bin_edges=bins,
                sig_label=slabels,
                bkg_label=blabels,
                xrange=x_range,
                **kwargs
            )
        else:
            ObjectPlotter.plot_var(
                ax=ax,
                hist=b_hists,
                bin_edges=bins,
                label=blabels,
                xrange=x_range,
                **kwargs
            )

        return order
    
    def plot_SvB(self, evts, attridict, bgroup, sgroup, title='', save_name='', lumi=220, **kwargs):
        """Plot the signal and background histograms."""
        self.__set_group(sgroup, bgroup)
        
        for att, options in attridict.items():
            xlabel = options['plot'].get('xlabel', '')
            fig, axes = PlotStyle.create_figure()
            PlotStyle.setup_cms_style(axes, lumi=lumi)
            PlotStyle.setup_axis(axes, xlabel=xlabel, title=title)  # Added title parameter here
            self.plot_SvBHist(axes, evts, att, options, **kwargs)
            
            fig.savefig(
                pjoin(self.outdir, f'{att}{save_name}.png'),
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1
            )

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
    def plot_var(ax, hist, bin_edges: np.ndarray, label, xrange, **kwargs):
        """Plot histograms on an axis"""
        hep.histplot(hist, bins=bin_edges, label=label, ax=ax, linewidth=1.5, **kwargs)
        ax.legend(fontsize=12, loc='upper right')
        ax.set_xlim(*xrange)

    @staticmethod
    def plot_var_with_err(ax, ax2, hist_list, wgt_list, bins, label, xrange, **kwargs):
        """Plot multiple histograms with error bars and ratio panel.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Main plot axes
        ax2 : matplotlib.axes.Axes
            Ratio panel axes
        hist_list : list
            List of histograms to plot
        wgt_list : list
            List of weights for each histogram
        bins : array-like
            Bin edges
        label : list
            Labels for each histogram
        xrange : tuple
            (xmin, xmax) for plot range
        **kwargs : dict
            Additional plotting parameters including 'styles' for individual histogram styling
        """
        bin_width = bins[1] - bins[0]
        colors = kwargs.pop('colors', PlotStyle.COLORS[:len(hist_list)])
        styles = kwargs.pop('styles', None) or [{'histtype': 'step', 'alpha': 1.0}] * len(hist_list)

        normalized_data = [
            HistogramHelper.normalize_histogram(hist, wgt, bin_width)
            for hist, wgt in zip(hist_list, wgt_list)
        ]
        norm_hist_list, norm_err_list = zip(*normalized_data)

        for hist, style, lbl, color in zip(norm_hist_list, styles, label, colors):
            hep.histplot(
                hist,
                bins=bins,
                label=lbl,
                ax=ax,
                color=color,
                **style,
                **kwargs
            )

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
    def plotSigVBkg(ax, sig_hists, bkg_hists, bin_edges, sig_label, bkg_label, xrange, **kwargs):
        """Plot signal and background histograms"""
        hep.histplot(
            bkg_hists, bins=bin_edges, label=bkg_label,
            ax=ax, histtype='fill', alpha=0.6,
            stack=True, linewidth=1
        )
        hep.histplot(
            sig_hists, bins=bin_edges, ax=ax,
            color=PlotStyle.SIGNAL_COLORS[:len(sig_hists)],
            label=sig_label, stack=False,
            histtype='step', alpha=1.0, linewidth=1.5
        )
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