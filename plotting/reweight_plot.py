from src.plotting.visutil import CSVPlotter, ObjectPlotter
import pandas as pd
import os
pjoin = os.path.join

class RwgtPlotter(CSVPlotter):
    def __init__(self, outdir, training_dir):
        super().__init__(outdir)
        self.train_dir = training_dir
    
    def load_rwgt(self, names):
        """Load the reweighted data.
        
        Parameters:
        - `names`: Names of the reweighted data, in the order of A, B, C, D."""
        self.dfA = pd.read_csv(pjoin(self.train_dir, names[0]))
        self.dfB = pd.read_csv(pjoin(self.train_dir, names[1]))
        self.dfC = pd.read_csv(pjoin(self.train_dir, names[2]))
        self.dfD = pd.read_csv(pjoin(self.train_dir, names[3]))
    
    def load_ori(self, ori_dfB_name, ori_dfD_name):
        """Load the original data."""
        self.ori_dfB = pd.read_csv(pjoin(self.train_dir, ori_dfB_name))
        self.ori_dfD = pd.read_csv(pjoin(self.train_dir, ori_dfD_name))
    
    def plot_rwgt_vars(self, att):
        """Plot the reweighted variables in separate panels.
        
        Parameters:
        - `att`: Attributes to plot."""
        ori_dfD = self.ori_dfD
        ori_dfB = self.ori_dfB
        dfA = self.dfA
        dfB = self.dfB
        dfC = self.dfC
        dfD = self.dfD

        plt_dir = self.outdir

        self.plot_shape([ori_dfD, dfC], ['Original Region D(1b)', 'Region C (2b)'], att, 'dfD/dfC', plt_dir, 
                title='SS Control Region (Before Reweighting)', save_suffix='oriCD')
        self.plot_shape([dfD, dfC], ['Reweighted Region D (1b)', 'Region C (2b)'], att, 'dfD/dfC', plt_dir, 
                title='SS Control Region (After Reweighting)', save_suffix='rwgtCD')
        self.plot_shape([ori_dfB, dfA], ['Original MR (1b)', 'SR (2b)'], att, 'dfB/dfA', plt_dir, 
                title='OS Control Region (Before Reweighting)', save_suffix='oriAB')
        self.plot_shape([dfB, dfA], ['Reweighted MR (1b)', 'SR (2b)'], att, 'dfB/dfA', plt_dir, 
                title='OS Control Region (After Reweighting)', save_suffix='rwgtAB')
    
    def plot_contrast(self, att_dict, ori_df, rwgt_df, tar_df, legend, ratio_label, title, save_suffix):
        """Plot the contrast between the original, reweighted, and target data.
        
        Parameters:
        - `att_dict`: Attributes to plot.
        - `ori_df`: Original data.
        - `rwgt_df`: Reweighted data.
        - `tar_df`: Target data.
        - `title`: Title of the plot.
        - `save_suffix`: Suffix to save the plot."""
        for att, options in att_dict.items():
            pltopts = options['plot'].copy()
            xlabels = [pltopts.pop('xlabel', '')] * 2
            ylabels = ['Normalized'] * 2
            r_labels = [ratio_label] * 2
            fig, axs, ax2s = ObjectPlotter.set_style_with_ratio(title, xlabels, ylabels, r_labels, num_figure=2)
            bins = options['hist']['bins']
            bin_range = options['hist']['range']
            ori_hist, bins = ObjectPlotter.hist_arr(ori_df[att], bins=bins, range=bin_range, weights=ori_df['weight'], density=False, keep_overflow=False)
            rwgt_hist, bins = ObjectPlotter.hist_arr(rwgt_df[att], bins=bins, range=bin_range, weights=rwgt_df['weight'], density=False, keep_overflow=False)
            tar_hist, bins = ObjectPlotter.hist_arr(tar_df[att], bins=bins, range=bin_range, weights=tar_df['weight'], density=False, keep_overflow=False)
            ObjectPlotter.plot_var_with_err(axs[0], ax2s[0], [ori_hist, tar_hist], [ori_df['weight'], tar_df['weight']], bins, [legend[0], legend[2]], bin_range, **pltopts)
            ObjectPlotter.plot_var_with_err(axs[1], ax2s[1], [rwgt_hist, tar_hist], [rwgt_df['weight'], tar_df['weight']], bins, [legend[1], legend[2]], bin_range, **pltopts)

            fig.savefig(pjoin(self.outdir, f'{att}_{save_suffix}.png'), dpi=350, bbox_inches='tight')
    
    def plot_contrast_ABCD(self, att_dict):
        """Plot the contrast between the original, reweighted, and target data for regions A, B, C, and D.
        
        Parameters:
        - `att_dict`: Attributes to plot."""
        ori_dfD = self.ori_dfD
        ori_dfB = self.ori_dfB
        dfA = self.dfA
        dfB = self.dfB
        dfC = self.dfC
        dfD = self.dfD

        self.plot_contrast(att_dict, ori_dfD, dfD, dfC, ['Original Region D', 'Reweighted Region D', 'Region C'], r'$N_D/N_C$', 'SS Control Region', 'CD')
        self.plot_contrast(att_dict, ori_dfB, dfB, dfA, ['Original Region B', 'Reweighted Region B', 'Region A'], r'$N_B/N_A$', 'OS Control Region', 'AB')
    
    def plot_vars_one_figure(self, att):
        """Plot the reweighted variables in the same figure.
        
        Parameters:
        - `att`: Attributes to plot."""
        ori_dfD = self.ori_dfD
        ori_dfB = self.ori_dfB
        dfA = self.dfA
        dfB = self.dfB
        dfC = self.dfC
        dfD = self.dfD

        plt_dir = self.outdir

        self.plot_shape([dfC, ori_dfD, dfD], ['Region C (2b)', 'Original Region D(1b)', 'Reweighted Region D'], att, 'prediction/data', plt_dir, 
                title='SS Control Region', save_suffix='CD')
        self.plot_shape([dfA, ori_dfB, dfB], ['Region A (2b)', 'Original Region B(1b)', 'Reweighted Region B'], att, 'prediction/data', plt_dir, 
                title='OS Control Region', save_suffix='AB')

        
        
        
    
    