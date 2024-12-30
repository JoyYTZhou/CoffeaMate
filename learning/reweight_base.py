from matplotlib import pyplot as plt
import pandas as pd



class ReweighterBase():
    def __init__(self, ori_data:'pd.DataFrame', tar_data:'pd.DataFrame', weight_column, results_dir):
        """
        Parameters:
        `ori_data`: pandas DataFrame containing the original data.
        `tar_data`: pandas DataFrame containing the target data."""
        self.ori_data = ori_data
        self.tar_data = tar_data
        self.weight_column = weight_column
        self.ori_weight = None
        self.tar_weight = None
        self.results_dir = results_dir

    @staticmethod
    def drop_likes(df: 'pd.DataFrame', drop_kwd: 'list[str]' = []):
        """Drop columns containing the keywords in `drop_kwd`."""
        for kwd in drop_kwd:
            df = df.drop(columns=df.filter(like=kwd).columns, inplace=False)
        return df
    
    @staticmethod
    def clean_data(df_original, label, drop_kwd, wgt_col, drop_neg_wgts=True) -> tuple['pd.DataFrame', 'pd.Series', 'pd.Series', 'pd.DataFrame']:
        """Clean the data by dropping columns containing the keywords in `drop_kwd`.

        Parameters:
        - `label`: Label column
        
        Return
        - `X`: Features, pandas DataFrame
        - `y`: Labels, pandas Series
        - `weights`: Weights, pandas Series
        - `neg_df`: DataFrame containing the events with negative weights."""
        df = df_original.copy()
        neg_df = df[df[wgt_col] < 0]
        df = df[df[wgt_col] > 0] if drop_neg_wgts else df

        print("Dropped ", len(df_original) - len(df), " events with negative weights out of ", len(df_original), " events.")

        drop_kwd.append(wgt_col)
        X = Reweighter.drop_likes(df, drop_kwd)

        y = pd.Series([label] * len(df))

        weights = df[wgt_col]
        
        return X, y, weights, neg_df
    
    @staticmethod
    def prep_ori_tar(ori, tar, drop_kwd, wgt_col, drop_neg_wgts=True):
        """Preprocess the original and target data by dropping columns containing the keywords in `drop_kwd`."""
        
        X_ori, y_ori, w_ori, _ = Reweighter.clean_data(ori, 0, drop_kwd, wgt_col, drop_neg_wgts)
        X_tar, y_tar, w_tar, _ = Reweighter.clean_data(tar, 1, drop_kwd, wgt_col, drop_neg_wgts)
        
        return pd.concat([X_ori, X_tar], ignore_index=True, axis=0), pd.concat([y_ori, y_tar], ignore_index=True, axis=0), pd.concat([w_ori, w_tar], ignore_index=True, axis=0)
    
    @staticmethod
    def draw_distributions(original, target, o_wgt, t_wgt, original_label, target_label, column, bins=10, range=None, save_path=False):
        """Draw the distributions of the original and target data. Normalized."""
        hist_settings = {'bins': bins, 'density': True, 'alpha': 0.5}
        plt.figure(figsize=[12, 7])
        xlim = np.percentile(np.hstack([target[column]]), [0.01, 99.99])
        range = xlim if range is None else range
        plt.hist(original[column], weights=o_wgt, range=range, label=original_label, **hist_settings)
        plt.hist(target[column], weights=t_wgt, range=range, label=target_label, **hist_settings)
        plt.legend(loc='best')
        plt.title(column)
        if save_path:
            plt.savefig(save_path)
