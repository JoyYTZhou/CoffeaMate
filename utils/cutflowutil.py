import os, subprocess
import numpy as np
import pandas as pd

from src.utils.filesysutil import FileSysHelper

pjoin = os.path.join
runcom = subprocess.run


def load_csvs(dirname, filepattern, func=None, *args, **kwargs) -> pd.DataFrame:
    """Load csv files matching a pattern into a list of DataFrames. Post process if func is provided.
    
    Parameters
    - `dirname`: directory name to search for
    - `startpattern`: pattern to match the file names
    - `func`: function to apply to the list of DataFrames. Must return an Pandas object.
    - `*args`, `**kwargs`: additional arguments to pass to the function
    """
    file_names = FileSysHelper.glob_files(dirname, filepattern=filepattern)
    dfs = [pd.read_csv(file_name, index_col=0, header=0) for file_name in file_names] 
    if not dfs:
        print(f"No files found with filepattern {filepattern} in {dirname}, please double check if your output files match the input filepattern to glob.")
    if func is None:
        return dfs
    else:
        return func(dfs, *args, **kwargs)

def combine_cf(inputdir, dsname, keyword='cutflow', output=True, outpath=None):
    """Combines(sums) all cutflow tables in a source directory belonging to one datset and output them into output directory.
    Essentially this will grep files of pattern "{dsname}*{keyword}*.csv" and combine them to one csv file.
    
    Parameters
    - `inputdir`: source directory
    - `dsname`: dataset name. 
    - `keyword`: keyword to search for 
    - `output`: whether to save the combined table into a csv file
    - `outpath`: path to the output
    """
    concat_df = load_csvs(dirname=inputdir, filepattern=f'{dsname}*{keyword}*.csv', 
                          func=lambda dfs: pd.concat(dfs))
    combined = concat_df.groupby(concat_df.index, sort=False).sum()
    if combined.shape[1] != 1:
        combined.columns = [f"{dsname}_{col}" for col in combined.columns]
    else:
        combined.columns = [dsname]
    if output and outpath is not None: combined.to_csv(outpath)
    return combined

def add_selcutflow(cutflowlist, save=True, outpath=None):
    """Add cutflows sequentially.
    
    Parameters
    - `cutflowlist`: list of cutflow csv files
    - `save`: whether to save the combined table into a csv file
    - `outpath`: path to the output
    
    Return
    - combined cutflow table"""
    dfs = load_csvs(cutflowlist)
    dfs = [df.iloc[1:] for i, df in enumerate(dfs) if i != 0]
    result = pd.concat(dfs, axis=1)
    if save: result.to_csv(outpath)
    return result

def weight_cf(wgt_dict, raw_cf, save=False, outname=None, lumi=50):
    """Calculate weighted table based on raw table.
    
    Parameters
    - `wgt_dict`: dictionary of weights
    - `raw_cf`: raw cutflow table
    - `lumi`: luminosity (pb^-1)

    Return
    - `wgt_df`: weighted cutflow table
    """ 
    weights = {key: wgt*lumi for key, wgt in wgt_dict.items()}
    wgt_df = raw_cf.mul(weights)
    if save and outname is not None: wgt_df.to_csv(outname)
    return wgt_df

def calc_eff(cfdf, column_name=None, type='incremental', inplace=True) -> pd.DataFrame:
    """Return efficiency for each column in the DataFrame right after the column itself.
    
    Parameters:
    - `cfdf`: DataFrame to calculate efficiency on
    - `column_name`: specific column to calculate efficiency on (optional)
    - `type`: type of efficiency calculation. 'incremental' or 'overall'
    """
    def calculate_efficiency(series):
        if type == 'incremental':
            return series.div(series.shift(1)).fillna(1)
        elif type == 'overall':
            return series.div(series.iloc[0]).fillna(1)
        else:
            raise ValueError("Invalid type. Expected 'incremental' or 'overall'.")

    if column_name:
        eff_series = calculate_efficiency(cfdf[column_name])
        eff_series.replace([np.inf, -np.inf], np.nan, inplace=True)
        eff_series.fillna(0 if type == 'incremental' else 1, inplace=True)
    else:
        for col in cfdf.columns[::-1]:  # Iterate in reverse to avoid column shifting issues
            eff_series = calculate_efficiency(cfdf[col])
            eff_series.replace([np.inf, -np.inf], np.nan, inplace=True)
            eff_series.fillna(0 if type == 'incremental' else 1, inplace=True)
            cfdf.insert(cfdf.columns.get_loc(col) + 1, f"{col}_eff", eff_series)
    if inplace: return cfdf
    else: return eff_series
    