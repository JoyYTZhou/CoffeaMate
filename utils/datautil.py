import pandas as pd
import awkward as ak
import uproot, pickle, os, subprocess
import numpy as np
from functools import wraps
import dask_awkward as dak
from src.utils.filesysutil import pjoin, FileSysHelper

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datadir = pjoin(parent_directory, 'data', 'preprocessed')

runcom = subprocess.run

class CutflowProcessor:
    @staticmethod
    def check_last_no(df, col_name, rootfiles):
        """Check if the last number in the cutflow table matches the number of events in the root files.
        
        Parameters
        - `df`: cutflow dataframe
        - `col_name`: name of the column to check
        - `rootfiles`: list of root files
        """
        if isinstance(rootfiles, str):
            rootfiles = [rootfiles]
        
        raw = 0
        for file in rootfiles:
            with uproot.open(file) as f:
                thisraw = f.get("Events").num_entries
            raw += thisraw
        
        print(f'Got {raw} events in root files!')
        print(f'Got {df[col_name].iloc[-1]} events in cutflow table!')
        
        return df[col_name].iloc[-1] == raw

    @staticmethod
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

    @staticmethod
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
    
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def process_yield(yield_df, signals) -> pd.DataFrame:
        """group the yield dataframe to include signal and background efficiencies.

        Parameters
        - `yield_df`: dataframe of yields
        - `signals`: list of signal group names
        
        Return
        - processed yield dataframe"""
        sig_list = [signal for signal in signals if signal in yield_df.columns]
        bkg_list = yield_df.columns.difference(sig_list)

        yield_df['Tot Bkg'] = yield_df[bkg_list].sum(axis=1)
        yield_df['Bkg Eff'] = CutflowProcessor.calc_eff(yield_df, 'Tot Bkg', inplace=False)

        for signal in sig_list:
            yield_df[f'{signal} Eff'] = CutflowProcessor.calc_eff(yield_df, signal, inplace=False)

        new_order = list(bkg_list) + ['Tot Bkg', 'Bkg Eff']
        for signal in sig_list:
            new_order.extend([signal, f'{signal} Eff'])
            
        yield_df = yield_df[new_order]
        return yield_df
    
    @staticmethod
    def categorize(df, group_kwd:'dict') -> pd.DataFrame:
        """Recalculate/categorize a table by the group keyword.
        
        Parameters
        - `group_kwd`: {name of new column: [keywords to search for in the column names]}"""
        for newcol, kwdlist in group_kwd.items():
            cols = [col for col in df.columns if any(kwd in col for kwd in kwdlist)]
            if cols:
                df[newcol] = df[cols].sum(axis=1)
                df.drop(columns=cols, inplace=True)
        return df
   
    @staticmethod
    def sum_kwd(cfdf, keyword) -> pd.Series:
        """Add a column to the cutflow table by summing up all columns with the keyword.

        Return
        - Series of the summed column"""
        same_cols = cfdf.filter(like=keyword)
        sumcol = same_cols.sum(axis=1)
        return sumcol 

class DataLoader:
    @staticmethod
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



def iterwgt(func):
    @wraps(func)
    def wrapper(instance, *args, **kwargs):
        for process, dsinfo in instance.wgt_dict.items():
            for ds in dsinfo.keys():
                func(instance, process, ds, *args, **kwargs)
    return wrapper

def extract_leaf_values(d) -> list:
    """Extract leaf values from a dictionary.
    
    Return
    - list of leaf values
    """
    leaf_values = []
    
    for value in d.values():
        if isinstance(value, dict):
            leaf_values.extend(extract_leaf_values(value))
        else:
            leaf_values.append(value)

    return leaf_values
    
def get_compression(**kwargs):
    """Returns the compression algorithm to use for writing root files."""
    compression = kwargs.pop('compression', None)
    compression_level = kwargs.pop('compression_level', 1)

    if compression in ("LZMA", "lzma"):
        compression_code = uproot.const.kLZMA
    elif compression in ("ZLIB", "zlib"):
        compression_code = uproot.const.kZLIB
    elif compression in ("LZ4", "lz4"):
        compression_code = uproot.const.kLZ4
    elif compression in ("ZSTD", "zstd"):
        compression_code = uproot.const.kZSTD
    elif compression is None:
        raise UserWarning("Not sure if this option is supported, should be...")
    else:
        msg = f"unrecognized compression algorithm: {compression}. Only ZLIB, LZMA, LZ4, and ZSTD are accepted."
        raise ValueError(msg)
    
    if compression is not None: 
        compression = uproot.compression.Compression.from_code_pair(compression_code, compression_level)

    return compression

def load_pkl(filename):
    """Load a pickle file and return the data."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def load_data(source, **kwargs):
    """Load a data file and return the data."""
    if isinstance(source, str):
        if source.endswith('.csv'):
            data = pd.read_csv(source, **kwargs)
        elif source.endswith('.root'):
            data = uproot.open(source)
        elif source.endswith('.parquet'):
            data = pd.read_parquet(source, **kwargs)
        else:
            raise ValueError("This is not a valid file type.")
    elif checkevents(source):
        data = source
    else:
        data = source
        raise UserWarning(f"This might not be a valid source. The data type is {type(source)}")
    return data

def arr_handler(dfarr, allow_delayed=True) -> ak.Array:
    """Handle different types of data arrays to convert them to awkward arrays."""
    if isinstance(dfarr, pd.core.series.Series):
        try: 
            ak_arr = dfarr.ak.array
            return ak_arr
        except AttributeError as e:
            return dfarr
    elif isinstance(dfarr, pd.core.frame.DataFrame):
        raise ValueError("specify a column. This is a dataframe.")
    elif isinstance(dfarr, ak.highlevel.Array):
        return dfarr
    elif isinstance(dfarr, dak.lib.core.Array):
        if allow_delayed:
            return dfarr
        else:
            return dfarr.compute()
    else:
        raise TypeError(f"This is of type {type(dfarr)}")

def checkevents(events):
    """Returns True if the events are in the right format, False otherwise."""
    if hasattr(events, 'keys') and callable(getattr(events, 'keys')):
        return True
    elif hasattr(events, 'fields'):
        return True
    elif isinstance(events, pd.core.frame.DataFrame):
        return True
    else:
        raise TypeError("Invalid type for events. Must be an awkward object or a DataFrame")

def findfields(dframe):
    """Find all fields in a dataframe."""
    if isinstance(dframe, pd.core.frame.DataFrame):
        return dframe.columns
    elif hasattr(dframe, 'keys') and callable(getattr(dframe, 'keys')):
        return dframe.keys()
    else:
        return "Not supported yet..."

def find_branches(file_path, object_list, tree_name, extra=[]) -> list:
    """ Return a list of branches for objects in object_list

    Paremters
    - `file_path`: path to the root file
    - `object_list`: list of objects to find branches for
    - `tree_name`: name of the tree in the root file

    Returns
    - list of branches
    """
    file = uproot.open(file_path)
    tree = file[tree_name]
    branch_names = tree.keys()
    branches = []
    for object in object_list:
        branches.extend([name for name in branch_names if name.startswith(object)])
    if extra != []:
        branches.extend([name for name in extra if name in branch_names])
    return branches

