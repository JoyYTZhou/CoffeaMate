import pandas as pd
import awkward as ak
import uproot, pickle, os, subprocess, json, gzip
import numpy as np
from functools import wraps
import dask_awkward as dak
import logging
import multiprocessing as mp
from functools import partial
from src.utils.filesysutil import pjoin, FileSysHelper

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datadir = pjoin(parent_directory, 'data', 'preprocessed')

runcom = subprocess.run

class CutflowProcessor:
    """Processes and manipulates cutflow tables."""
    @staticmethod
    def check_events_match(df, col_name, rootfiles, empty_kwd='empty'):
        """Validates if events count in cutflow matches root files.
        
        Parameters
        - df: pandas DataFrame containing cutflow
        - col_name: column name in DataFrame containing event counts
        - rootfiles: single root file path or list of root file paths
        - empty_kwd: keyword to identify empty files
        
        Returns
        - bool: True if events match or file is properly marked as empty, False if mismatch or corruption
        """
        cutflow_events = df[col_name].iloc[-1]

        if isinstance(rootfiles, str):
            if empty_kwd in rootfiles:
                return cutflow_events == 0
            else:
                rootfiles = [rootfiles]
        
        try:
            total_events = 0
            corrupted_files = []
            
            for root_file in rootfiles:
                try:
                    with uproot.open(root_file) as f:
                        total_events += f["Events"].num_entries
                except Exception as e:
                    logging.error(f"Corrupted root file detected - {root_file}: {str(e)}")
                    corrupted_files.append(root_file)
            
            if corrupted_files:
                logging.warning(f"Found {len(corrupted_files)} corrupted root files")
                return False
            
            logging.debug("Checking event counts...")
            logging.debug(f"Events in root files: {total_events}")
            logging.debug(f"Events in cutflow: {cutflow_events}")
            
            return total_events == cutflow_events
        
        except Exception as e:
            logging.error(f"Error checking events match: {str(e)}")
            return False

    @staticmethod
    def merge_cutflows(inputdir, dataset_name, keyword='cutflow', save=True, outpath=None):
        """Merges multiple cutflow tables for a single dataset."""
        # Load and concatenate all matching cutflow files
        pattern = f'{dataset_name}*{keyword}*.csv'
        cutflow_dfs = DataLoader.load_csvs(
            dirname=inputdir, 
            filepattern=pattern,
            func=lambda dfs: pd.concat(dfs)
        )
        
        merged_df = cutflow_dfs.groupby(cutflow_dfs.index, sort=False).sum()
        
        # Standardize column names
        if merged_df.shape[1] > 1:
            merged_df.columns = [f"{dataset_name}_{col}" for col in merged_df.columns]
        else:
            merged_df.columns = [dataset_name]
            
        if save and outpath:
            merged_df.to_csv(outpath)
            
        return merged_df

    @staticmethod
    def combine_selections(cutflow_files, save=True, outpath=None):
        """Combines cutflow tables from different selection steps."""
        # Load all cutflow files
        cutflows = DataLoader.load_csvs(cutflow_files)
        
        # Skip first row (usually total events) except for first file
        processed_dfs = [
            df.iloc[1:] if i > 0 else df 
            for i, df in enumerate(cutflows)
        ]
        
        # Combine all selections
        combined = pd.concat(processed_dfs, axis=1)
        
        if save and outpath:
            combined.to_csv(outpath)
            
        return combined
    
    @staticmethod
    def apply_weights(cutflow_df, weight_dict, luminosity=50.0, save=False, outpath=None):
        """Applies physics weights to cutflow table."""
        # Calculate final weights including luminosity
        final_weights = {
            process: weight * luminosity 
            for process, weight in weight_dict.items()
        }
        
        # Apply weights to each process
        weighted_df = cutflow_df.copy()
        for process, weight in final_weights.items():
            cols = weighted_df.filter(like=process).columns
            weighted_df[cols] = weighted_df[cols] * weight
            
        if save and outpath:
            weighted_df.to_csv(outpath)
            
        return weighted_df

    @staticmethod
    def apply_physics_scale(cutflow_df, metadata, luminosity):
        """Applies physics-specific scaling factors to cutflow.
        
        Parameters
        ----------
        cutflow_df : pd.DataFrame
            Input cutflow dataframe
        metadata : dict
            Dataset metadata containing {dataset: {xsection, nwgt, ...}}
        luminosity : float
            Luminosity in pb^-1
            
        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            (Scaled cutflow dataframe, Combined group weights)
        """
        # Calculate physics weights
        physics_weights = {
            dsinfo['shortname']: (1/dsinfo['nwgt']) * dsinfo['xsection']
            for dsname, dsinfo in metadata.items()
        }
        
        # Apply weights using existing method
        scaled_df = CutflowProcessor.apply_weights(
            cutflow_df, 
            physics_weights, 
            luminosity=luminosity
        )
        
        # Get combined weights
        combined = CutflowProcessor.sum_kwd(scaled_df, 'wgt')
        
        return scaled_df, combined
    
    @staticmethod
    def calculate_efficiency(df, column=None, method='incremental', inplace=True):
        """Calculates selection efficiencies."""
        def _calc_single_efficiency(series, method):
            if method == 'incremental':
                eff = series.div(series.shift(1)).fillna(1)
            elif method == 'overall':
                eff = series.div(series.iloc[0]).fillna(1)
            else:
                raise ValueError("Method must be 'incremental' or 'overall'")
                
            # Clean up infinities and NaNs
            eff.replace([np.inf, -np.inf], np.nan, inplace=True)
            eff.fillna(0 if method == 'incremental' else 1, inplace=True)
            return eff

        result_df = df.copy() if inplace else df
        
        if column:
            # Calculate efficiency for single column
            return _calc_single_efficiency(result_df[column], method)
        else:
            # Calculate efficiency for all columns
            for col in result_df.columns[::-1]:
                eff = _calc_single_efficiency(result_df[col], method)
                result_df.insert(
                    result_df.columns.get_loc(col) + 1,
                    f"{col}_eff",
                    eff
                )
                
        return result_df
    
    @staticmethod
    def calculate_yield(df, signal_processes):
        """Calculates physics yields with background and signal efficiencies."""
        result_df = df.copy()
        
        # Separate signals and backgrounds
        sig_cols = [col for col in df.columns if col in signal_processes]
        bkg_cols = df.columns.difference(sig_cols)

        # Calculate total background
        result_df['Total_Background'] = result_df[bkg_cols].sum(axis=1)
        
        # Calculate efficiencies
        result_df['Background_Efficiency'] = CutflowProcessor.calculate_efficiency(
            result_df, 'Total_Background', inplace=False
        )

        # Calculate signal efficiencies
        for signal in sig_cols:
            result_df[f'{signal}_Efficiency'] = CutflowProcessor.calculate_efficiency(
                result_df, signal, inplace=False
            )

        # Organize columns
        column_order = (
            list(bkg_cols) +
            ['Total_Background', 'Background_Efficiency'] +
            sum([[sig, f'{sig}_Efficiency'] for sig in sig_cols], [])
        )
        
        return result_df[column_order]

    @staticmethod
    def categorize_processes(df, grouping_dict):
        """Groups processes into categories based on keywords."""
        result_df = df.copy()
        
        for new_category, process_keywords in grouping_dict.items():
            # Find columns matching any keyword
            matching_cols = [
                col for col in df.columns 
                if any(kwd in col for kwd in process_keywords)
            ]
            
            if matching_cols:
                # Sum matching processes into new category
                result_df[new_category] = result_df[matching_cols].sum(axis=1)
                # Remove original columns
                result_df.drop(columns=matching_cols, inplace=True)
                
        return result_df

    @staticmethod
    def get_process_summary(df, step_name):
        """Creates a summary of all processes at a specific selection step.
        
        New utility method for quick process comparisons.
        """
        if step_name not in df.index:
            raise ValueError(f"Step '{step_name}' not found in cutflow")
            
        summary = pd.Series({
            'Total Events': df.loc[step_name].sum(),
            'N Processes': len(df.columns),
            'Largest Process': df.loc[step_name].idxmax(),
            'Smallest Process': df.loc[step_name].idxmin(),
            'Mean Events': df.loc[step_name].mean(),
            'Std Events': df.loc[step_name].std()
        })
        
        return summary
   
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

    @staticmethod
    def load_fields(file, branch_names=None, tree_name='Events', lib='ak') -> tuple[ak.Array, list]:
        """Load specific fields if any. Otherwise load all. If the file is a list, concatenate the data from all files.
        
        Parameters:
        - file: path to the root file or list of paths
        - branch_names: list of branch names to load
        - tree_name: name of the tree in the root file
        - lib: library to use for loading the data

        Returns:
        - awkward array of the loaded data (, list of empty files)
        """
        def load_one(fi):
            with uproot.open(fi) as file:
                if file.keys() == []:
                    return False
                else:
                    tree = file[tree_name] 
            return tree.arrays(branch_names, library=lib)

        returned = None
        if isinstance(file, str):
            file = [file]
        dfs = []
        emptylist = []
        for root_file in file:
            result = load_one(root_file)
            if result: dfs.append(load_one(file))
            else: emptylist.append(root_file)
        combined_evts = ak.concatenate(dfs)
        return combined_evts, emptylist

    @staticmethod
    def load_pkl(filename):
        """Load a pickle file and return the data."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
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
    
    @staticmethod
    def process_file(path, group, resolution):
        """Read and group a file based on resolution."""
        df = pd.read_csv(path, index_col=0)
        if resolution == 0:
            df = df.sum(axis=1).to_frame(name=group)
        return df

class DataSetUtil:
    @staticmethod
    def extract_uuids(json_path: str) -> dict:
        """Extract UUIDs from dataset JSON files.
        
        Parameters
        - json_path: path to the JSON.GZ file containing dataset information
        
        Returns
        - dict: Nested dictionary containing dataset information and UUIDs
            Format: {dataset_era: {sample_name: {"shortname": str, "uuids": list[str]}}}
        
        Example:
        >>> extract_uuids("data/preprocessed/DYJets_2022PostEE.json")
        {
            "DYJetsToLL_M-50": ["0f7376a8-61bc-11ee-95cf-3401a8c0beef", ...],
            "DYJetsToLL_M-10to50": ["0f7376a8-61bc-11ee-95cf-3401a8c0beef", ...],
        }
        """
        # Read JSON file
        with gzip.open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract UUIDs and metadata
        shortname = None
        result = {}
        
        # Iterate through each dataset
        for dataset_info in data.values():
            uuids = []
            # Get metadata
            if 'metadata' in dataset_info and 'shortname' in dataset_info['metadata']:
                shortname = dataset_info['metadata']['shortname']
            
            # Get UUIDs from files
            if 'files' in dataset_info:
                for file_info in dataset_info['files'].values():
                    if 'uuid' in file_info:
                        uuids.append(file_info['uuid'])
            result[shortname] = uuids
        
        return result

    @staticmethod
    def _validate_single_pair(uuid_info, root_dir, csv_dir):
        """Helper function to validate a single UUID pair.
        
        Parameters
        - uuid_info: tuple of (shortname, uuid)
        - root_dir: directory containing root files
        - csv_dir: directory containing cutflow CSV files
        
        Returns
        - dict: Validation result for this UUID
        """
        shortname, uuid = uuid_info
        base_pattern = f"{shortname}_{uuid}"
        
        # Find matching files
        root_files = FileSysHelper.glob_files(
            root_dir, 
            filepattern=f"{base_pattern}*.root"
        )
        csv_files = FileSysHelper.glob_files(
            csv_dir, 
            filepattern=f"{base_pattern}*cutflow.csv"
        )
        
        # Check if files exist
        if not root_files or not csv_files:
            return ("missing", {
                "shortname": shortname,
                "uuid": uuid,
                "missing_root": len(root_files) == 0,
                "missing_csv": len(csv_files) == 0
            })
        
        # Validate event counts
        try:
            df = pd.read_csv(csv_files[0], index_col=0)
            events_match = CutflowProcessor.check_events_match(
                df=df,
                col_name=df.columns[0],
                rootfiles=root_files
            )
            
            if events_match:
                return ("valid", {
                    "shortname": shortname,
                    "root": root_files[0],
                    "csv": csv_files[0],
                    "uuid": uuid
                })
            else:
                return ("mismatched", {
                    "shortname": shortname,
                    "root": root_files[0],
                    "csv": csv_files[0],
                    "uuid": uuid
                })
        except Exception as e:
            logging.error(f"Error validating files for {shortname} UUID {uuid}: {str(e)}")
            return ("missing", {
                "shortname": shortname,
                "uuid": uuid,
                "error": str(e)
            })

    @staticmethod
    def validate_file_pairs(json_path: str, root_dir: str, csv_dir: str, n_workers: int = None) -> dict:
        """Validate matching root and cutflow CSV files for a dataset in parallel.
        
        Parameters
        - json_path: path to the JSON file containing dataset information
        - root_dir: directory containing root files
        - csv_dir: directory containing cutflow CSV files
        - n_workers: number of worker processes (default: number of CPU cores - 1)
        
        Returns
        - dict: Dictionary with validation results
        """
        # Get dataset information
        dataset_info = DataSetUtil.extract_uuids(json_path)
        
        # Prepare list of work items
        work_items = [
            (shortname, uuid) 
            for shortname, uuids in dataset_info.items()
            for uuid in uuids
        ]
        
        # Set up multiprocessing
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 2)
            
        # Prepare results container
        results = {
            "valid_pairs": [],
            "mismatched_events": [],
            "missing_files": []
        }
        
        # Process in parallel
        with mp.Pool(n_workers) as pool:
            validate_func = partial(DataSetUtil._validate_single_pair, 
                                 root_dir=root_dir, 
                                 csv_dir=csv_dir)
            
            # Process all pairs and collect results
            for result_type, result_info in pool.imap_unordered(validate_func, work_items):
                if result_type == "valid":
                    results["valid_pairs"].append(result_info)
                elif result_type == "mismatched":
                    results["mismatched_events"].append(result_info)
                else:  # missing
                    results["missing_files"].append(result_info)
        
        return results
     
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
