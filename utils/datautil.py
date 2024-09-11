import pandas as pd
import awkward as ak
import uproot, pickle, os
from functools import wraps
import dask_awkward as dak
import json
from src.utils.filesysutil import glob_files, pjoin

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datadir = pjoin(parent_directory, 'data', 'preprocessed')

def iterwgt(func):
    @wraps(func)
    def wrapper(instance, *args, **kwargs):
        for process, dsinfo in instance.wgt_dict.items():
            for ds in dsinfo.keys():
                func(instance, process, ds, *args, **kwargs)
    return wrapper
    
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

