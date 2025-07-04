import uproot, dask, logging, psutil, os
import pandas as pd
import concurrent.futures
import awkward as ak
import dask_awkward as dak
from datetime import datetime
from uproot.writing._dask_write import ak_to_root

from src.utils.displayutil import create_rich_logger
from src.utils.filesysutil import XRootDHelper, pjoin
import logging
from datetime import datetime

def setup_logging(console_level=logging.INFO, file_level=logging.DEBUG, log_to_file=True):  # Changed default console_level to INFO
    """Setup logging configuration."""
    # Remove any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    rich_handler = create_rich_logger()
    rich_handler.setLevel(console_level)
    root_logger.addHandler(rich_handler)

    if log_to_file:
        log_file = f'debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
    root_logger.setLevel(min(console_level, file_level))

    # Configure library loggers
    logging.getLogger('uproot').setLevel(logging.WARNING)
    logging.getLogger('dask').setLevel(logging.WARNING)
    logging.getLogger('distributed').setLevel(logging.WARNING)

    # Test logging
    logging.debug("Logging setup complete - DEBUG test")
    logging.info("Logging setup complete - INFO test")
    logging.warning("Logging setup complete - WARNING test")

def check_open_files(auto_close_threshold: float = 100, max_objects: int = 10) -> tuple[int, list[str]]:
    """Check and log details about currently open files with error handling.

    Optionally close files exceeding a size threshold.

    Args:
        auto_close_threshold (float): Size threshold in MB to automatically close files.
                                      Set to None to disable auto-closing.
        max_objects (int): Number of largest open files to log.

    Returns:
        tuple: (number of remaining open files, list of file paths)
    """
    try:
        process = psutil.Process()
        open_files = process.open_files()

        file_info = []
        closed_files = []
        nfs_files = []  # Track .nfsXXXX files

        for file in open_files:
            try:
                if os.path.exists(file.path):
                    size = os.path.getsize(file.path)
                    file_info.append((file, size))

                    # Auto-close files if they exceed threshold
                    if auto_close_threshold is not None and size > auto_close_threshold * 1024 * 1024:
                        try:
                            os.close(file.fd)  # Close file descriptor
                            closed_files.append((file.path, size))
                            logging.warning(
                                f"Closed large file: {file.path} "
                                f"(size: {size/1024/1024:.2f}MB, fd: {file.fd})"
                            )
                        except Exception as close_error:
                            logging.error(f"Failed to close file {file.path}: {close_error}")

                # Check if file is a .nfsXXXX file
                if "/.nfs" in file.path:
                    nfs_files.append(file.path)

            except (OSError, IOError) as e:
                logging.warning(f"Could not get size for {file.path}: {e}")

        # Sort files by size (descending)
        file_info.sort(key=lambda x: x[1], reverse=True)
        largest_files = file_info[:max_objects]

        # Log overall summary
        logging.info(f"Total open files: {len(open_files)}")
        if closed_files:
            logging.info(f"Automatically closed {len(closed_files)} files exceeding {auto_close_threshold}MB:")
            for path, size in closed_files:
                logging.info(f"  - {path} ({size/1024/1024:.2f}MB)")

        if nfs_files:
            logging.warning(f"Detected {len(nfs_files)} lingering .nfsXXXX files:")
            for path in nfs_files:
                logging.warning(f"  - {path}")

        # Log details of largest open files
        for file, size in largest_files:
            try:
                file_details = {
                    "path": file.path,
                    "fd": file.fd,
                    "mode": getattr(file, "mode", "unknown"),
                    "size": f"{size/1024/1024:.2f}MB" if size > 0 else "unknown",
                    "position": getattr(file, "position", "unknown"),
                    "status": "closed" if file.path in [f[0] for f in closed_files] else "open"
                }
                logging.info(f"File Details: {file_details}")
            except Exception as e:
                logging.warning(f"Error getting details for file {file.path}: {e}")

        remaining_files = len(open_files) - len(closed_files)
        active_paths = [f[0].path for f in largest_files if f[0].path not in [cf[0] for cf in closed_files]]

        return remaining_files, active_paths

    except psutil.AccessDenied:
        logging.error("Access denied when trying to get open files")
        return 0, []
    except psutil.NoSuchProcess:
        logging.error("Process not found")
        return 0, []
    except Exception as e:
        logging.error(f"Unexpected error in check_open_files: {str(e)}")
        return 0, []

def write_empty_root(filename):
    """Creates an empty ROOT file as a placeholder."""
    with uproot.recreate(filename):
        pass

def write_root(evts: 'ak.Array | pd.DataFrame', destination, outputtree="Events", title="Events", compression=None):
    """Write arrays to root file. Highly inefficient methods in terms of data storage.

    Parameters
    - `destination`: path to the output root file
    - `outputtree`: name of the tree to write to
    - `title`: title of the tree
    - `compression`: compression algorithm to use"""
    branch_types = {name: evts[name].type for name in evts.fields}
    with uproot.recreate(destination, compression=compression) as file:
        file.mktree(name=outputtree, branch_types=branch_types, title=title)
        file[outputtree].extend({name: evts[name] for name in evts.fields})

def load_files_remote_delayed(files_dict, uproot_args={}, remote_read=True) -> tuple:
    """Load ROOT files remotely using uproot.dask.

    Parameters
    ----------
    files_dict : dict
        Dictionary containing file information with paths and UUIDs
        Example format:
        {
            "root://path/to/file1.root:Events": {"uuid": "1"},
            "root://path/to/file2.root:Events": {"uuid": "2"}
        }
    uproot_args : dict, optional
        Additional arguments to pass to uproot
    remote_read : bool, optional
        If True, read directly from remote location

    Returns
    -------
    tuple
        (events, suffixes) where events is a dask array and suffixes is a list of UUIDs
    """
    try:
        # Extract suffixes (UUIDs) from the files dictionary
        suffixes = [info['uuid'] for info in files_dict.values()]

        # Load the files using uproot.dask
        events = uproot.dask(
            files=files_dict,
            **uproot_args
        )

        logging.debug(f"Successfully loaded {len(suffixes)} files remotely")
        return (events, suffixes)

    except Exception as e:
        logging.exception(f"Failed to load files remotely: {str(e)}")

def process_file(filename, fileinfo, copydir, delayed_open=True, uproot_args={}, remote_read=False) -> tuple:
    """Handles file loading, either remotely or by copying locally first.

    Parameters:
    --------
    filename : str
        Path to the input file
    fileinfo : dict
        Dictionary containing file information including 'uuid'
    copydir : str
        Directory to copy files to when not reading remotely
    delayed_open : bool, optional
        Whether to use delayed (dask) reading
    uproot_args : dict, optional
        Additional arguments to pass to uproot
    remote_read : bool, optional
        If True, read directly from remote location without copying

    Returns:
    --------
    tuple : (events, suffix)
        events: either a dask array or uproot array depending on delayed_open
        suffix: the file's uuid
    """
    suffix = fileinfo['uuid']

    if remote_read:
        input_path = filename
        logging.debug(f"Reading remotely from {input_path}")
    else:
        dest_file = pjoin(copydir, f"{suffix}.root")
        logging.debug(f"Copying and loading {filename} to {dest_file}")
        XRootDHelper.copy_local(filename, dest_file)
        input_path = dest_file

    try:
        if delayed_open:
            events = uproot.dask(files={input_path: fileinfo}, **uproot_args)
            return (events, suffix)
        else:
            return (uproot.open(input_path + ":Events").arrays(**uproot_args), suffix)

    except Exception as e:
        logging.error(f"Failed to read file {input_path}: {str(e)}")
        raise

def submit_copy_and_load(fileargs, copydir, rtcfg, read_args, max_workers=3) -> list:
    """Runs file copying and loading in parallel"""
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = parallel_process_files(fileargs, copydir, executor, rtcfg, read_args)

        for future in concurrent.futures.as_completed(future_to_file):
            try:
                results.append(future.result())
            except Exception as e:
                logging.exception(f"Error copying and loading {future_to_file[future]}: {e}")
    return results

def parallel_process_files(fileargs, executor, copydir=None, delayed_open=True, uproot_args={}) -> dict:
    """Load files individually in parallel using a ThreadPoolExecutor.
    
    Parameters 
    ----------
    - fileargs : dict
        { "files": {filename: fileinfo}, "metadata": {metadata} }
    
    Returns
    -------
    - future_to_file : dict
        Dictionary of futures to file names
    """
    remote_load = False
    if copydir is None:
        remote_load = True
    future_to_file = {
        filename: executor.submit(process_file, filename, fileinfo, copydir, delayed_open, uproot_args, remote_read=remote_load)
        for filename, fileinfo in fileargs['files'].items()
    }
    return future_to_file

def compute_dask_array(passed, force_compute=True) -> ak.Array:
    if hasattr(passed, 'npartitions'):
        passed = passed.persist()

        length_calcs = [dask.delayed(len)(passed.partitions[i]) for i in range(passed.npartitions)]
        lengths = dask.compute(*length_calcs)

        has_zero_lengths = any(l == 0 for l in lengths)

        if not has_zero_lengths:
            logging.debug("No zero-arrays found")
            if force_compute:
                logging.debug("Computing dask arrays...")
                return dask.compute(passed)[0]
            else:
                return passed
        else:
            logging.debug("Found zero-length partitions, filtering them out")
            valid_indices = [i for i, l in enumerate(lengths) if l > 0]
            if not valid_indices:
                logging.debug("No valid partitions found, skipping write")
                return None
            else:
                logging.debug("Valid indices: %s", valid_indices)
                try:
                    valid_partitions = [passed.partitions[i] for i in valid_indices]
                    valid_data = dak.concatenate(valid_partitions)
                    computed_data = dask.compute(valid_data)[0]
                    return computed_data
                except Exception as e:
                    logging.exception(f"Error encountered in computing partitions: {e}")
    else:
        logging.debug(f"passed is of type {type(passed)}")
        return passed

def write_computed_array(computed_array, outdir, dataset, suffix, write_args={}) -> int:
    """Write array using appropriate method based on type."""
    write_options = {
        "initial_basket_capacity": 50,
        "resize_factor": 1.5,
        "compression": "ZLIB",
        "compression_level": 1
    }

    if isinstance(computed_array, dak.Array):
        try:
            uproot.dask_write(
                computed_array, destination=outdir, tree_name="Events", compute=True,
                prefix=f'{dataset}_{suffix}', **write_options, **write_args )
        except MemoryError:
            logging.exception(f"MemoryError encountered in writing outputs with dask_write() for file index {suffix}.")
            return 1
        except Exception as e:
            logging.exception(f"Error encountered in writing outputs with dask_write() for file index {suffix}: {e}")
            return 1
    elif computed_array is not None:
        output_path = pjoin(outdir, f'{dataset}_{suffix}.root')
        try:
            ak_to_root(
                output_path,
                computed_array,
                tree_name="Events",
                title="",
                counter_name=lambda counted: 'n' + counted,
                field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner,
                storage_options=None,
                **write_options
            )
            logging.debug(f"Writing selected events from {suffix} to output ...")
        except MemoryError:
            logging.exception(f"MemoryError encountered in writing outputs with ak_to_root() for file index {suffix}.")
            return 1
        except Exception as e:
            logging.exception(f"Error encountered in writing outputs with ak_to_root() for file index {suffix}: {e}")
    else:
        raise TypeError("computed_array is not a valid type")
    return 0

def compute_and_write_skimmed(passed, outdir, dataset, suffix, write_args={}) -> int:
    """Compute and write the skimmed events to a file."""
    computed_array = compute_dask_array(passed)
    if computed_array is not None:
        return write_computed_array(computed_array, outdir, dataset, suffix, write_args)
    else:
        write_empty_root(pjoin(outdir, f'{dataset}_{suffix}_empty.root'))
        return 0