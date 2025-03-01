import uproot, dask, logging
import concurrent.futures
import awkward as ak
import dask_awkward as dak
from uproot.writing._dask_write import ak_to_root

from src.utils.filesysutil import XRootDHelper, pjoin

def write_empty_root(filename):
    """Creates an empty ROOT file as a placeholder."""
    with uproot.recreate(filename):
        pass
        
def process_file(filename, fileinfo, copydir, rtcfg, read_args) -> tuple:
    """Handles file copying and loading"""
    suffix = fileinfo['uuid']
    dest_file = pjoin(copydir, f"{suffix}.root")
    
    logging.debug(f"Copying and loading {filename} to {dest_file}")
    XRootDHelper.copy_local(filename, dest_file)
    
    delayed_open = rtcfg.get("DELAYED_OPEN", True)
    if delayed_open:
        events = uproot.dask(files={dest_file: fileinfo}, **read_args)
        
        if hasattr(events, 'npartitions'):
            nparts = events.npartitions
            logging.debug(f"File {suffix} has {nparts} partitions")
            
            if nparts <= 8:  # You can adjust this threshold
                logging.debug("Not persisting events")
                # logging.debug(f"Persisting events for {suffix} ({nparts} partitions)")
                # events = events.persist()
            else:
                logging.debug(f"Skipping persist for {suffix} due to high partition count ({nparts})")
        
        return (events, suffix)
    else:
        return (uproot.open(dest_file + ":Events").arrays(**read_args), suffix)

def submit_copy_and_load(fileargs, copydir, rtcfg, read_args, max_workers=3) -> list:
    """Runs file copying and loading in parallel"""
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = parallel_copy_and_load(fileargs, copydir, executor, rtcfg, read_args)
        
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                results.append(future.result())
            except Exception as e:
                logging.exception(f"Error copying and loading {future_to_file[future]}: {e}")
    return results

def parallel_copy_and_load(fileargs, copydir, executor, rtcfg, read_args) -> dict:
    """Runs file copying and loading in parallel"""
    future_to_file = {filename: executor.submit(process_file, filename, fileinfo, copydir, rtcfg, read_args) for filename, fileinfo in fileargs['files'].items()}
    return future_to_file

def compute_dask_array(passed, force_compute=True) -> ak.Array:
    """Compute the dask array and handle zero-length partitions."""
    if hasattr(passed, 'npartitions'):
        passed = passed.persist()

        length_calcs = [dask.delayed(len)(passed.partitions[i]) for i in range(passed.npartitions)]
        lengths = dask.compute(*length_calcs)

        has_zero_lengths = any(l == 0 for l in lengths)

        if not has_zero_lengths:
            logging.debug("No zero-arrays found, using uproot.dask_write directly")
            if force_compute:
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
        write_empty_root(pjoin(outdir, f'{dataset}_{suffix}.root'))
        return 0