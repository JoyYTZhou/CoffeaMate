# This file contains the Processor class, which is used to process individual files or filesets.
# The behavior of the Processor class is highly dependent on run time configurations and the event selection class used.
import uproot._util
import uproot, pickle, gc, logging, threading
from uproot.writing._dask_write import ak_to_root
import pandas as pd
import dask_awkward as dak
import dask, psutil
import awkward as ak
import concurrent.futures

from src.utils.filesysutil import FileSysHelper, pjoin, XRootDHelper
from src.analysis.evtselutil import BaseEventSelections
from src.utils.testutils import log_memory

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

def compute_dask_array(passed) -> ak.Array:
    """Compute the dask array and handle zero-length partitions."""
    if hasattr(passed, 'npartitions'):
        passed = passed.persist()

        length_calcs = [dask.delayed(len)(passed.partitions[i]) for i in range(passed.npartitions)]
        lengths = dask.compute(*length_calcs)

        has_zero_lengths = any(l == 0 for l in lengths)

        if not has_zero_lengths:
            logging.debug("No zero-arrays found, using uproot.dask_write directly")
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

def write_dask_array(computed_array, outdir, dataset, suffix, write_args={}) -> int:
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
        return write_dask_array(computed_array, outdir, dataset, suffix, write_args)
    else:
        write_empty_root(pjoin(outdir, f'{dataset}_{suffix}.root'))
        return 0

def writeCF(evtsel, suffix, outdir, dataset, write_npz=False) -> str:
    """Write the cutflow to a file. 
    
    Return 
    - the name of the cutflow file"""
    if write_npz:
        npzname = pjoin(outdir, f'cutflow_{suffix}.npz')
        evtsel.cfobj.to_npz(npzname)
    cutflow_name = f'{dataset}_{suffix}_cutflow.csv'
    cutflow_df = evtsel.cf_to_df() 
    output_name = pjoin(outdir, cutflow_name)
    cutflow_df.to_csv(output_name)
    logging.debug("Cutflow written to %s", output_name)
    return cutflow_name

class Processor:
    write_skim_semaphore = threading.Semaphore(3)
    """Process individual file or filesets given strings/dicts belonging to one dataset."""
    def __init__(self, rtcfg, dsdict, transferP=None, evtselclass=BaseEventSelections, **kwargs):
        """
        Parameters
        - `ds_dict`: Example dictionary should look like this,
        {"files": {"file1.root": {"steps": [...], "uuid": ...}}, "metadata": {"shortname": ...}}
        """
        self.rtcfg = rtcfg
        self.dsdict = dsdict
        self.dataset = dsdict['metadata']['shortname']
        self.evtsel_kwargs = kwargs
        self.evtselclass = evtselclass
        self.transfer = transferP
        self.filehelper = FileSysHelper()
        self.initdir()
    
    def initdir(self) -> None:
        """Initialize the output directory and copy directory if necessary.
        If the copy directory is specified, it will be created and checked.
        The output directory will be checked and created if necessary."""
        self.outdir = pjoin(self.rtcfg.get("OUTPUTDIR_PATH", "outputs"), self.dataset)
        self.copydir = self.rtcfg.get("COPYDIR_PATH", "copydir")
        self.filehelper.checkpath(self.outdir)
        self.filehelper.checkpath(self.copydir)
    
    def loadfile(self, fileargs: dict, copy_local: bool = False, **kwargs) -> ak.Array:
        """Load a ROOT file either directly or after copying locally.

        Parameters
        - fileargs: {"files": {filename1: fileinfo1}, ...}
        - copy_local: if True, copy the file locally before loading
        - kwargs: additional arguments passed to uproot.open() or uproot.dask()

        Returns
        - events as an awkward array
        """
        filename = list(fileargs['files'].keys())[0]
        suffix = fileargs['files'][filename]['uuid']

        if copy_local:
            if filename.endswith(":Events"):
                filename = filename.split(":Events")[0]
            XRootDHelper.copy_local(filename, pjoin(self.copydir, f"{suffix}.root"))
            filename = pjoin(self.copydir, f"{suffix}.root")
            fileargs = {"files": {filename: fileargs['files'][list(fileargs['files'].keys())[0]]}}

        delayed_open = self.rtcfg.get("DELAYED_OPEN", True)
        if delayed_open:
            events = uproot.dask(**fileargs, **kwargs)
        else:
            logging.debug("Loading %s", filename)
            if not filename.endswith(":Events"):
                filename += ":Events"
            events = uproot.open(filename, **kwargs).arrays(
                filter_name=self.rtcfg.get("FILTER_NAME", None)
            )
        return events

    def run_skims(self, write_npz=False, max_workers=2, readkwargs={}, writekwargs={}, **kwargs) -> int:
        """Process files in parallel. Recommended for skimming."""
        logging.debug(f"Expected to see {len(self.dsdict['files'])} outputs")
        rc = 0
        process = psutil.Process()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_loaded = parallel_copy_and_load(
                fileargs={"files": self.dsdict["files"]},
                copydir=self.copydir,
                executor=executor,
                rtcfg=self.rtcfg,
                read_args=readkwargs)
            
            future_cf, future_writes, future_passed = [], [], {}

            log_memory(process, "before processing")
            
            for future in concurrent.futures.as_completed(future_loaded.values()):
                filename = next(f for f, future in future_loaded.items() if future == future)
                
                try: 
                    events, suffix = future.result()
                    future_passed[suffix] = executor.submit(self.evtselclass(**self.evtsel_kwargs).callevtsel, events)
                except Exception as e:
                    logging.exception(f"Error copying and loading {filename}: {e}")
                    gc.collect()
            
            log_memory(process, "after loading")
                
            for future in concurrent.futures.as_completed(future_passed.values()):
                suffix = next(s for s, f in future_passed.items() if f == future)

                try:
                    log_memory(process, f"before computing/writing {suffix}")
                    passed, evtsel_state = future.result()

                    future_cf.append(executor.submit(writeCF, evtsel_state, suffix, self.outdir, self.dataset))
                    future_writes.append(executor.submit(self.writeskimmed, passed, suffix, **writekwargs))
                except Exception as e:
                    logging.exception(f"Error processing {suffix}: {e}")
            
            cutflow_files = []
            for future in concurrent.futures.as_completed(future_cf):
                cutflow_files.append(future.result())
                del future
                gc.collect()
            
            for future in concurrent.futures.as_completed(future_writes):
                rc += future.result()
                del future
                gc.collect()
            
            del future_cf, future_writes, future_passed, future_loaded
            gc.collect()

            log_memory(process, "after computing + writing")
        
            if self.transfer:
                for cutflow_file in cutflow_files:
                    self.filehelper.transfer_files(self.outdir, self.transfer, filepattern=cutflow_file, remove=True)
                self.filehelper.transfer_files(self.outdir, self.transfer, filepattern=f'{self.dataset}_*.root', remove=True)

            if not self.rtcfg.get("REMOTE_LOAD", True):
                self.filehelper.remove_files(self.copydir)
        return rc
    
    def runfiles_sequential(self, write_npz=False, **kwargs) -> int:
        """Process files sequentially."""
        print(f"Expected to see {len(self.dsdict['files'])} outputs")
        rc = 0
        for filename, fileinfo in self.dsdict["files"].items():
            print(filename)
            try:
                suffix = fileinfo['uuid']
                self.evtsel = self.evtselclass(**self.evtsel_kwargs)
                remote_load = self.rtcfg.get("REMOTE_LOAD", True)
                events = self.loadfile_remote(fileargs={"files": {filename: fileinfo}}) if remote_load else self.loadfile_local(fileargs={"files": {filename: fileinfo}})
                if events is not None: 
                    events = self.evtsel(events)
                    self.writeCF(suffix, write_npz=write_npz)
                    self.writeevts(events, suffix, **kwargs)
                else:
                    rc += 1
                del events
            except Exception as e:
                logging.exception(f"Error encountered when sequentially processing file index {suffix} in {self.dataset}: {e}")
                rc += 1
                gc.collect()
            if not remote_load: self.filehelper.remove_files(self.copydir)
        return rc
    
    def writeevts(self, passed, suffix, **kwargs) -> int:
        """Write the events to a file, filename formated as {dataset}_{suffix}*."""
        output_format = self.rtcfg.get("OUTPUT_FORMAT", None)
        print(type(passed))
        if isinstance(passed, dak.lib.core.Array):
            if_parquet = (output_format == 'parquet')
            rc = self.writedask(passed, suffix, parquet=if_parquet)
        elif isinstance(passed, ak.Array):
            rc = self.writeak(passed, suffix)
        elif isinstance(passed, pd.DataFrame):
            rc = self.writedf(passed, suffix)
        else:
            rc = self.writepickle(passed, suffix, **kwargs)

        if self.transfer is not None:
            self.filehelper.transfer_files(self.outdir, self.transfer, filepattern=f'{self.dataset}_{suffix}*', remove=True)
            print(f"Files transferred to: {self.transfer}" )
        return rc
    
    def writeskimmed(self, passed, suffix, **kwargs) -> int:
        with self.write_skim_semaphore:
            return compute_and_write_skimmed(passed, self.outdir, self.dataset, suffix, **kwargs)

    def writedask(self, passed, suffix, parquet=False, fields=None) -> int:
        """Wrapper around uproot.dask_write(),
        transfer all root files generated to a destination location.
        
        Parameters:
        - `parquet`: if True, write to parquet instead of root"""
        rc = 0
        delayed = self.rtcfg.get("DELAYED_WRITE", False)
        if not parquet:
            if delayed: uproot.dask_write(passed, destination=self.outdir, tree_name="Events", compute=False, prefix=f'{self.dataset}_{suffix}')
            else: 
                try:
                    uproot.dask_write(passed, destination=self.outdir, tree_name="Events", compute=True, prefix=f'{self.dataset}_{suffix}')
                except MemoryError:
                    logging.exception(f"MemoryError encountered in writing outputs with dask_write() for file index {suffix}.")
                    rc = 1
        else:
            dak.to_parquet(passed, destination=self.outdir, prefix=f'{self.dataset}_{suffix}')
        return rc
    
    def writeak(self, passed: 'ak.Array', suffix, fields=None) -> int:
        """Writes an awkward array to a root file. Wrapper around ak_to_root."""
        rc = 0
        outputname = pjoin(self.outdir, f'{self.dataset}_{suffix}.root') 
        if len(passed) == 0:
            print(f"Warning: No events passed the selection, writing an empty placeholder ROOT file {outputname}")
            self.write_empty(outputname)
        if fields is None:
            ak_to_root(outputname, passed, tree_name='Events',
                       counter_name=lambda counted: 'n' + counted, 
                       field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner,
                       storage_options=None, compression="ZLIB", compression_level=1, title="", initial_basket_capacity=50, resize_factor=5)
    
    def writedf(self, passed: pd.DataFrame, suffix) -> int:
        """Writes a pandas DataFrame to a csv file.
        
        Parameters:
        - `passed`: DataFrame to write
        - `suffix`: index to append to filename"""
        outname = pjoin(self.outdir, f'{self.dataset}_{suffix}_output.csv')
        passed.to_csv(outname)
        return 0
        
    def writepickle(self, passed, suffix):
        """Writes results to pkl. No constraints on events type."""
        finame = pjoin(self.outdir, f"{self.dataset}_{suffix}.pkl")
        with open(finame, 'wb') as f:
            pickle.dump(passed, f)
        return 0