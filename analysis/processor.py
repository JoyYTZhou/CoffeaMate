# This file contains the Processor class, which is used to process individual files or filesets.
# The behavior of the Processor class is highly dependent on run time configurations and the event selection class used.
import uproot._util
import uproot, pickle, gc, logging, threading, statistics, resource
import pandas as pd
import dask_awkward as dak
import psutil
import awkward as ak
import concurrent.futures
from itertools import islice

from src.utils.filesysutil import FileSysHelper, pjoin, XRootDHelper, release_mapped_memory
from src.analysis.evtselutil import BaseEventSelections
from src.utils.memoryutil import check_and_release_memory, log_memory
from src.utils.ioutil import ak_to_root, parallel_copy_and_load, compute_and_write_skimmed, check_open_files

def infer_fragment_size(files_dict, available_memory) -> int:
    """Infer the fragment size based on the filesize and available memory.
    
    Args:
        available_memory: Available memory in GB (minus overhead)
        mem_per_part: Memory estimated taken by one fragment in GB"""
    logging.debug("Inferring fragment size based on filesize and available memory")
    sample_files = dict(islice(files_dict.items(), 5)) if len(files_dict) > 5 else files_dict
    list_len_steps = []
    list_part_size = []
    for _, fileinfo in sample_files.items():
        list_len_steps.append(len(fileinfo['steps']))
        list_part_size.append(fileinfo['steps'][0][1])

    med_num_part = statistics.median(list_len_steps)
    med_part_size = statistics.median(list_part_size)/1000 # k Events
    logging.debug("Median number of steps: %s", med_num_part)
    logging.debug(f"Median partition size: {med_part_size} k Events")

    mem_per_kevts = 6 # MB
    mem_per_part = mem_per_kevts * med_part_size / 1024 # GB
    logging.debug("Estimating memory per partition: %s GB", mem_per_part)
    logging.debug("Available memory: %s GB", available_memory)
    allowed_parts = available_memory/mem_per_part
    
    frag_size = int(allowed_parts/med_num_part)
    logging.debug("Inferred fragment size: %s", frag_size)

    return frag_size

def calc_skim_params(filesize, avail_memory) -> tuple:
    """Calculate the number of workers and fragment size based on the filesize and available memory."""
    n_workers = avail_memory // (filesize*3)
    n_workers = max(1, n_workers)
    fragment_size = n_workers + 2

    return (n_workers, fragment_size)

def fragment_files(dsdict, fragment_size: int = 2) -> list[dict]:
    """Split files into smaller fragments if needed.
    
    Args:
        fragment_size: Maximum number of files per fragment; If None, will infer from memory size + step size if available
        
    Returns:
        List of dictionaries, each containing a subset of files
    """
    if fragment_size is None:
        available_mem = int(os.environ.get("REQUEST_MEMORY", 22000)) / 1024
        fragment_size = infer_fragment_size(dsdict['files'], available_memory=available_mem)
    
    files = dsdict['files']
    if len(files) <= fragment_size:
        return [dsdict]
        
    fragments = []
    file_items = list(files.items())
    
    for i in range(0, len(file_items), fragment_size):
        fragment_files = dict(file_items[i:i + fragment_size])
        fragment_dict = {
            'files': fragment_files,
        }
        fragments.append(fragment_dict)
    
    logging.info(f"Split {len(files)} files into {len(fragments)} fragments "
                f"of size {fragment_size}")
    return fragments


def dynamic_worker_number(peak_mem_usage, current_mem, current_worker=3, min_workers=1, max_workers=8) -> int:
    """Dynamically adjust the number of workers based on peak memory usage.

    Args:
    - peak_mem_usage (float): Peak memory usage in GB.
    - current_mem (float): Current memory usage in GB.
    - current_worker (int): Current number of workers.
    - min_workers (int): Minimum number of workers.
    - max_workers (int): Maximum number of workers.

    Returns:
    - int: Adjusted number of workers.
    """
    avail_mem = psutil.virtual_memory().available / (1024**3)
    peak_mem_usage_percent = peak_mem_usage / avail_mem
    curr_mem_percent = current_mem / avail_mem

    if curr_mem_percent < 0.2 and peak_mem_usage_percent < 0.7:
        n_worker = min(current_worker + 1, max_workers)
        logging.debug(f"Increasing workers to {n_worker}")
    elif curr_mem_percent > 0.8 or peak_mem_usage_percent > 0.9:
        n_worker = max(current_worker - 1, min_workers)
        logging.debug(f"Decreasing workers to {n_worker}")
    else:
        n_worker = current_worker
    
    return n_worker
    
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
    """Process individual file or filesets given strings/dicts belonging to one dataset."""
    def __init__(self, rtcfg, dsdict, transferP=None, evtselclass=BaseEventSelections, proc_kwargs={}, **kwargs):
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
        self.write_skim_semaphore = threading.Semaphore()
        self.proc_kwargs = proc_kwargs
        self.load_skim_semaphore = threading.Semaphore()
    
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

    def load_for_skims(self, fileargs, executor, readkwargs) -> ak.Array:
        with self.load_skim_semaphore:
            return parallel_copy_and_load(fileargs, self.copydir, executor, self.rtcfg, readkwargs)
    
    def run_skims(self, write_npz=False, frag_threshold=None, readkwargs={}, writekwargs={}, **kwargs) -> int:
        """Process files in parallel. Recommended for skimming."""
        total_files = len(self.dsdict['files'])
        logging.debug(f"Expected to see {total_files} outputs")
        rc = 0
        process = psutil.Process()

        batch_dicts = fragment_files(self.dsdict, frag_threshold)
        frag_size = len(batch_dicts)
        self.load_skim_semaphore = threading.Semaphore(min(frag_size, 4))
        self.write_skim_semaphore = threading.Semaphore(min(frag_size, 4))

        worker_no = 2
            
        for batch_dict in batch_dicts:
            try: 
                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_no) as executor:
                    future_loaded = self.load_for_skims(
                        fileargs=batch_dict,
                        executor=executor,
                        readkwargs=readkwargs)
                    
                    future_cf, future_writes, future_passed = [], [], {}
                    log_memory(process, "before processing")
                    for future in concurrent.futures.as_completed(future_loaded.values()):
                        filename = next(f for f, future in future_loaded.items() if future == future)
                        try: 
                            events, suffix = future.result()
                            future_passed[suffix] = executor.submit(self.evtselclass(**self.evtsel_kwargs).callevtsel, events)
                        except Exception as e:
                            logging.exception(f"Error copying and loading {filename}: {e}")
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
                    for future in concurrent.futures.as_completed(future_writes):
                        rc += future.result()
                        del future
                    del future_cf, future_writes, future_passed, future_loaded
                    log_memory(process, "after computing + writing + garbage collection")
                    if self.transfer:
                        for cutflow_file in cutflow_files:
                            self.filehelper.transfer_files(self.outdir, self.transfer, filepattern=cutflow_file, remove=True)
                        self.filehelper.transfer_files(self.outdir, self.transfer, filepattern=f'{self.dataset}_*.root', remove=True)
                    if not self.rtcfg.get("REMOTE_LOAD", True):
                        self.filehelper.close_open_files_delete(self.copydir, "*.root")
                release_mapped_memory()
                check_open_files()
                check_and_release_memory(process)
            except Exception as e:
                logging.exception(f"Error encountered when processing {self.dataset}: {e}")
            peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
            curr_mem = process.memory_info().rss / (1024**3)
            worker_no = dynamic_worker_number(peak_mem, curr_mem, current_worker=worker_no)
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
    
    def writedask(self, passed: dak.lib.core.Array, suffix, parquet=False) -> int:
        pass
    
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