# This file contains the Processor class, which is used to process individual files or filesets.
# The behavior of the Processor class is highly dependent on run time configurations and the event selection class used.
import pickle, gc, logging, threading, statistics, resource, os
import pandas as pd
import psutil
import awkward as ak
import concurrent.futures
from itertools import islice

from src.utils.filesysutil import FileSysHelper, pjoin, release_mapped_memory
from src.analysis.evtselutil import BaseEventSelections
from src.utils.memoryutil import check_and_release_memory, log_memory, dynamic_worker_number
from src.utils.ioutil import parallel_process_files, compute_and_write_skimmed, check_open_files

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

    mem_per_kevts = 8 # MB
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
    def __init__(self, rtcfg, dsdict, transferP=None, evtselclass=BaseEventSelections, proc_kwargs={}):
        """
        Parameters
        - `ds_dict`: Example dictionary should look like this,
        {"files": {"file1.root": {"steps": [...], "uuid": ...}}, "metadata": {"shortname": ...}}
        """
        self.rtcfg = rtcfg
        self.dsdict = dsdict
        self.dataset = dsdict['metadata']['shortname']
        self._ismc = dsdict['metadata']['is_mc']
        self.evtselclass = evtselclass
        self.transfer = transferP
        self.filehelper = FileSysHelper()
        self.__initdir()
        self.proc_kwargs = proc_kwargs
    
    def __initdir(self) -> None:
        """Initialize the output directory and copy directory if necessary.
        If the copy directory is specified, it will be created and checked.
        The output directory will be checked and created if necessary."""
        self.outdir = pjoin(self.rtcfg.get("OUTPUTDIR_PATH", "outputs"), self.dataset)
        self.copydir = self.rtcfg.get("COPYDIR_PATH", "copydir")
        self.filehelper.checkpath(self.outdir)
        self.filehelper.checkpath(self.copydir)
    
    def _load_files(self, fileargs, executor, uproot_args={}) -> ak.Array:
        raise NotImplementedError("Processor._load_files() must be implemented in a subclass.")
    
    def _write_events(self, passed, suffix, **kwargs) -> int:
        raise NotImplementedError("Processor._write_events() must be implemented in a subclass.")
    
    def _process_event_selections(self, future_loaded, executor) -> dict:
        """Process event selections for loaded files.
        Args:
            future_loaded: Dictionary of futures containing loaded events
            executor: ThreadPoolExecutor instance
        Returns:
            Dictionary containing futures for passed events and their corresponding suffixes
        """
        future_passed = {}
        for future in concurrent.futures.as_completed(future_loaded.values()):
            filename = next(f for f, future in future_loaded.items() if future == future)
            try:
                events, suffix = future.result()
                future_passed[suffix] = executor.submit(self.evtselclass(is_mc=self._ismc).callevtsel, events)
            except Exception as e:
                logging.exception(f"Error copying and loading {filename}: {e}")
        return future_passed

    def _collect_results(self, future_passed, executor, process, writekwargs) -> tuple[int, list]:
        """Collect and write results from event selections.

        Args:
            future_passed: Dictionary of futures containing passed events
            executor: ThreadPoolExecutor instance
            process: psutil Process object for memory monitoring
            writekwargs: Arguments for writing events

        Returns:
            Tuple of (return code, list of cutflow files)
        """
        rc = 0
        future_cf, future_writes = [], []
        cutflow_files = []
        for future in concurrent.futures.as_completed(future_passed.values()):
            suffix = next(s for s, f in future_passed.items() if f == future)
            try:
                log_memory(process, f"before computing/writing {suffix}")
                passed, evtsel_state = future.result()
                future_cf.append(executor.submit(writeCF, evtsel_state, suffix, self.outdir, self.dataset))
                future_writes.append(executor.submit(self._write_events, passed, suffix, **writekwargs))
            except Exception as e:
                logging.exception(f"Error processing file {suffix}: {e}")

        for future in concurrent.futures.as_completed(future_cf + future_writes):
            if future in future_cf:
                cutflow_files.append(future.result())
            else:
                rc += future.result()
            del future

        return rc, cutflow_files
    
    def _process_batch(self, batch_dict, worker_no, process, output_pattern, readkwargs, writekwargs) -> tuple[int, int]:
        """Process a single batch of files.

        Args:
            batch_dict: Dictionary containing files to process
            worker_no: Number of workers to use
            process: psutil.Process object for memory monitoring
            output_pattern: Pattern for output files
            readkwargs: Arguments for reading files
            writekwargs: Arguments for writing files

        Returns:
            Tuple of (return code, new worker number)
        """
        rc = 0
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_no) as executor:
                future_loaded = self._load_files(
                    fileargs=batch_dict,
                    executor=executor,
                    uproot_args=readkwargs)

                log_memory(process, "before processing")

                # Process event selections
                future_passed = self._process_event_selections(future_loaded, executor)

                # Collect and write results
                rc, cutflow_files = self._collect_results(future_passed, executor, process, writekwargs)

                del future_loaded, future_passed
                log_memory(process, "after computing + writing + garbage collection")
                
                self._transfer_and_cleanup(cutflow_files, output_pattern=output_pattern)

            self.filehelper.close_open_files_delete(self.copydir, "*.root")
            release_mapped_memory()
            check_open_files()
            check_and_release_memory(process)

        except Exception as e:
            logging.exception(f"Error encountered when processing {self.dataset}: {e}")

        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
        curr_mem = process.memory_info().rss / (1024**3)
        new_worker_no = dynamic_worker_number(peak_mem, curr_mem, current_worker=worker_no)

        return rc, new_worker_no

    def _transfer_and_cleanup(self, cutflow_files, output_pattern):
        """Transfer output files and clean up temporary files.

        Args:
            cutflow_files: List of cutflow file names to transfer
            output_pattern: Pattern for output files to transfer
        """
        if self.transfer:
            for cutflow_file in cutflow_files:
                self.filehelper.transfer_files(
                    self.outdir,
                    self.transfer,
                    filepattern=cutflow_file,
                    remove=True,
                    overwrite=True
                )
            self.filehelper.transfer_files(
                self.outdir,
                self.transfer,
                filepattern=output_pattern,
                remove=True,
                overwrite=True
            )
 
    def run(self, *args, **kwargs) -> int:
        raise NotImplementedError("Processor.run() must be implemented in a subclass.")

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
    
    def _write_df(self, passed: pd.DataFrame, suffix) -> int:
        """Writes a pandas DataFrame to a csv file.
        
        Parameters:
        - `passed`: DataFrame to write
        - `suffix`: index to append to filename"""
        outname = pjoin(self.outdir, f'{self.dataset}_{suffix}_output.csv')
        passed.to_csv(outname)
        return 0

    def _write_pkl(self, passed, suffix):
        """Writes results to pkl. No constraints on events type."""
        finame = pjoin(self.outdir, f"{self.dataset}_{suffix}.pkl")
        with open(finame, 'wb') as f:
            pickle.dump(passed, f)
        return 0 

class PreselProcessor(Processor):
    """Processor for preselections."""
    def __init__(self, rtcfg, dsdict, transferP=None, evtselclass=BaseEventSelections, proc_kwargs={}):
        super().__init__(rtcfg, dsdict, transferP, evtselclass, proc_kwargs)
        self._delayed_open = self.rtcfg.get("DELAYED_OPEN", True)
        if self._delayed_open:
            logging.info("Using delayed open for files")
        else:
            logging.info("Using normal file open")
    
    def _load_files(self, fileargs, executor, uproot_args={}) -> dict:
        return parallel_process_files(fileargs, executor, None, self._delayed_open, uproot_args)
    
    def _write_events(self, passed, suffix, **kwargs) -> int:
        if isinstance(passed, pd.DataFrame):
            return self._write_df(passed, suffix)
        else:
            logging.error("Unsupported output type")
            return 1
    
    def run(self, readkwargs={}, writekwargs={}) -> int:
        """Process files in parallel. Recommended for preselection."""
        total_files = len(self.dsdict['files'])
        logging.debug(f"Expected to see {total_files} outputs")
        rc = 0
        process = psutil.Process()
       
        cpu_no = psutil.cpu_count(logical=False) 
        worker_no = cpu_no
        logging.debug(f"Using {worker_no} workers")

        output_pattern = "{self.dataset}_*output.csv"
        rc = self._process_batch(self.dsdict, worker_no, process, output_pattern, readkwargs, writekwargs)
        return rc
    
class SkimProcessor(Processor):
    def __init__(self, rtcfg, dsdict, transferP=None, evtselclass=BaseEventSelections, proc_kwargs={}):
        super().__init__(rtcfg, dsdict, transferP, evtselclass, proc_kwargs)
        self._write_semaphore = threading.Semaphore()
        self._load_semaphore = threading.Semaphore()
    
    def _load_files(self, fileargs, executor, uproot_args={}) -> dict:
        with self._load_semaphore:
            return parallel_process_files(fileargs, executor, self.copydir, True, uproot_args)
    
    def run(self, write_npz=False, frag_threshold=None, readkwargs={}, writekwargs={}, **kwargs) -> int:
        """Process files in parallel. Recommended for skimming."""
        total_files = len(self.dsdict['files'])
        logging.debug(f"Expected to see {total_files} outputs")
        rc = 0
        process = psutil.Process()

        batch_dicts = fragment_files(self.dsdict, frag_threshold)
        frag_size = len(batch_dicts)
        self._load_semaphore = threading.Semaphore(min(frag_size, 1))
        self._write_semaphore = threading.Semaphore(min(frag_size, 1))

        worker_no = 1

        for batch_dict in batch_dicts:
            output_pattern = f'{self.dataset}_*.root'
            batch_rc, worker_no = self._process_batch(batch_dict, worker_no, process, output_pattern, readkwargs, writekwargs)
            rc += batch_rc
        return rc

    def _write_events(self, passed, suffix, **kwargs) -> int:
        with self._write_semaphore:
            return compute_and_write_skimmed(passed, self.outdir, self.dataset, suffix, **kwargs)
    
