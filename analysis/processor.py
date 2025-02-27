# This file contains the Processor class, which is used to process individual files or filesets.
# The behavior of the Processor class is highly dependent on run time configurations and the event selection class used.
import uproot._util
import uproot, pickle, gc, logging, threading
import pandas as pd
import dask_awkward as dak
import psutil
import awkward as ak
import concurrent.futures

from src.utils.filesysutil import FileSysHelper, pjoin, XRootDHelper
from src.analysis.evtselutil import BaseEventSelections
from src.utils.testutils import log_memory
from src.utils.ioutil import ak_to_root, parallel_copy_and_load, compute_and_write_skimmed

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
    load_skim_semaphore = threading.Semaphore(3)
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

    def load_for_skims(self, fileargs, executor, readkwargs) -> ak.Array:
        with self.load_skim_semaphore:
            return parallel_copy_and_load(fileargs, self.copydir, executor, self.rtcfg, readkwargs)

    def run_skims(self, write_npz=False, max_workers=2, readkwargs={}, writekwargs={}, **kwargs) -> int:
        """Process files in parallel. Recommended for skimming."""
        logging.debug(f"Expected to see {len(self.dsdict['files'])} outputs")
        rc = 0
        process = psutil.Process()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_loaded = self.load_for_skims(
                fileargs={"files": self.dsdict["files"]},
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

            log_memory(process, "after computing + writing + garbage collection")
        
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