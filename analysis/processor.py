# This file contains the Processor class, which is used to process individual files or filesets.
# The behavior of the Processor class is highly dependent on run time configurations and the event selection class used.
import uproot._util
import uproot, pickle, gc
from uproot.writing._dask_write import ak_to_root
import pandas as pd
import dask_awkward as dak
import dask
import awkward as ak
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from src.utils.filesysutil import FileSysHelper, pjoin, XRootDHelper
from src.analysis.evtselutil import BaseEventSelections

class Processor:
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
        self.copy_queue = Queue(maxsize=2)
    
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
            print(f"Loading {filename}")
            if not filename.endswith(":Events"):
                filename += ":Events"
            events = uproot.open(filename, **kwargs).arrays(
                filter_name=self.rtcfg.get("FILTER_NAME", None)
            )
        return events

    def runfiles(self, write_npz=False, **kwargs):
        """Run test selections on file dictionaries.

        Parameters
        - write_npz: if write cutflow out
        
        Returns
        - number of failed files
        """
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
                print(f"Error encountered for file index {suffix} in {self.dataset}: {e}")
                rc += 1
                gc.collect()
            if not remote_load: self.filehelper.remove_files(self.copydir)
        return rc
    
    def writeCF(self, suffix, **kwargs) -> int:
        """Write the cutflow to a file. Transfer the file if necessary"""
        if kwargs.get('write_npz', False):
            npzname = pjoin(self.outdir, f'cutflow_{suffix}.npz')
            self.evtsel.cfobj.to_npz(npzname)
        cutflow_name = f'{self.dataset}_{suffix}_cutflow.csv'
        cutflow_df = self.evtsel.cf_to_df() 
        cutflow_df.to_csv(pjoin(self.outdir, cutflow_name))
        print("Cutflow written to local!")
        if self.transfer is not None:
            self.filehelper.transfer_files(self.outdir, self.transfer, filepattern=cutflow_name, remove=True, overwrite=True)
        return 0
    
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
                    print(f"dask_write encountered error: MemoryError for file index {suffix}.")
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
    
    @staticmethod
    def write_empty(filename):
        """Creates an empty ROOT file as a placeholder."""
        with uproot.recreate(filename):
            pass  # Do nothing, just create an empty file