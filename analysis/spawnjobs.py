from dask.distributed import as_completed
import json as json
import gzip, glob, traceback, os
from itertools import islice

from src.analysis.processor import Processor
from src.utils.filesysutil import FileSysHelper, pjoin

def get_fi_prefix(filepath):
    return os.path.basename(filepath).split('.')[0].split('_')[0]

def div_dict(original_dict, chunk_size):
    """Divide a dictionary into smaller dictionaries of given size."""
    it = iter(original_dict.items())
    for _ in range(0, len(original_dict), chunk_size):
        yield dict(islice(it, chunk_size))

def filterExisting(ds: 'str', dsdata: 'dict', tsferP, out_endpattern=[".root", "cutflow.csv"], prefix=None) -> bool:
    """Update dsdata on files that need to be processed for a MC dataset based on the existing output files and cutflow tables.
    
    Parameters
    - `ds`: Dataset name
    - `dsdata`: A dictionary of dataset information with keys 'files', 'metadata', 'filelist'
    - `out_endpattern`: A string or a list of strings representing output ending file pattern to check for. No wildcards needed.
    - `prefix`: Prefix to be used in case of xrootd system for tsferP. Treat tsferP as a local path if None.

    Return
    - bool: True if some files need to be processed, False otherwise. 
    """
    helper = FileSysHelper()
    dir_exist = helper.checkpath(tsferP, createdir=False)
    if not tsferP or not dir_exist:
        return True
    
    if isinstance(out_endpattern, str):
        outputpattern = [outputpattern]
    
    files_to_remove = [] 

    for filename, fileinfo in dsdata['files'].items():
        prefix = f"{ds}_{fileinfo['uuid']}"
        matched = 0
        for pattern in out_endpattern:
            outfiles = helper.glob_files(tsferP, f"*{pattern}")
            expected = f"{prefix}*{pattern}"
            matched += helper.cross_check(expected, outfiles)
        
        if matched == len(out_endpattern):
            files_to_remove.append(filename)
        
    for file in files_to_remove:
        dsdata['files'].pop(file)
    
    return len(dsdata['files']) > 0
    
class JobRunner:
    """
    Attributes
    - `selclass`: Event selection class. Derived from BaseEventSelections"""
    def __init__(self, runsetting, jobfile, eventSelection, dasksetting=None):
        """Initialize the job runner with the job file and event selection class.
        
        Parameters
        - `runsetting`: Run settings, a dynaconf object/dictionary
        - `jobfile`: Job file path
        - `eventSelection`: Event selection class"""
        self.selclass = eventSelection
        self.rs = runsetting 
        with open(jobfile, 'r') as job:
            self.loaded = json.load(job)
            grp_name = get_fi_prefix(jobfile)
        self.grp_name = grp_name
        self.transferPBase = self.rs.get("TRANSFER_PATH", None)
        if self.transferPBase is not None:
            helper = FileSysHelper()
            helper.checkpath(self.transferPBase, createdir=True)
        
    def submitjobs(self, client, **kwargs) -> int:
        """Run jobs based on client settings.
        If a valid client is found and future mode is true, submit simultaneously run jobs.
        If not, fall back into a loop mode. Note that even in this mode, any dask computations will be managed by client explicitly or implicitly.

        Parameters
        - `kwargs`: Additional keyword arguments to be passed to the processor.writeevts() method.
        """
        proc = Processor(self.rs, self.loaded, f'{self.transferPBase}/{self.grp_name}', self.selclass)
        rc = proc.runfiles(**kwargs)
        return 0
    
    def submitfutures(self, client, filelist, indx) -> list:
        """Submit jobs as futures to client.
        
        Parameters
        - `client`: Dask client

        Returns
        list: List of futures for each file in the dataset.
        """
        futures = []
        def job(fn, i):
            proc = Processor(filelist, self.grp_name, self.transferPBase) 
            rc = proc.runfile(fn, i)
            return rc
        if indx is None:
            futures.extend([client.submit(job, fn, i) for i, fn in enumerate(filelist)])
        else:
            futures.extend([client.submit(job, filelist[i]) for i in indx])
        return futures

class JobLoader():
    """Load meta job files and prepare for processing by slicing the files into smaller jobs. 
    Job files are created in the jobpath, which can be passed into main.py."""
    def __init__(self, datapath, groupname, jobpath, transferPBase, out_endpattern) -> None:
        """Initialize the job loader.
        
        Parameters
        - `datapath`: directory path from which the json.zp files containing dataset information will be grepped.
        - `groupname`: keyword contained in the json input files to be processed. Example: "TTbar" will grep any .json.gz files starting with TTbar.
        - `jobpath`: Path to one job file in json format
        - `transferPBase`: Path to which root/cutflow output files of the selections will be ultimately transferred."""
        self.jobpath = jobpath
        self.helper = FileSysHelper()
        self.helper.checkpath(datapath, createdir=False, raiseError=True)
        self.helper.checkpath(self.jobpath, createdir=True)
        self.datafile = glob.glob(pjoin(datapath, f'{groupname}*json.gz'))
        raise FileNotFoundError(f"No files found in {datapath} with {groupname} keyword!") if not self.datafile else None
        self.tsferP = transferPBase
        self.out_endpattern = out_endpattern

    def __call__(self) -> None:
        """Dissect the json files and create job files."""
        for file in self.datafile:
            self.prepjobs_from_dict(file)
    
    def prepjobs_from_dict(self, inputdatap, batch_size=10, **kwargs) -> bool:
        """Prepare job files from a group dictionary containing datasets and the files. Job files are created in the jobpath,
        with name format: {groupname}_{shortname}_job_{j}.json"""
        with gzip.open(inputdatap, 'rt') as samplepath:
            grp_name = get_fi_prefix(inputdatap)
            loaded = json.load(samplepath)
        for ds, dsdata in loaded.items():
            shortname = dsdata['metadata']['shortname']
            print(f"===============Preparing job files for {ds}========================")
            need_process = filterExisting(shortname, dsdata, tsferP=pjoin(self.tsferP, grp_name), out_endpattern=self.out_endpattern)
            if need_process:
                for j, sliced in enumerate(div_dict(dsdata['files'], batch_size)):
                    baby_job = {'metadata': dsdata['metadata'], 'files': sliced}
                    finame = pjoin(self.jobpath, f'{grp_name}_{shortname}_job_{j}.json')
                    with open(finame, 'w') as fp:
                        json.dump(baby_job, fp)
                    print("Job file created: ", finame)
            else:
                print(f"All the files have been processed for {ds}! No job files are needed!")
    
def process_futures(futures, results_file='futureresult.txt', errors_file='futureerror.txt'):
    """Process a list of Dask futures.
    :param futures: List of futures returned by client.submit()
    :return: A list of results from successfully completed futures.
    """
    processed_results = []
    errors = []
    for future in as_completed(futures):
        try:
            if future.exception():
                error_msg = f"An error occurred: {future.exception()}"
                print(future.traceback())
                print(f"Traceback: {traceback.extract_tb(future.traceback())}")
                print(error_msg)
                errors.append(error_msg)
            else:
                result = future.result()
                processed_results.append(result)
        except Exception as e:
            error_msg = f"Error processing future result: {e}"
            print(error_msg)
            errors.append(error_msg)
    with open(results_file, 'w') as f:
        for result in processed_results:
            f.write(str(result) + '\n')
    if errors:
        with open(errors_file, 'w') as f:
            for error in errors:
                f.write(str(error) + '\n')
    return processed_results, errors

