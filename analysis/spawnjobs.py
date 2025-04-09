from dask.distributed import as_completed
import json as json
import gzip, glob, traceback, os
from itertools import islice
from rich.console import Console
from rich.table import Table

from src.analysis.processor import Processor
from src.utils.filesysutil import FileSysHelper, pjoin
from src.utils.memoryutil import limit_memory_usage

def read_pattern(filepath):
    basename = os.path.basename(filepath).split('.')[0]
    keys = basename.split('_')
    return keys

def div_dict(original_dict, chunk_size):
    """Divide a dictionary into smaller dictionaries of given size."""
    it = iter(original_dict.items())
    for _ in range(0, len(original_dict), chunk_size):
        yield dict(islice(it, chunk_size))

def filterExisting(ds: str, dsdata: dict, tsferP, out_endpattern=[".root", "cutflow.csv"], prefix=None) -> bool:
    """Update dsdata on files that need to be processed for a MC dataset 
    based on the existing output files and cutflow tables.
    
    Parameters:
    - `ds`: Dataset name
    - `dsdata`: A dictionary of dataset information with keys 'files', 'metadata', 'filelist'
    - `tsferP`: Path where processed files are stored.
    - `out_endpattern`: A string or a list of strings representing output ending file pattern to check for.
    - `prefix`: Prefix to be used in case of xrootd system for tsferP. Treat tsferP as a local path if None.

    Returns:
    - bool: True if some files need to be processed, False otherwise. 
    """
    
    helper = FileSysHelper()
    dir_exist = helper.checkpath(tsferP, createdir=False)
    
    if not tsferP or not dir_exist:
        return True
    
    if isinstance(out_endpattern, str):
        out_endpattern = [out_endpattern]  # Convert to list if single string

    files_to_remove = [] 

    existing_outfiles = helper.glob_files(tsferP, "*")  # Get all files in tsferP once

    file_match_counts = {filename: 0 for filename in dsdata['files']}

    for filename, fileinfo in dsdata['files'].items():
        expected_prefix = f"{ds}_{fileinfo['uuid']}"

        for pattern in out_endpattern:
            expected_file = f"{expected_prefix}*{pattern}"  # Expected file pattern
            if helper.cross_check(expected_file, existing_outfiles):  
                file_match_counts[filename] += 1  # Increase match count

        if file_match_counts[filename] == len(out_endpattern):
            files_to_remove.append(filename)

    for file in files_to_remove:
        dsdata['files'].pop(file)

    return len(dsdata['files']) > 0  # Return True if files remain, False if all processed
    
class JobRunner:
    """
    Attributes
    - `selclass`: Event selection class. Derived from BaseEventSelections"""
    def __init__(self, runsetting, jobfile, eventSelection, procClass, dasksetting=None):
        """Initialize the job runner with the job file and event selection class.
        
        Parameters
        - `runsetting`: Run settings, a dynaconf object/dictionary
        - `jobfile`: Job file path
        - `eventSelection`: Event selection class"""
        self.selclass = eventSelection
        self.procClass = procClass
        self.rs = runsetting 
        with open(jobfile, 'r') as job:
            self.loaded = json.load(job)
            keys = read_pattern(jobfile)
            self.grp_name = keys[0]
            year = keys[1]
        self.transferPBase = self.rs.get("TRANSFER_PATH", None)
        if self.transferPBase is not None:
            helper = FileSysHelper()
            helper.checkpath(self.transferPBase, createdir=True)
            self.transferPBase = f'{self.transferPBase}/{year}'
            helper.checkpath(self.transferPBase, createdir=True)
    
    def submitjobs(self, client, proc_kwargs={}) -> int:
        """Run jobs based on client settings.
        If a valid client is found and future mode is true, submit simultaneously run jobs.
        If not, fall back into a loop mode. Note that even in this mode, any dask computations will be managed by client explicitly or implicitly.

        Parameters
        - `kwargs`: Additional keyword arguments to be passed to the processor.writeevts() method.
        """
        proc = self.procClass(self.rs, self.loaded, f'{self.transferPBase}/{self.grp_name}', self.selclass, proc_kwargs=proc_kwargs)
        read_kwargs = {}
        filter_name = self.rs.get("FILTER_NAME", None)
        if filter_name:
            read_kwargs = {"filter_name": filter_name}
        rc = proc.run(readkwargs=read_kwargs)
        return rc
    
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
    def __init__(self, datapath, kwd, jobpath, transferPBase, out_endpattern) -> None:
        """Initialize the job loader.
        
        Parameters
        - `datapath`: directory path from which the json.zp files containing dataset information will be grepped.
        - `kwd`: kwd to grep from the job file, should take the form of {groupname}_{year}
        - `jobpath`: Path to one job file in json format
        - `transferPBase`: Path to which root/cutflow output files of the selections will be ultimately transferred."""
        self.inpath = datapath
        self.kwd = kwd
        self.helper = FileSysHelper()
        self.helper.checkpath(self.inpath, createdir=False, raiseError=True)
        self.tsferP = transferPBase
        self.jobpath = jobpath
        self.helper.checkpath(self.jobpath, createdir=True)
        self.out_endpattern = out_endpattern

    def writejobs(self, batch_size=10) -> None:
        """Write job parameters to json file"""
        datafile = glob.glob(pjoin(self.inpath, f'{self.kwd}*json.gz'))
        for file in datafile:
            self.prepjobs_from_dict(file, batch_size=batch_size)
    
    def prepjobs_from_dict(self, inputdatap, batch_size=5) -> bool:
        """Prepare job files from a group dictionary containing datasets and the files. Job files are created in the jobpath,
        with name format: {groupname}_{year}_{shortname}_job_{j}.json"""
        console = Console()
        table = Table(title="Job Preparation Status")
        table.add_column("Dataset", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Job Files", style="yellow")

        with gzip.open(inputdatap, 'rt') as samplepath:
            keys = read_pattern(inputdatap)
            grp_name = keys[0]
            yr = keys[1]
            loaded = json.load(samplepath)

        console.print(f"\n[bold cyan]Processing input file: {inputdatap}[/bold cyan]\n")

        for ds, dsdata in loaded.items():
            shortname = dsdata['metadata']['shortname']
            job_files = []
            need_process = filterExisting(shortname, dsdata, tsferP=pjoin(self.tsferP, yr, grp_name),
                                    out_endpattern=self.out_endpattern)

            if need_process:
                for j, sliced in enumerate(div_dict(dsdata['files'], batch_size)):
                    baby_job = {'metadata': dsdata['metadata'], 'files': sliced}
                    finame = pjoin(self.jobpath, f'{grp_name}_{yr}_{shortname}_job_{j}.json')
                    with open(finame, 'w') as fp:
                        json.dump(baby_job, fp)
                    job_files.append(os.path.basename(finame))
                table.add_row(
                    ds,
                    "[green]Needs Processing[/green]",
                    job_files[-1]
                )
            else:
                table.add_row(
                    ds,
                    "[blue]All Processed[/blue]",
                    "[dim]No job files needed[/dim]"
                )
        console.print(table)

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

