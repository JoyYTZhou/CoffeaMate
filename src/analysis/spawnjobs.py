from dask.distributed import Client, LocalCluster
from dask.distributed import as_completed
import json as json
import gzip, glob, traceback, os
from itertools import islice

from .custom import switch_selections
from .processor import Processor
from src.utils.filesysutil import glob_files, cross_check, checkpath, pjoin
from config.selectionconfig import runsetting as rs
from config.selectionconfig import dasksetting as daskcfg

evtselclass = switch_selections(rs.SEL_NAME)

p_dirname = os.path.dirname

src_dir = p_dirname(p_dirname(os.path.abspath(__file__)))
base_dir = p_dirname(src_dir)
data_dir = pjoin(base_dir, 'data')

datapath = pjoin(data_dir, 'availableQuery.json')
with open(datapath, 'r') as data:
    realmeta = json.load(data)

transferPBase = rs.get("TRANSFER_PATH", None)
if transferPBase is not None: checkpath(transferPBase, createdir=True)

def get_fi_prefix(filepath):
    return os.path.basename(filepath).split('.')[0].split('_')[0]

def div_list(original_list, chunk_size):
    """Divide a list into smaller lists of given size."""
    for i in range(0, len(original_list), chunk_size):
        yield original_list[i:i + chunk_size]

def div_dict(original_dict, chunk_size):
    """Divide a dictionary into smaller dictionaries of given size."""
    it = iter(original_dict.items())
    for _ in range(0, len(original_dict), chunk_size):
        yield dict(islice(it, chunk_size))

def filterExisting(ds: 'str', dsdata: 'dict', outputpattern=".root", tsferP=transferPBase) -> bool:
    """Update dsdata on files that need to be processed for a MC dataset based on the existing output files and cutflow tables.
    
    Parameters
    - `ds`: Dataset name
    - `dsdata`: A dictionary of dataset information with keys 'files', 'metadata', 'filelist'

    Return
    - bool: True if some files need to be processed, False otherwise. 
    """
    if not tsferP or checkpath(tsferP, createdir=False) != 0:
        return True
    
    files_to_remove = [] 

    for filename, fileinfo in dsdata['files'].items():
        prefix = f"{ds}_{fileinfo['uuid']}"
        outputfile = f"{prefix}*{outputpattern}"
        cutflowfile = f"{prefix}_cutflow.csv"
        outputfiles = glob_files(tsferP, '*.root')
        cutflowfiles = glob_files(tsferP, '*cutflow.csv')
        if cross_check(outputfile, outputfiles) and cross_check(cutflowfile, cutflowfiles):
            files_to_remove.append(filename)
        
    for file in files_to_remove:
        dsdata['files'].pop(file)
    
    return len(dsdata['files']) > 0
    
class JobRunner:
    def __init__(self, jobfile, eventSelection=evtselclass) -> None:
        self.selclass = eventSelection
        with open(jobfile, 'r') as job:
            self.loaded = json.load(job)
            grp_name = get_fi_prefix(jobfile)
        self.grp_name = grp_name
        
    def submitjobs(self, client) -> int:
        """Run jobs based on client settings.
        If a valid client is found and future mode is true, submit simultaneously run jobs.
        If not, fall back into a loop mode. Note that even in this mode, any dask computations will be managed by client explicitly or implicitly.
        """
        proc = Processor(rs, self.loaded, f'{transferPBase}/{self.grp_name}', self.selclass)
        rc = proc.runfiles()
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
            proc = Processor(rs, filelist, self.grp_name, transferPBase) 
            rc = proc.runfile(fn, i)
            return rc
        if indx is None:
            futures.extend([client.submit(job, fn, i) for i, fn in enumerate(filelist)])
        else:
            futures.extend([client.submit(job, filelist[i]) for i in indx])
        return futures

class JobLoader():
    """Load meta job files and prepare for processing."""
    def __init__(self, jobpath, datapath=pjoin(data_dir, 'preprocessed')) -> None:
        self.inpath = datapath
        checkpath(self.inpath, createdir=False, raiseError=True)
        self.tsferP = transferPBase
        self.jobpath = jobpath
        checkpath(jobpath)

    def writejobs(self) -> None:
        """Write job parameters to json file"""
        datafile = glob.glob(pjoin(self.inpath, '*.json.gz'))
        for file in datafile:
            self.prepjobs(file)
        
    def prepjobs(self, inputdatap, batch_size=15) -> bool:
        with gzip.open(inputdatap, 'rt') as samplepath:
            grp_name = get_fi_prefix(inputdatap)
            loaded = json.load(samplepath)
        for ds, dsdata in loaded.items():
            shortname = dsdata['metadata']['shortname']
            print(f"===============Preparing job files for {ds}========================")
            need_process = filterExisting(shortname, dsdata, tsferP=pjoin(self.tsferP, grp_name))
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

def spawnLocal():
    """Spawn dask client for local cluster"""
    # size_mb = size/1024/1024
    cluster = LocalCluster(processes=daskcfg.get('SPAWN_PROCESS', False), threads_per_worker=daskcfg.get('THREADS_NO', 2))
    cluster.adapt(minimum=1, maximum=4)
    client = Client(cluster)
    print("successfully created a dask client in local cluster!")
    print("===================================")
    print(client)
    return client

def spawnclient(default=False):
    """Spawn appropriate client based on runsetting."""
    if not daskcfg.SPAWN_CONDOR:
        client = spawnLocal()
    else:
        client = spawnCondor(default)
    return client 

def spawnCondor(default=False):
    """Spawn dask client for condor cluster"""
    from lpcjobqueue import LPCCondorCluster

    print("Trying to submit jobs to condor via dask!")

    if default:
        cluster = LPCCondorCluster(ship_env=True)
        cluster.adapt(maximum=3)
        print(cluster.job_script())
    else:
        condor_args = {"ship_env": True, 
                    "processes": daskcfg.PROCESS_NO,
                    "cores": daskcfg.CORE_NO,
                    "memory": daskcfg.MEMORY,
                    "disk": daskcfg.DISK
                    }
        cluster = LPCCondorCluster(**condor_args)
        cluster.job_extra_directives = {
            'output': 'dask_output.$(ClusterId).$(ProcId).out',
            'error': 'daskr_error.$(ClusterId).$(ProcId).err',
            'log': 'dask_log.$(ClusterId).log',
        }
        cluster.adapt(minimum=daskcfg.MIN_WORKER, maximum=daskcfg.MAX_WORKER)
        print(cluster.job_script())

    client = Client(cluster)
    print("One client created in LPC Condor!")
    print("===================================")
    print(client)

    return client

